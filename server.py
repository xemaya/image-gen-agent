"""image-gen-agent — a2hmarket shop agent: prompt → OpenRouter image model → PNG.

Flow per /chat turn:
1. Parse ChatRequest, take the latest buyer text as the image prompt.
2. POST to OpenRouter /chat/completions with modalities=["image","text"].
3. Read the returned data URL from choices[0].message.images[0].image_url.url,
   strip the `data:image/png;base64,` prefix, decode.
4. Presign a `chatfile/image` upload via findu-oss, PUT the PNG to S3.
5. Emit `ui("show_file", ...)` with the public URL, then `done()`.

No orders, no payment — free generation. Single tool: `a2h.file.upload`.
"""

from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache

import boto3
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from a2h_agent import (
    ChatRequest,
    done,
    error,
    text,
    ui,
)


LOG = logging.getLogger("image_gen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


MODEL_ID = os.environ.get("IMAGE_GEN_MODEL", "openai/gpt-5.4-image-2")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
OPENROUTER_KEY_PARAM = os.environ.get(
    "OPENROUTER_API_KEY_PARAM", "/a2h/agents/image-gen/openrouter-api-key"
)
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")

# Internal ALB + gateway bypass for findu-oss presign (same pattern as
# zhangxuefeng/renew-life). A2HClient's bearer-token path doesn't cover
# /findu-oss/*, so we bypass-call the ALB directly.
FINDU_ALB = os.environ.get(
    "A2H_FINDU_ALB",
    "http://findu-alb-476446960.us-east-1.elb.amazonaws.com",
).rstrip("/")
SSM_GATEWAY_BYPASS = "/a2h/agents/shared/gateway-bypass-key"


@lru_cache(maxsize=1)
def openrouter_api_key() -> str:
    """Resolve the OpenRouter key: env OPENROUTER_API_KEY first (local dev),
    otherwise pull from SSM once per worker."""
    direct = os.environ.get("OPENROUTER_API_KEY")
    if direct:
        return direct
    ssm = boto3.client("ssm", region_name=AWS_REGION)
    resp = ssm.get_parameter(Name=OPENROUTER_KEY_PARAM, WithDecryption=True)
    return resp["Parameter"]["Value"]


@lru_cache(maxsize=1)
def gateway_bypass_key() -> str:
    """Shared platform bypass key for internal-ALB findu-* calls."""
    direct = os.environ.get("A2H_GATEWAY_BYPASS")
    if direct:
        return direct
    ssm = boto3.client("ssm", region_name=AWS_REGION)
    resp = ssm.get_parameter(Name=SSM_GATEWAY_BYPASS, WithDecryption=True)
    return resp["Parameter"]["Value"]


def seller_id() -> str:
    return os.environ.get("A2H_SELLER_ID", "")


app = FastAPI(title="image-gen-agent", version="1.1.2")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    body = await request.json()
    req = ChatRequest.from_json(body)
    LOG.info(
        "chat session=%s shop=%s event=%s text=%r",
        req.session_id, req.shop_id, req.event_type, req.text[:120],
    )

    async def stream():
        try:
            if req.is_event:
                yield text("图片生成服务已就绪。发我一段 prompt，我就给你画。")
                yield done()
                return

            prompt = req.text.strip()
            if not prompt:
                yield text("请发一段文字描述你想要的画面，我这就给你生成。")
                yield done()
                return

            yield text(f"正在用 {MODEL_ID} 生成图片，请稍候…")

            try:
                png_bytes = await generate_png(prompt)
            except Exception as ex:  # noqa: BLE001
                LOG.exception("openrouter image generation failed")
                yield error("IMAGE_GEN_FAILED", f"生成失败：{ex.__class__.__name__}: {ex}")
                yield done()
                return

            file_name = _derive_filename(req.session_id)
            try:
                public_url = await upload_png(png_bytes, file_name=file_name)
            except Exception as ex:  # noqa: BLE001
                LOG.exception("upload to findu-oss failed")
                yield error("UPLOAD_FAILED", f"上传失败：{ex.__class__.__name__}: {ex}")
                yield done()
                return

            yield ui(
                "show_file",
                url=public_url,
                name=file_name,
                mime="image/png",
                size=len(png_bytes),
            )
            yield done()
        except Exception:  # noqa: BLE001
            LOG.exception("chat handler crashed")
            yield error("INTERNAL_ERROR", "主理人临时开小差，请稍后再试")
            yield done()

    return StreamingResponse(stream(), media_type="text/event-stream")


async def generate_png(prompt: str) -> bytes:
    """Call OpenRouter /chat/completions with multimodal image output.
    Response shape: choices[0].message.images[i].image_url.url is a data URL.
    """
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
    }
    headers = {
        "Authorization": f"Bearer {openrouter_api_key()}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180.0) as http:
        resp = await http.post(f"{OPENROUTER_BASE}/chat/completions",
                               json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter HTTP {resp.status_code}: {resp.text[:400]}"
            )
        body = resp.json()

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError) as ex:
        raise RuntimeError(f"Unexpected OpenRouter response: {body!r}") from ex

    images = message.get("images") or []
    if not images:
        finish = body["choices"][0].get("finish_reason")
        snippet = str(message.get("content") or "")[:200]
        raise RuntimeError(
            f"OpenRouter returned no images (finish_reason={finish}, text={snippet!r})"
        )

    data_url = images[0].get("image_url", {}).get("url") or ""
    if "," not in data_url:
        raise RuntimeError(f"bad data URL shape: {data_url[:80]!r}")
    _, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)


async def upload_png(png_bytes: bytes, *, file_name: str) -> str:
    """Presign via findu-oss (chatfile/image, max 10MB) then PUT to S3.
    Returns the public URL. Bypasses the public gateway and hits the
    internal ALB with the shared platform bypass key."""
    headers = {
        "Content-Type": "application/json",
        "X-Gateway-Bypass": gateway_bypass_key(),
        "X-User-ID": seller_id(),
    }
    body = {
        "uploadType": "chatfile",
        "uploadSubtype": "image",
        "fileName": file_name,
        "fileSize": len(png_bytes),
        "fileType": "image/png",
    }
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            f"{FINDU_ALB}/findu-oss/api/v1/oss_signurl/upload/sign",
            json=body, headers=headers,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"presign HTTP {resp.status_code}: {resp.text[:300]}")
        env = resp.json()
        if env.get("code") not in ("OK", "0", 0, None):
            raise RuntimeError(f"presign {env.get('code')}: {env.get('message')}")
        signed = env.get("data") or {}

        upload_url = (signed.get("uploadUrl") or signed.get("signedUrl")
                      or signed.get("url"))
        if not upload_url:
            raise RuntimeError(f"presign missing uploadUrl: {signed}")
        public_url = (signed.get("publicUrl") or signed.get("downloadUrl")
                      or upload_url.split("?", 1)[0])

        put_resp = await http.put(upload_url, content=png_bytes,
                                  headers={"Content-Type": "image/png"})
        put_resp.raise_for_status()
    return public_url


def _derive_filename(session_id: str) -> str:
    import re
    import time
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")[:16] or "img"
    return f"{safe}-{int(time.time())}.png"
