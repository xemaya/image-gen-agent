"""image-gen-agent — a2hmarket shop agent: prompt → Gemini image model → image.

Flow per /chat turn:
1. Parse ChatRequest, take the latest buyer text as the image prompt.
2. POST to Google's generativelanguage API
   (/v1beta/models/{model}:generateContent) with
   responseModalities=[TEXT, IMAGE].
3. Read inlineData (base64) from candidates[0].content.parts[*].inlineData
   — keep whatever mimeType Gemini returns (typically image/jpeg).
4. Presign a `chatfile/image` upload via findu-oss (slot accepts jpeg/png/gif/webp,
   10 MB cap) and PUT the bytes to S3.
5. Emit `ui("show_file", ...)` with the public URL, then `done()`.

Model: gemini-3.1-flash-image-preview. 10-20 s / image, ~1408x768 JPEG,
~900 KB average.

No orders, no payment — free generation. Single tool: `a2h.file.upload`.
"""

from __future__ import annotations

import base64
import logging
import os
from functools import lru_cache

import boto3
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from a2h_agent import (
    ChatRequest,
    done,
    error,
    text,
    ui,
)


LOG = logging.getLogger("image_gen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


MODEL_ID = os.environ.get("IMAGE_GEN_MODEL", "gemini-3.1-flash-image-preview")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
GEMINI_KEY_PARAM = os.environ.get(
    "GEMINI_API_KEY_PARAM", "/a2h/agents/image-gen/gemini-api-key"
)
GEMINI_BASE = os.environ.get(
    "GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

# Internal ALB + gateway bypass for findu-oss presign (same pattern as
# zhangxuefeng/renew-life). A2HClient's bearer-token path doesn't cover
# /findu-oss/*, so we bypass-call the ALB directly.
FINDU_ALB = os.environ.get(
    "A2H_FINDU_ALB",
    "http://findu-alb-476446960.us-east-1.elb.amazonaws.com",
).rstrip("/")
SSM_GATEWAY_BYPASS = "/a2h/agents/shared/gateway-bypass-key"


@lru_cache(maxsize=1)
def gemini_api_key() -> str:
    """Resolve the Gemini key: env GEMINI_API_KEY first (local dev),
    otherwise pull from SSM once per worker."""
    direct = os.environ.get("GEMINI_API_KEY")
    if direct:
        return direct
    ssm = boto3.client("ssm", region_name=AWS_REGION)
    resp = ssm.get_parameter(Name=GEMINI_KEY_PARAM, WithDecryption=True)
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


app = FastAPI(title="image-gen-agent", version="1.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invoke")
async def invoke(request: Request) -> dict[str, object]:
    """RPC-style endpoint exposed via the platform's
    ``POST /api/v1/shops/{shopId}/invoke`` proxy. Body: {"prompt": "..."}.

    Generates the image, uploads to findu-oss (public findu-media-us
    bucket) and returns a JSON envelope with the public URL. Returning
    JSON instead of raw bytes is what downstream agents' generic
    httpPost tools expect — binary PNG read as String gets mangled by
    charset decoding, and JSON is the lingua franca.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="body must be JSON")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")

    try:
        img_bytes, mime = await generate_image(prompt)
    except Exception as ex:  # noqa: BLE001
        LOG.exception("invoke: image gen failed")
        raise HTTPException(
            status_code=502,
            detail=f"image generation failed: {ex.__class__.__name__}: {ex}",
        )

    file_name = _derive_filename("invoke", mime)
    try:
        public_url = await upload_image(img_bytes, file_name=file_name, mime=mime)
    except Exception as ex:  # noqa: BLE001
        LOG.exception("invoke: upload failed")
        raise HTTPException(
            status_code=502,
            detail=f"upload failed: {ex.__class__.__name__}: {ex}",
        )

    return {
        "url": public_url,
        "name": file_name,
        "mime": mime,
        "size": len(img_bytes),
    }


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
                img_bytes, mime = await generate_image(prompt)
            except Exception as ex:  # noqa: BLE001
                LOG.exception("gemini image generation failed")
                yield error("IMAGE_GEN_FAILED", f"生成失败：{ex.__class__.__name__}: {ex}")
                yield done()
                return

            file_name = _derive_filename(req.session_id, mime)
            try:
                public_url = await upload_image(img_bytes, file_name=file_name, mime=mime)
            except Exception as ex:  # noqa: BLE001
                LOG.exception("upload to findu-oss failed")
                yield error("UPLOAD_FAILED", f"上传失败：{ex.__class__.__name__}: {ex}")
                yield done()
                return

            yield ui(
                "show_file",
                url=public_url,
                name=file_name,
                mime=mime,
                size=len(img_bytes),
            )
            yield done()
        except Exception:  # noqa: BLE001
            LOG.exception("chat handler crashed")
            yield error("INTERNAL_ERROR", "主理人临时开小差，请稍后再试")
            yield done()

    return StreamingResponse(stream(), media_type="text/event-stream")


async def generate_image(prompt: str) -> tuple[bytes, str]:
    """Call Gemini generativelanguage v1beta :generateContent with
    responseModalities=[TEXT, IMAGE]. Returns ``(bytes, mime)`` where
    mime is whatever the model returned (typically image/jpeg)."""
    url = f"{GEMINI_BASE}/models/{MODEL_ID}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    # Key goes in query for this API; header also works but query is canonical.
    params = {"key": gemini_api_key()}

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(url, json=payload, params=params,
                               headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            raise RuntimeError(
                f"Gemini HTTP {resp.status_code}: {resp.text[:400]}"
            )
        body = resp.json()

    candidates = body.get("candidates") or []
    if not candidates:
        pf = body.get("promptFeedback") or {}
        raise RuntimeError(f"Gemini returned no candidates (promptFeedback={pf})")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    for p in parts:
        # Google normalises to camelCase (inlineData) but protobuf fallback
        # is snake_case — handle both so we don't break on SDK changes.
        data = p.get("inlineData") or p.get("inline_data")
        if not data:
            continue
        b64 = data.get("data")
        mime = data.get("mimeType") or data.get("mime_type") or "image/png"
        if b64:
            return base64.b64decode(b64), mime

    # Log what we got so ops can tell whether the model declined (only text
    # response, safety block, etc.) or Gemini shape changed upstream.
    text_snippet = " ".join(str(p.get("text", ""))[:200] for p in parts if "text" in p)
    raise RuntimeError(
        f"Gemini returned no inlineData (finishReason="
        f"{candidates[0].get('finishReason')!r}, text={text_snippet!r})"
    )


async def upload_image(img_bytes: bytes, *, file_name: str, mime: str) -> str:
    """Presign via findu-oss (chatfile/image, max 10MB) then PUT to S3.
    Accepts any image/* mime the slot allows (jpeg/png/gif/webp). Bypasses
    the public gateway and hits the internal ALB with the shared platform
    bypass key."""
    headers = {
        "Content-Type": "application/json",
        "X-Gateway-Bypass": gateway_bypass_key(),
        "X-User-ID": seller_id(),
    }
    body = {
        "uploadType": "chatfile",
        "uploadSubtype": "image",
        "fileName": file_name,
        "fileSize": len(img_bytes),
        "fileType": mime,
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

        put_resp = await http.put(upload_url, content=img_bytes,
                                  headers={"Content-Type": mime})
        put_resp.raise_for_status()
    return public_url


_MIME_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _derive_filename(session_id: str, mime: str = "image/png") -> str:
    import re
    import time
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")[:16] or "img"
    ext = _MIME_EXT.get(mime, "png")
    return f"{safe}-{int(time.time())}.{ext}"
