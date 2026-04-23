"""image-gen-agent — a2hmarket shop agent: prompt → OpenAI gpt-image-2 → PNG.

Flow per /chat turn:
1. Parse ChatRequest, take the latest buyer text as the image prompt.
2. Call OpenAI's `images.generate` (model `gpt-image-2-2026-04-21`), get b64.
3. Presign a `chatfile/image` upload via findu-oss, PUT the PNG to S3.
4. Emit `ui("show_file", ...)` with the public URL, then `done()`.

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
from openai import AsyncOpenAI

from a2h_agent import (
    ChatRequest,
    PlatformClient,
    done,
    error,
    text,
    ui,
)


LOG = logging.getLogger("image_gen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


MODEL_ID = os.environ.get("IMAGE_GEN_MODEL", "gpt-image-2-2026-04-21")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
OPENAI_KEY_PARAM = os.environ.get(
    "OPENAI_API_KEY_PARAM", "/a2h/agents/image-gen/openai-api-key"
)


@lru_cache(maxsize=1)
def openai_api_key() -> str:
    """Resolve the OpenAI key: env OPENAI_API_KEY first (local dev),
    otherwise pull from SSM once per worker."""
    direct = os.environ.get("OPENAI_API_KEY")
    if direct:
        return direct
    ssm = boto3.client("ssm", region_name=AWS_REGION)
    resp = ssm.get_parameter(Name=OPENAI_KEY_PARAM, WithDecryption=True)
    return resp["Parameter"]["Value"]


@lru_cache(maxsize=1)
def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=openai_api_key())


app = FastAPI(title="image-gen-agent", version="1.0.0")


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
                LOG.exception("openai image generation failed")
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
    """Call OpenAI images.generate; return raw PNG bytes."""
    client = openai_client()
    result = await client.images.generate(model=MODEL_ID, prompt=prompt)
    b64 = result.data[0].b64_json
    if not b64:
        raise RuntimeError("OpenAI returned no b64_json")
    return base64.b64decode(b64)


async def upload_png(png_bytes: bytes, *, file_name: str) -> str:
    """Presign via findu-oss (chatfile/image, max 10MB) then PUT to S3.
    Returns the public URL."""
    async with PlatformClient() as p:
        signed = await p.file_upload_presign(
            file_name=file_name,
            file_size=len(png_bytes),
            file_type="image/png",
            upload_type="chatfile",
            upload_subtype="image",
        )
    upload_url = signed["uploadUrl"]
    public_url = signed.get("publicUrl") or upload_url.split("?", 1)[0]

    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.put(
            upload_url,
            content=png_bytes,
            headers={"Content-Type": "image/png"},
        )
        resp.raise_for_status()
    return public_url


def _derive_filename(session_id: str) -> str:
    import re
    import time
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")[:16] or "img"
    return f"{safe}-{int(time.time())}.png"
