#!/usr/bin/env python3
"""Route multimodal chat requests to the vision sidecar; text stays on Lucebox."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# vision_detect lives in the mounted lucebox patch dir when present.
_PATCH = Path(os.environ.get("DFLASH_PATCH_SCRIPTS", "/opt/lucebox-hub/patch/dflash/scripts"))
if _PATCH.is_dir() and str(_PATCH) not in sys.path:
    sys.path.insert(0, str(_PATCH))

from vision_detect import request_has_vision  # noqa: E402

TEXT_BACKEND = os.environ.get("DFLASH_TEXT_BACKEND", "http://127.0.0.1:18080").rstrip("/")
VISION_BACKEND = os.environ.get("DFLASH_VISION_BACKEND", "http://vision:8081").rstrip("/")
VISION_ENABLED = os.environ.get("DFLASH_VISION_ENABLED", "1") == "1"
HOST = os.environ.get("DFLASH_GATEWAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("DFLASH_GATEWAY_PORT", "8080"))

app = FastAPI(title="lucebox-vision-gateway", version="0.1.0")
_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))


def _pick_backend(body: dict[str, Any]) -> str:
    if not VISION_ENABLED:
        return TEXT_BACKEND
    messages = body.get("messages") or []
    if request_has_vision(messages):
        return VISION_BACKEND
    return TEXT_BACKEND


async def _proxy(request: Request, path: str) -> Response:
    body = await request.body()
    backend = TEXT_BACKEND
    if body and request.method in {"POST", "PUT", "PATCH"}:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                backend = _pick_backend(parsed)
        except json.JSONDecodeError:
            pass

    url = f"{backend}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "transfer-encoding"}
    }

    upstream = await _client.request(
        request.method,
        url,
        headers=headers,
        content=body,
    )

    hop_by_hop = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                  "te", "trailers", "transfer-encoding", "upgrade"}
    out_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in hop_by_hop
    }

    if "text/event-stream" in upstream.headers.get("content-type", ""):
        async def _stream() -> Any:
            async for chunk in upstream.aiter_bytes():
                yield chunk

        return StreamingResponse(
            _stream(),
            status_code=upstream.status_code,
            headers=out_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=out_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.get("/health")
async def health() -> JSONResponse:
  text_ok = vision_ok = False
  try:
      r = await _client.get(f"{TEXT_BACKEND}/health", timeout=10.0)
      text_ok = r.status_code == 200
  except httpx.HTTPError:
      pass
  if VISION_ENABLED:
      try:
          r = await _client.get(f"{VISION_BACKEND}/health", timeout=10.0)
          vision_ok = r.status_code == 200
      except httpx.HTTPError:
          pass
  else:
      vision_ok = True
  ok = text_ok and vision_ok
  return JSONResponse(
      {
          "status": "ok" if ok else "degraded",
          "text_backend": TEXT_BACKEND,
          "vision_backend": VISION_BACKEND if VISION_ENABLED else None,
          "text_ok": text_ok,
          "vision_ok": vision_ok,
          "vision_enabled": VISION_ENABLED,
      },
      status_code=200 if ok else 503,
  )


@app.get("/props")
async def props() -> Response:
    upstream = await _client.get(f"{TEXT_BACKEND}/props", timeout=30.0)
    if upstream.status_code != 200:
        return Response(content=upstream.content, status_code=upstream.status_code)
    try:
        body = upstream.json()
    except json.JSONDecodeError:
        return Response(content=upstream.content, status_code=upstream.status_code)
    caps = body.setdefault("capabilities", {})
    if isinstance(caps, dict) and VISION_ENABLED:
        caps["vision_supported"] = True
        caps["vision_backend"] = VISION_BACKEND
    return JSONResponse(body)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str) -> Response:
    return await _proxy(request, path)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
