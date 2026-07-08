#!/usr/bin/env python3
"""Vision smoke test — describe a tiny inline PNG via the inference stack."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

# 1x1 red PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

BASE = os.environ.get("INFERENCE_BASE", "http://ai.local:8000").rstrip("/")
MODEL = os.environ.get("INFERENCE_MODEL", "dflash")


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    print(f"== vision smoke base={BASE} model={MODEL} ==")

    try:
        props = json.loads(
            urllib.request.urlopen(f"{BASE}/props", timeout=30).read().decode()
        )
        caps = props.get("capabilities") or {}
        print(f"vision_supported={caps.get('vision_supported')} backend={caps.get('vision_backend')}")
    except Exception as exc:
        print(f"WARN: /props check failed: {exc}")

    body = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? One word only."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
                    },
                ],
            }
        ],
    }

    try:
        out = _post("/v1/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"FAIL: HTTP {exc.code}: {exc.read().decode()[:500]}")
        return 1
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    text = (
        out.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
        .lower()
    )
    print(f"response={text!r}")

    if not text:
        print("FAIL: empty response")
        return 1
    if "red" in text:
        print("PASS: model identified red")
        return 0
    print("WARN: response did not mention red — vision path ran but answer unexpected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
