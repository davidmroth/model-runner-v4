#!/usr/bin/env python3
"""Reproduce Hermes WebUI vision payloads (large tool schema + image).

Fails with CUDA OOM on unchunked multimodal prefill; should pass after
lucebox-hub feat/native-mmproj chunked text prefill fix.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

BASE = os.environ.get("INFERENCE_BASE", "http://ai.local:8000").rstrip("/")
MODEL = os.environ.get("INFERENCE_MODEL", "dflash")
NUM_TOOLS = int(os.environ.get("HERMES_REPRO_TOOLS", "38"))


def _stub_tools(n: int) -> list[dict]:
    tools = []
    for i in range(n):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": (
                        f"Stub Hermes tool {i} for vision prefill VRAM repro. "
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string", "description": "example arg"},
                            "payload": {"type": "string", "description": "x" * 200},
                        },
                        "required": ["arg"],
                    },
                },
            }
        )
    return tools


def _post(path: str, body: dict, timeout: int = 600) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"POST {path} body_bytes={len(data)} tools={len(body.get('tools') or [])}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    print(f"== hermes vision repro base={BASE} model={MODEL} tools={NUM_TOOLS} ==")

    body = {
        "model": MODEL,
        "max_tokens": 128,
        "stream": False,
        "tools": _stub_tools(NUM_TOOLS),
        "messages": [
            {
                "role": "system",
                "content": "# Tools\n\nYou have access to the following functions.\n"
                + ("<tools>\n" + "\n".join(f"- tool_{i}" for i in range(NUM_TOOLS)) + "\n</tools>\n")
                * 3,
            },
            {
                "role": "user",
                "content": "Can you view images?",
            },
            {
                "role": "assistant",
                "content": "Yes, I can view images when you attach them.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see here? One word only."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
                    },
                ],
            },
        ],
    }

    try:
        out = _post("/v1/chat/completions", body)
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()[:800]
        print(f"FAIL: HTTP {exc.code}: {err}")
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
        print("FAIL: empty response (check lucebox logs for cudaMalloc / error=prefill)")
        return 1
    if "red" in text:
        print("PASS: vision with Hermes-scale tool payload")
        return 0
    print("WARN: got text but not 'red' — vision path ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
