#!/usr/bin/env python3
"""Regression: benchmark traffic must not break the next agent tool call.

Run on ai-inference network (proxy or lucebox direct):
  docker run --rm --network ai-inference \\
    -v $PWD:/w:ro python:3.12-slim python3 /w/test_cache_pollution.py \\
    http://ai-platform-proxy:8000 qwen3.6-27b-autoround
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b-autoround"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

REF_BENCH = {
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two closer than threshold.\n'
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            '    """\n'
        ),
    }],
    "max_tokens": 128,
    "temperature": 0,
    "stream": False,
}


def post(payload: dict, headers: dict | None = None, timeout: int = 1800) -> dict:
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers=hdrs,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def agent_turn(
    messages: list,
    *,
    conversation_id: str | None = None,
    max_tokens: int = 64,
) -> tuple[dict, dict]:
    headers = {}
    if conversation_id:
        headers["X-Conversation-Id"] = conversation_id
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    body = post(payload, headers=headers or None)
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    usage = body.get("usage") or {}
    return msg, {"elapsed_s": round(elapsed, 2), "usage": usage, "body": body}


def main() -> int:
    failures: list[str] = []
    print(f"=== cache pollution test @ {BASE} model={MODEL} ===\n")

    # 1. Pollute with reference-style decode (no conversation id).
    print("--- Step 1: reference benchmark (polluter) ---")
    try:
        ref = post(REF_BENCH)
        ref_u = ref.get("usage") or {}
        print(
            f"  ok completion_tokens={ref_u.get('completion_tokens')} "
            f"prompt_tokens={ref_u.get('prompt_tokens')}"
        )
    except Exception as exc:
        failures.append(f"reference bench failed: {exc}")
        print(f"  FAIL: {exc}")

    time.sleep(0.5)

    # 2. Agent request immediately after — must not be empty or leak raw tags.
    print("--- Step 2: agent tool turn (scoped session A) ---")
    conv_a = "cert-pollution-a"
    msgs = [{"role": "user", "content": "Use terminal to run: echo CACHE_TEST_A"}]
    try:
        msg, meta = agent_turn(msgs, conversation_id=conv_a)
        ct = (meta["usage"] or {}).get("completion_tokens", 0)
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        print(f"  elapsed={meta['elapsed_s']}s ct={ct} tool_calls={len(tool_calls)}")
        if ct == 0 and not tool_calls:
            failures.append("agent turn A: empty response (0 tokens, no tool_calls)")
        if "<function=" in content or "<tool_call>" in content:
            failures.append(f"agent turn A: raw tool syntax in content: {content[:120]!r}")
        if not tool_calls:
            failures.append("agent turn A: expected structured tool_calls")
        else:
            names = [tc.get("function", {}).get("name") for tc in tool_calls]
            print(f"  tools: {names}")
    except Exception as exc:
        failures.append(f"agent turn A failed: {exc}")
        print(f"  FAIL: {exc}")

    time.sleep(0.5)

    # 3. Different conversation — must not inherit polluted KV.
    print("--- Step 3: agent tool turn (scoped session B) ---")
    conv_b = "cert-pollution-b"
    msgs_b = [{"role": "user", "content": "Use terminal to run: echo CACHE_TEST_B"}]
    try:
        msg, meta = agent_turn(msgs_b, conversation_id=conv_b)
        ct = (meta["usage"] or {}).get("completion_tokens", 0)
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        print(f"  elapsed={meta['elapsed_s']}s ct={ct} tool_calls={len(tool_calls)}")
        if ct == 0 and not tool_calls:
            failures.append("agent turn B: empty response")
        if "<function=" in content:
            failures.append("agent turn B: raw tool syntax in content")
        if not tool_calls:
            failures.append("agent turn B: expected tool_calls")
    except Exception as exc:
        failures.append(f"agent turn B failed: {exc}")
        print(f"  FAIL: {exc}")

    print("\n=== Summary ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 2
    print("  All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
