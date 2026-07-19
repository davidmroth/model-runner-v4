#!/usr/bin/env python3
"""L0 harness gate: streamed tool-call turn through the proxy.

Pass criteria:
  - HTTP 200 (no 503)
  - SSE role chunk arrives
  - Valid tool_calls with expected function name(s)
  - No engine_timeout / empty failure payload
  - Optional: keepalive comments when decode is long (best-effort)

Scoped to inference path (proxy → lucebox). Does not touch Hermes.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b-autoround"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
]

SYSTEM = (
    "You are a careful coding agent. When the user asks about files, you MUST "
    "call tools — never invent file contents. Prefer read_file for file reads."
)

USER = (
    "I need a robust check that tooling works end-to-end. "
    "Please call read_file with path='/etc/hostname' and also call "
    "list_directory with path='/tmp'. Do not summarize first — emit the tool "
    "calls now."
)


def main() -> int:
    conv = str(uuid.uuid4())
    body = {
        "model": MODEL,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 1024,
        "temperature": 0.2,
        "tools": TOOLS,
        "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE.rstrip('/')}/v1/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Conversation-Id": conv,
            "Accept": "text/event-stream",
        },
    )
    t0 = time.time()
    keepalives = 0
    role_seen = False
    finish = None
    tool_names: list[str] = []
    tool_args_bufs: dict[int, str] = {}
    err_payload = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            if status == 503:
                print("FAIL: HTTP 503")
                return 1
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line.startswith(":"):
                    keepalives += 1
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err_payload = obj["error"]
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                finish = ch0.get("finish_reason") or finish
                delta = ch0.get("delta") or {}
                if delta.get("role"):
                    role_seen = True
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    fn = (tc.get("function") or {}).get("name")
                    if fn:
                        while len(tool_names) <= idx:
                            tool_names.append("")
                        tool_names[idx] = fn
                    args = (tc.get("function") or {}).get("arguments")
                    if args:
                        tool_args_bufs[idx] = tool_args_bufs.get(idx, "") + args
    except urllib.error.HTTPError as e:
        print(f"FAIL: HTTP {e.code}: {e.read()[:400]!r}")
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    elapsed = time.time() - t0
    names = [n for n in tool_names if n]
    print(
        json.dumps(
            {
                "status": status,
                "elapsed_s": round(elapsed, 2),
                "conversation_id": conv,
                "role_seen": role_seen,
                "finish_reason": finish,
                "tool_names": names,
                "keepalive_comments": keepalives,
                "error": err_payload,
                "arg_lens": {str(k): len(v) for k, v in tool_args_bufs.items()},
            },
            indent=2,
        )
    )

    failures: list[str] = []
    if status != 200:
        failures.append(f"status={status}")
    if err_payload:
        failures.append(f"error={err_payload}")
    if not role_seen:
        failures.append("missing role chunk")
    if "read_file" not in names:
        failures.append("missing read_file tool call")
    if "list_directory" not in names:
        failures.append("missing list_directory tool call")
    if finish not in ("tool_calls", "stop", None):
        # None can happen if stream ended oddly; still require tools
        failures.append(f"unexpected finish_reason={finish!r}")
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    print("PASS: L0 tooling gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
