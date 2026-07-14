#!/usr/bin/env python3
"""Repro: N=2 quantum admit truncates chat to ~first quantum (~8–16 toks).

Live signature (2026-07-14, conversation b9d8c6a0-…):
  RESTORE_CHAIN first quantum gen=8 → HTTP returns completion_tokens≈10–16
  while SCHED_DRAIN may still run for tens of seconds afterward.

Gate: GATE_PHASE3_HTTP_QUANTUM_TRUNC=1
Expects lucebox with:
  DFLASH_TARGET_CACHE_SLOTS=2
  DFLASH_STREAM_TAGGED=1
  DFLASH_MULTI_SLOT_DROP_EXCLUSIVE=1

FAIL (current bug) when completion stays near DFLASH_SCHED_QUANTUM.
PASS when a clearly long answer is produced well beyond one quantum.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def _post_chat(
    base: str,
    *,
    model: str,
    conversation_id: str,
    content: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
) -> tuple[int, str, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Conversation-Id": conversation_id,
        },
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        payload = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}", time.monotonic() - t0
    return status, payload, time.monotonic() - t0


def _json_usage_and_text(payload: str) -> tuple[dict, str]:
    obj = json.loads(payload)
    usage = obj.get("usage") or {}
    text = ""
    try:
        text = obj["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    return usage, text


def _sse_usage_and_text(payload: str) -> tuple[dict, str]:
    text_parts: list[str] = []
    usage: dict = {}
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            text_parts.append(piece)
    return usage, "".join(text_parts)


def _docker_logs_since(container: str, since: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "logs", "--since", since, container],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"(log scrape failed: {exc})"


def main() -> int:
    if os.environ.get("GATE_PHASE3_HTTP_QUANTUM_TRUNC", "0") not in (
        "1", "true", "yes", "on",
    ):
        print("SKIP: set GATE_PHASE3_HTTP_QUANTUM_TRUNC=1 to run")
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dflash")
    ap.add_argument("--container", default="model-runner-v4-lucebox")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--quantum", type=int,
                    default=int(os.environ.get("DFLASH_SCHED_QUANTUM", "8")))
    ap.add_argument("--min-completion", type=int, default=0,
                    help="Override pass bar (default 4× quantum)")
    args = ap.parse_args()

    min_completion = args.min_completion or max(32, args.quantum * 4)
    conv = f"phase3-quantum-trunc-{uuid.uuid4().hex[:10]}"

    print("== warm (seed tool/thick path) ==")
    st, body, dt = _post_chat(
        args.base,
        model=args.model,
        conversation_id=conv,
        content="Reply with exactly: warm-quantum",
        max_tokens=16,
        stream=False,
        timeout=args.timeout,
    )
    print(f"warm status={st} dt={dt:.1f}s body[:120]={body[:120]!r}")
    if st != 200:
        print("FAIL: warm request", file=sys.stderr)
        return 1

    prompt = (
        "Write a careful multi-paragraph answer (at least 250 words) explaining "
        "how layer-split KV caching works on dual GPUs. Number the paragraphs "
        "1, 2, 3, 4. Do not stop after a short preamble — finish all paragraphs."
    )
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    print("== long completion (agent-shaped) ==")
    st, body, dt = _post_chat(
        args.base,
        model=args.model,
        conversation_id=conv,
        content=prompt,
        max_tokens=args.max_tokens,
        stream=True,
        timeout=args.timeout,
    )
    usage, text = _sse_usage_and_text(body) if st == 200 else ({}, body[:300])
    # Some setups put usage only on non-stream; fall back.
    if st == 200 and not usage.get("completion_tokens"):
        st2, body2, _ = _post_chat(
            args.base,
            model=args.model,
            conversation_id=conv,
            content="Continue the previous answer with two more full paragraphs.",
            max_tokens=args.max_tokens,
            stream=False,
            timeout=args.timeout,
        )
        if st2 == 200:
            usage, text = _json_usage_and_text(body2)
            st, dt = st2, dt  # keep first wall for timing note

    completion = int(usage.get("completion_tokens") or 0)
    # Prefer stream text length as a secondary signal when usage is missing.
    words = len(text.split())
    print(f"long status={st} dt={dt:.1f}s completion_tokens={completion} "
          f"words≈{words} text[:160]={text[:160]!r}")

    logs = _docker_logs_since(args.container, since)
    gen8 = [ln for ln in logs.splitlines() if " gen=8 " in ln or " gen=8\n" in ln
            or "gen=8 " in ln]
    sched = [ln for ln in logs.splitlines() if "SCHED_DRAIN" in ln]
    print("-- daemon snippets --")
    for ln in (gen8[-3:] + sched[-5:]):
        print(ln)

    if st != 200:
        print("FAIL: long request not 200", file=sys.stderr)
        return 1

    # Classic N=2 truncation: completion ≈ one quantum (8) or just above it.
    if completion and completion <= max(args.quantum * 2, 16):
        print(
            f"FAIL: quantum truncation suspected — completion_tokens={completion} "
            f"<= {max(args.quantum * 2, 16)} (quantum={args.quantum})",
            file=sys.stderr,
        )
        return 1
    if not completion and words < 80:
        print(
            f"FAIL: short answer (words≈{words}) with no usage.completion_tokens",
            file=sys.stderr,
        )
        return 1
    if completion and completion < min_completion:
        print(
            f"FAIL: completion_tokens={completion} < min_completion={min_completion}",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: long completion survived past first quantum "
        f"(completion_tokens={completion}, words≈{words})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
