#!/usr/bin/env python3
"""Phase 3 M3b gate: dual HTTP chat streams while overlap mode is on.

Flag-gated. Expects model-runner-v4-lucebox running with:
  DFLASH_TARGET_CACHE_SLOTS=2
  DFLASH_STREAM_TAGGED=1
  DFLASH_MULTI_SLOT_DROP_EXCLUSIVE=1

Protocol under test (tool-split warm path):
  1) Warm two conversation scopes so tool pin + thick slots exist.
  2) Fire two concurrent streaming /v1/chat/completions.
  3) Pass if both get 200 + tokens and neither 503s; optionally both
     ``target_cache_slot=`` appear in container logs during the overlap window.

Compose defaults stay N=1 — run this only against a temporary N=2 recreate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


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


def _sse_text(payload: str) -> str:
    chunks: list[str] = []
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
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
    return "".join(chunks)


def _json_text(payload: str) -> str:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def _docker_logs_since(container: str, since_iso: str) -> str:
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--since", since_iso, container],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"(log fetch failed: {exc})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        default=os.environ.get("PHASE3_HTTP_BASE", "http://127.0.0.1:8080"),
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("PRIMARY_MODEL", "qwen3.6-27b-autoround"),
    )
    ap.add_argument(
        "--container",
        default=os.environ.get("PHASE3_HTTP_CONTAINER", "model-runner-v4-lucebox"),
    )
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--allow",
        action="store_true",
        help="Allow run without GATE_PHASE3_HTTP_OVERLAP=1 (interactive only).",
    )
    args = ap.parse_args()

    gated = os.environ.get("GATE_PHASE3_HTTP_OVERLAP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not gated and not args.allow:
        print(
            "REFUSE: set GATE_PHASE3_HTTP_OVERLAP=1 (or --allow) after recreating "
            "compose with N=2 + tagged + drop-exclusive (defaults stay N=1).",
            file=sys.stderr,
        )
        return 2

    # Probe health
    try:
        with urllib.request.urlopen(f"{args.base.rstrip('/')}/health", timeout=10) as r:
            health = r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"FAIL: health unreachable: {exc}", file=sys.stderr)
        return 1
    print(f"health={health.strip()[:200]}")

    ts = int(time.time())
    conv_a = f"phase3-overlap-a-{ts}"
    conv_b = f"phase3-overlap-b-{ts}"

    print("== warm A ==")
    st, body, dt = _post_chat(
        args.base,
        model=args.model,
        conversation_id=conv_a,
        content="Reply with exactly: warm-a",
        max_tokens=8,
        stream=False,
        timeout=args.timeout,
    )
    text_a = _json_text(body)
    print(f"warm A status={st} dt={dt:.1f}s text={text_a!r}")
    if st != 200:
        print("FAIL: warm A", file=sys.stderr)
        return 1

    print("== warm B ==")
    st, body, dt = _post_chat(
        args.base,
        model=args.model,
        conversation_id=conv_b,
        content="Reply with exactly: warm-b",
        max_tokens=8,
        stream=False,
        timeout=args.timeout,
    )
    text_b = _json_text(body)
    print(f"warm B status={st} dt={dt:.1f}s text={text_b!r}")
    if st != 200:
        print("FAIL: warm B", file=sys.stderr)
        return 1

    print("== dual stream overlap ==")
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    results: dict[str, tuple[int, str, float]] = {}

    def worker(name: str, conv: str, prompt: str) -> None:
        results[name] = _post_chat(
            args.base,
            model=args.model,
            conversation_id=conv,
            content=prompt,
            max_tokens=args.max_tokens,
            stream=True,
            timeout=args.timeout,
        )

    t0 = time.monotonic()
    ta = threading.Thread(
        target=worker,
        args=("A", conv_a, "Count slowly from 1 to 12 with spaces."),
        daemon=True,
    )
    tb = threading.Thread(
        target=worker,
        args=("B", conv_b, "Count slowly from 101 to 112 with spaces."),
        daemon=True,
    )
    ta.start()
    time.sleep(0.15)
    tb.start()
    ta.join(timeout=args.timeout + 30)
    tb.join(timeout=args.timeout + 30)
    wall = time.monotonic() - t0

    ok = True
    for name in ("A", "B"):
        if name not in results:
            print(f"FAIL: missing result {name}", file=sys.stderr)
            ok = False
            continue
        st, payload, dt = results[name]
        text = _sse_text(payload) if st == 200 else payload[:300]
        print(f"stream {name} status={st} dt={dt:.1f}s text={text!r}")
        if st != 200 or not text.strip():
            ok = False

    logs = _docker_logs_since(args.container, since)
    slot_lines = [
        ln for ln in logs.splitlines() if "target_cache_slot=" in ln or "RESTORE_CHAIN_ADMIT" in ln
    ]
    print("-- overlap window logs (slot/admit) --")
    for ln in slot_lines[-40:]:
        print(ln)
    slots_seen = {
        ln.split("target_cache_slot=", 1)[1].split()[0]
        for ln in slot_lines
        if "target_cache_slot=" in ln and "(chat" in ln
    }
    print(f"slots_seen={sorted(slots_seen)} wall={wall:.1f}s")

    if not ok:
        print("FAIL: one or both streams missing tokens/200", file=sys.stderr)
        return 1

    # Overlap signal: wall clock closer to max(dt) than sum(dt).
    dts = [results["A"][2], results["B"][2]]
    max_dt = max(dts)
    sum_dt = sum(dts)
    if wall > (sum_dt * 0.85) and min(dts) > 2.0:
        print(
            f"FAIL: wall={wall:.1f}s ≈ sum(dts)={sum_dt:.1f}s — requests appear serial",
            file=sys.stderr,
        )
        return 1
    if len(slots_seen) >= 2:
        print(f"PASS: dual HTTP streams overlapped (slots={sorted(slots_seen)})")
    else:
        print(
            "PASS: dual HTTP streams overlapped "
            f"(wall={wall:.1f}s max_dt={max_dt:.1f}s; "
            f"slot log scrape incomplete: {sorted(slots_seen)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
