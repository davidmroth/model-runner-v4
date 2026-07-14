#!/usr/bin/env python3
"""Repro: background max_tokens=64000 ephemeral storm vs scoped chat (N=2).

Live signature (2026-07-14):
  After a scoped chat turn, agentlens shows a burst of
  ``traffic_class=background max_tokens=64000`` starts (no conversation id).
  Scoped chat then hits ``target_cache_slot wait timed out`` → HTTP 503.

Gate: GATE_PHASE3_HTTP_BG_STORM=1
Expects lucebox with N=2 overlap flags on.

This smoke synthesizes the storm from HTTP (ephemeral POSTs) while a scoped
chat holds/uses a slot, then asserts the chat still completes (or records the
known 503 failure mode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


def _post(
    base: str,
    *,
    model: str,
    content: str,
    max_tokens: int,
    conversation_id: str | None,
    stream: bool,
    timeout: float,
) -> tuple[int, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    headers = {"Content-Type": "application/json"}
    if conversation_id:
        headers["X-Conversation-Id"] = conversation_id
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            resp.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            exc.read()
        except Exception:
            pass
    except Exception:
        status = -1
    return status, time.monotonic() - t0


def main() -> int:
    if os.environ.get("GATE_PHASE3_HTTP_BG_STORM", "0") not in (
        "1", "true", "yes", "on",
    ):
        print("SKIP: set GATE_PHASE3_HTTP_BG_STORM=1 to run")
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dflash")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--storm-n", type=int, default=12)
    ap.add_argument("--storm-max-tokens", type=int, default=64000)
    ap.add_argument("--expect-fail", action="store_true",
                    help="Exit 0 when the 503 bug is reproduced (failing-test mode)")
    args = ap.parse_args()

    conv = f"phase3-bg-storm-{uuid.uuid4().hex[:10]}"

    print("== warm scoped ==")
    st, dt = _post(
        args.base,
        model=args.model,
        content="Reply with exactly: warm-bg",
        max_tokens=16,
        conversation_id=conv,
        stream=False,
        timeout=args.timeout,
    )
    print(f"warm status={st} dt={dt:.1f}s")
    if st != 200:
        print("FAIL: warm", file=sys.stderr)
        return 1

    storm_statuses: list[int] = []
    stop = threading.Event()

    def storm_worker(i: int) -> None:
        # Tiny prompt, huge max_tokens — matches hindsight/extractor shape.
        st_i, dt_i = _post(
            args.base,
            model=args.model,
            content=f"Extract facts from: hello #{i}. One short sentence.",
            max_tokens=args.storm_max_tokens,
            conversation_id=None,  # ephemeral
            stream=False,
            timeout=min(30.0, args.timeout),
        )
        storm_statuses.append(st_i)
        print(f"  storm[{i}] status={st_i} dt={dt_i:.1f}s")

    print(f"== launch background storm n={args.storm_n} max_tokens={args.storm_max_tokens} ==")
    threads = [
        threading.Thread(target=storm_worker, args=(i,), daemon=True)
        for i in range(args.storm_n)
    ]
    for t in threads:
        t.start()
        time.sleep(0.05)

    print("== scoped chat during storm ==")
    chat_st, chat_dt = _post(
        args.base,
        model=args.model,
        content=(
            "List five concrete steps to debug a multi-slot inference engine. "
            "One sentence each."
        ),
        max_tokens=256,
        conversation_id=conv,
        stream=True,
        timeout=args.timeout,
    )
    print(f"chat status={chat_st} dt={chat_dt:.1f}s")

    stop.set()
    for t in threads:
        t.join(timeout=60.0)

    n_503 = sum(1 for s in storm_statuses if s == 503)
    n_200 = sum(1 for s in storm_statuses if s == 200)
    print(f"storm results: 200={n_200} 503={n_503} other={len(storm_statuses)-n_200-n_503}")

    chat_busy = chat_st == 503
    if args.expect_fail:
        if chat_busy or n_503 >= max(3, args.storm_n // 3):
            print(
                "REPRODUCED: background storm caused chat 503 and/or storm 503 flood "
                f"(chat={chat_st}, storm_503={n_503})"
            )
            return 0
        print(
            "UNEXPECTED PASS under --expect-fail: storm did not stress admission",
            file=sys.stderr,
        )
        return 1

    if chat_busy:
        print(
            f"FAIL: scoped chat got 503 during background storm (dt={chat_dt:.1f}s)",
            file=sys.stderr,
        )
        return 1
    if chat_st != 200:
        print(f"FAIL: scoped chat status={chat_st}", file=sys.stderr)
        return 1

    print("PASS: scoped chat completed during background storm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
