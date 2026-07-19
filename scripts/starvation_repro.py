#!/usr/bin/env python3
"""Starvation reproduction harness — prove (or disprove) the multi-slot stall.

The pathology we are chasing: under concurrency, an *already-admitted, already-
decoding* request stops receiving the worker for minutes ("started, then
dropped"), even though it holds a live slot. This is different from polite
queueing (where a request simply waits its turn and then runs steadily). The
tell is a large **mid-stream** inter-token gap on a request that had already
begun emitting tokens.

This harness measures that directly:

  1. BASELINE  — run a steady "victim" generation *solo* and record the
     inter-token gaps. Healthy: gaps are small and uniform.
  2. CONTENDED — start a long "runaway" generation, wait until it is confirmed
     decoding, then run the *same* victim alongside it and record its gaps.

     - If the victim starts promptly but then shows large MID-stream gaps
       -> resume/reschedule stall (the bug: interleave engaged, then dropped).
     - If the victim's FIRST token is delayed but mid-stream gaps stay small
       -> plain head-of-line queueing (single slot / no interleave). Bounded,
       self-healing, not the bug.

It measures, it does not mutate. The runaway is bounded by ``max_tokens`` and is
torn down (connection closed -> engine disconnect-cancel) as soon as the victim
finishes, so it does not linger on a shared engine.

Stdlib only (urllib, json, argparse, threading) — the ai.local containers have
hit ModuleNotFoundError on yaml/httpx before.

Hit the ENGINE directly (default ``http://127.0.0.1:8080``) so the proxy's 600s
wall does not truncate the measurement. Run inside a container on ai.local, per
the repo docker rule.

Usage:

  # Full before/after: victim solo, then victim + runaway
  python3 starvation_repro.py --endpoint http://127.0.0.1:8080

  # Skip the baseline, only reproduce under contention
  python3 starvation_repro.py --no-baseline

  # Tune load / sensitivity
  python3 starvation_repro.py --runaway-max-tokens 8000 --stall-threshold 5 \
      --victim-max-tokens 400 --json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_MODEL = "qwen3.6-27b-autoround"

# A steady, easy-to-stream victim: uniform token cadence makes gaps meaningful.
VICTIM_PROMPT = (
    "Count from 1 to 300. Print one integer per line, in order, and nothing "
    "else. Do not add any commentary."
)
# A verbose runaway that keeps the worker busy long enough to expose the stall.
RUNAWAY_PROMPT = (
    "Write an exhaustive, extremely detailed technical reference covering "
    "operating systems, distributed systems, databases, networking, and "
    "concurrency. Include many sections and subsections. Be maximally verbose "
    "and do not stop early."
)


def _pct(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolation percentile over a pre-sorted list (0.0-1.0)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stream_request(
    endpoint: str,
    body: dict,
    timeout: float,
    *,
    label: str,
    stop_event: threading.Event | None = None,
    admitted_event: threading.Event | None = None,
    first_token_event: threading.Event | None = None,
) -> dict:
    """Stream a chat-completions request, recording a timestamp per token.

    Returns a metrics dict; never raises on a backend error (captures it). If
    ``stop_event`` is set, the read loop breaks and closes the connection, which
    the engine sees as a client disconnect (bounded teardown of the runaway).

    ``admitted_event`` fires when the HTTP response is open (request accepted /
    slot held). That matters on this engine because the SSE stream is often
    *buffered* — all tokens arrive in one burst at the end — so waiting on the
    first client-visible token is waiting on the whole generation.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"

    req_body = dict(body)
    req_body["stream"] = True
    opts = dict(req_body.get("stream_options") or {})
    opts["include_usage"] = True
    req_body["stream_options"] = opts

    data = json.dumps(req_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    metrics: dict = {
        "label": label,
        "endpoint": url,
        "model": req_body.get("model"),
        "requested_max_tokens": req_body.get("max_tokens"),
        "http_error": None,
        "backend_error": None,
        "finish_reason": None,
        "t0": None,
        "t_admitted": None,
        "t_first_token": None,
        "t_end": None,
        "token_times": [],  # monotonic timestamp per emitted token
        "stopped_early": False,
        "buffered_stream": False,
    }

    t0 = time.monotonic()
    metrics["t0"] = t0
    try:
        resp = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            metrics["http_error"] = (
                f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
            )
        except Exception:
            metrics["http_error"] = f"HTTP {exc.code}"
        return metrics
    except (urllib.error.URLError, TimeoutError) as exc:
        metrics["http_error"] = f"connection: {exc}"
        return metrics

    metrics["t_admitted"] = time.monotonic()
    if admitted_event is not None:
        admitted_event.set()

    try:
        for raw_line in resp:
            if stop_event is not None and stop_event.is_set():
                metrics["stopped_early"] = True
                break
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue  # keepalive / blank
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(obj.get("error"), dict):
                metrics["backend_error"] = obj["error"].get("message") or json.dumps(
                    obj["error"]
                )
                break

            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                emitted = False
                if delta.get("content"):
                    emitted = True
                for _tc in delta.get("tool_calls") or []:
                    emitted = True
                if emitted:
                    now = time.monotonic()
                    metrics["token_times"].append(now)
                    if metrics["t_first_token"] is None:
                        metrics["t_first_token"] = now
                        if first_token_event is not None:
                            first_token_event.set()
                if choice.get("finish_reason"):
                    metrics["finish_reason"] = choice["finish_reason"]
    finally:
        try:
            resp.close()
        except Exception:
            pass

    metrics["t_end"] = time.monotonic()
    return metrics


def gap_stats(metrics: dict, stall_threshold: float) -> dict:
    """Summarize a request's timing: TTFT, inter-token gaps, and stalls."""
    t0 = metrics.get("t0")
    t_first = metrics.get("t_first_token")
    t_end = metrics.get("t_end")
    times = metrics.get("token_times") or []

    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    gaps_sorted = sorted(gaps)
    stalls = [g for g in gaps if g >= stall_threshold]

    decode_s = (times[-1] - t_first) if (t_first and len(times) >= 2) else None
    # Buffered stream: engine withholds tokens until the generation finishes,
    # then flushes them in a tiny window. Client-side inter-token gaps are then
    # meaningless; TTFT ≈ total generation (+ wait) time is the real signal.
    buffered = bool(
        len(times) > 50 and decode_s is not None and decode_s < 1.0
    )
    tok_per_s = None
    if not buffered and decode_s and decode_s > 0:
        tok_per_s = round((len(times) - 1) / decode_s, 2)
    elif buffered and t0 and t_end and (t_end - t0) > 0:
        tok_per_s = round(len(times) / (t_end - t0), 2)

    return {
        "label": metrics.get("label"),
        "http_error": metrics.get("http_error"),
        "backend_error": metrics.get("backend_error"),
        "finish_reason": metrics.get("finish_reason"),
        "stopped_early": metrics.get("stopped_early"),
        "buffered_stream": buffered,
        "tokens": len(times),
        "ttft_s": round(t_first - t0, 2) if (t0 and t_first) else None,
        "admit_s": (
            round(metrics["t_admitted"] - t0, 3)
            if (t0 and metrics.get("t_admitted"))
            else None
        ),
        "total_s": round(t_end - t0, 1) if (t0 and t_end) else None,
        "decode_s": round(decode_s, 1) if decode_s is not None else None,
        "tok_per_s": tok_per_s,
        "gap_max_s": round(max(gaps), 2) if gaps else None,
        "gap_p50_s": round(_pct(gaps_sorted, 0.50), 3) if gaps else None,
        "gap_p95_s": round(_pct(gaps_sorted, 0.95), 3) if gaps else None,
        "stall_count": len(stalls),
        "stall_total_s": round(sum(stalls), 1) if stalls else 0.0,
        "stall_threshold_s": stall_threshold,
    }


def build_body(prompt: str, model: str, max_tokens: int, temperature: float) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def run_baseline(args) -> dict:
    body = build_body(
        VICTIM_PROMPT, args.model, args.victim_max_tokens, args.victim_temperature
    )
    m = stream_request(args.endpoint, body, args.client_timeout, label="victim-solo")
    return gap_stats(m, args.stall_threshold)


def run_contended(args) -> tuple[dict, dict]:
    stop_event = threading.Event()
    runaway_admitted = threading.Event()
    runaway_metrics: dict = {}

    def _runaway() -> None:
        body = build_body(
            RUNAWAY_PROMPT,
            args.model,
            args.runaway_max_tokens,
            args.runaway_temperature,
        )
        m = stream_request(
            args.endpoint,
            body,
            args.client_timeout,
            label="runaway",
            stop_event=stop_event,
            admitted_event=runaway_admitted,
        )
        runaway_metrics.update(m)

    t = threading.Thread(target=_runaway, daemon=True)
    t.start()

    # Wait until the runaway is *admitted* (HTTP response open / slot held), not
    # until its first client-visible token. On this engine the SSE stream is
    # often buffered to completion, so "first token" ≈ "whole generation done".
    if not runaway_admitted.wait(timeout=args.warmup_timeout):
        print(
            "warning: runaway not admitted within "
            f"{args.warmup_timeout:.0f}s; starting victim anyway",
            file=sys.stderr,
        )

    victim_body = build_body(
        VICTIM_PROMPT, args.model, args.victim_max_tokens, args.victim_temperature
    )
    victim_metrics = stream_request(
        args.endpoint, victim_body, args.client_timeout, label="victim-contended"
    )
    victim_card = gap_stats(victim_metrics, args.stall_threshold)

    # Tear the runaway down promptly so we do not linger on a shared engine.
    stop_event.set()
    t.join(timeout=10.0)
    runaway_card = gap_stats(runaway_metrics, args.stall_threshold) if runaway_metrics else {}
    return victim_card, runaway_card


def verdict(baseline: dict | None, contended: dict, stall_threshold: float) -> dict:
    """Decide whether worker monopolization / starvation reproduced.

    Two signals, because this engine often *buffers* the SSE stream (all tokens
    arrive in one end-of-generation burst):

    1. Mid-stream inter-token gaps — only meaningful on a live incremental stream.
    2. Contended TTFT vs solo baseline — on a buffered stream, TTFT ≈ wait-for-
       worker + generation. A victim that sits admitted while a runaway holds the
       single worker shows up as a huge TTFT inflation (not as mid-stream gaps).
    """
    reasons: list[str] = []
    reproduced = False

    if contended.get("http_error") or contended.get("backend_error"):
        return {
            "reproduced": None,
            "regime": "error",
            "note": contended.get("http_error") or contended.get("backend_error"),
        }

    gap_max = contended.get("gap_max_s") or 0.0
    stalls = contended.get("stall_count") or 0
    ttft = contended.get("ttft_s") or 0.0
    base_ttft = (baseline or {}).get("ttft_s") or 0.0
    base_gap = (baseline or {}).get("gap_max_s") or 0.0
    buffered = bool(contended.get("buffered_stream") or (baseline or {}).get("buffered_stream"))

    if stalls > 0 or gap_max >= stall_threshold:
        reproduced = True
        reasons.append(
            f"victim mid-stream gap_max={gap_max}s (>= {stall_threshold}s) "
            f"with {stalls} stall(s) totaling {contended.get('stall_total_s')}s"
        )
        regime = "resume-stall (started-then-dropped)"
    elif base_gap and gap_max >= max(5.0 * base_gap, 1.0) and not buffered:
        reproduced = True
        reasons.append(
            f"victim gap_max={gap_max}s is >=5x the solo baseline ({base_gap}s)"
        )
        regime = "degraded interleave"
    elif base_ttft > 0 and ttft >= max(stall_threshold, 3.0 * base_ttft):
        # On a buffered stream, TTFT ≈ wait-for-worker + generation. Distinguish
        # catastrophic monopolization (minutes) from the bounded wait for one
        # peer START quantum (tens of seconds under fair SCHED_STEP).
        wait_s = round(ttft - base_ttft, 1)
        reasons.append(
            f"victim ttft={ttft}s vs solo baseline {base_ttft}s "
            f"(~{wait_s}s extra wait while a concurrent request held the worker)"
        )
        if buffered:
            reasons.append(
                "SSE stream is buffered (tokens flush at end) — mid-stream gaps "
                "are not observable; TTFT inflation is the starvation signal"
            )
        # Catastrophic: multi-minute hold (legacy DRAIN / cold blocking generate).
        if wait_s >= 120.0 or ttft >= max(10.0 * base_ttft, 120.0):
            reproduced = True
            regime = "worker monopolization (DRAIN / no fair interleave)"
        else:
            # Bounded elevation — typically waiting out one peer START quantum
            # before fair SCHED_STEP sharing begins. Not the bug.
            regime = "bounded contention (first-quantum wait)"
            reasons.append(
                "extra wait is under 120s — consistent with waiting for one peer "
                "START quantum, then fair interleave (not monopolization)"
            )
    elif ttft >= stall_threshold and gap_max < stall_threshold and not buffered:
        regime = "head-of-line queueing (bounded)"
        reasons.append(
            f"victim ttft={ttft}s delayed but mid-stream gaps small "
            f"(gap_max={gap_max}s) — waited its turn, then ran steadily"
        )
    else:
        regime = "healthy under contention"
        reasons.append(
            f"victim gap_max={gap_max}s, ttft={ttft}s — no starvation observed"
        )

    return {
        "reproduced": reproduced,
        "regime": regime,
        "reasons": reasons,
        "buffered_stream": buffered,
    }


def print_human(baseline: dict | None, contended: dict, runaway: dict, v: dict) -> None:
    line = "─" * 66

    def _row(card: dict) -> None:
        if card.get("http_error") or card.get("backend_error"):
            print(f"    ERROR    : {card.get('http_error') or card.get('backend_error')}")
            return
        buf = "  [buffered SSE]" if card.get("buffered_stream") else ""
        print(
            f"    tokens={card.get('tokens')}  ttft={card.get('ttft_s')}s  "
            f"total={card.get('total_s')}s  tok/s={card.get('tok_per_s')}{buf}"
        )
        print(
            f"    gaps: max={card.get('gap_max_s')}s  p50={card.get('gap_p50_s')}s  "
            f"p95={card.get('gap_p95_s')}s"
        )
        print(
            f"    stalls>={card.get('stall_threshold_s')}s: {card.get('stall_count')} "
            f"(total {card.get('stall_total_s')}s)  finish={card.get('finish_reason')}"
        )

    print(line)
    print("  STARVATION REPRODUCTION")
    print(line)
    if baseline is not None:
        print("  BASELINE  victim solo (healthy reference):")
        _row(baseline)
        print()
    print("  CONTENDED victim alongside runaway:")
    _row(contended)
    if runaway:
        print()
        print("  runaway (load generator):")
        _row(runaway)
    print(line)
    print(f"  VERDICT   : reproduced={v.get('reproduced')}  regime={v.get('regime')}")
    for r in v.get("reasons") or []:
        print(f"              - {r}")
    if v.get("note"):
        print(f"              note: {v['note']}")
    print(line)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce the multi-slot started-then-dropped stall."
    )
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="engine base URL; hit the ENGINE directly to bypass the "
                         "600s proxy wall (default: %(default)s)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="served model id (default: %(default)s)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the solo victim baseline; only run the contended phase")
    ap.add_argument("--victim-max-tokens", type=int, default=400,
                    help="victim generation cap (default: %(default)s)")
    ap.add_argument("--victim-temperature", type=float, default=0.0,
                    help="victim temperature; low = steady cadence (default: %(default)s)")
    ap.add_argument("--runaway-max-tokens", type=int, default=6000,
                    help="runaway load-generator cap; large enough to keep the "
                         "worker busy through the victim (default: %(default)s)")
    ap.add_argument("--runaway-temperature", type=float, default=0.7,
                    help="runaway temperature (default: %(default)s)")
    ap.add_argument("--stall-threshold", type=float, default=5.0,
                    help="an inter-token gap >= this many seconds counts as a "
                         "stall (default: %(default)s)")
    ap.add_argument("--warmup-timeout", type=float, default=60.0,
                    help="max seconds to wait for the runaway's first token before "
                         "starting the victim (default: %(default)s)")
    ap.add_argument("--client-timeout", type=float, default=1800.0,
                    help="per-request client read timeout; keep > 600 so the wall "
                         "does not truncate the measurement (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="emit results as JSON only")
    args = ap.parse_args(argv)

    baseline = None
    if not args.no_baseline:
        if not args.json:
            print("# phase 1/2: baseline (victim solo)...", file=sys.stderr)
        baseline = run_baseline(args)

    if not args.json:
        print("# phase 2/2: contended (victim + runaway)...", file=sys.stderr)
    contended, runaway = run_contended(args)
    v = verdict(baseline, contended, args.stall_threshold)

    if args.json:
        print(json.dumps(
            {"baseline": baseline, "contended": contended, "runaway": runaway,
             "verdict": v},
            indent=2,
        ))
    else:
        print_human(baseline, contended, runaway, v)

    if contended.get("http_error") or contended.get("backend_error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
