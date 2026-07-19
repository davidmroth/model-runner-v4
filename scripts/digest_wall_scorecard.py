#!/usr/bin/env python3
"""Digest wall scorecard — read-only measurement of a single long generation.

Answers one question: for a real "Daily AI News Digest" turn, how far over the
600s proxy wall is the generation *solo*, and which lever closes the gap?

  verdict = output_tokens / decode_tokens_per_sec   vs   600s

It measures, it does not mutate: it replays one recorded request (or a supplied
prompt) against the engine, streams the result, and reports:

  - decode rate (tok/s) — both wall-clock-measured and server-reported
  - output token count
  - projected seconds to finish, and headroom against the 600s wall
  - max tokens that fit in 600s at the measured rate
  - whether the malformed <tool_call> leak recurs (validity signal)
  - draft commit/step + acceptance, if a correlation log is provided

Stdlib only (urllib, json, argparse). No third-party imports — the ai.local
containers have hit ModuleNotFoundError on yaml/httpx before.

Bypass the proxy and hit the engine directly (default) so the 600s wall does
NOT truncate the measurement; the wall is what we are measuring the turn
*against*, not with.

Usage (run inside a container on ai.local, per repo docker rule):

  # Replay the latest recorded briefing request against the engine directly
  python3 digest_wall_scorecard.py --auto --endpoint http://127.0.0.1:8080

  # Replay a specific dump, and parse a corr log for draft acceptance
  python3 digest_wall_scorecard.py \
      --dump /opt/data/sessions/request_dump_cron_XXXX.json \
      --endpoint http://127.0.0.1:8080 \
      --corr-log /opt/data/logs/dflash_corr.log

  # Ad-hoc prompt instead of a recorded dump
  python3 digest_wall_scorecard.py --endpoint http://127.0.0.1:8080 \
      --model qwen3.6-27b-autoround --max-tokens 32768 \
      --prompt "Write today's AI news digest with 12 stories..."
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
import urllib.error

WALL_SEC = 600.0
SESSIONS_GLOB = "/opt/data/sessions/request_dump_*.json"


def _fail(msg: str, code: int = 2) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def find_latest_dump() -> str:
    matches = sorted(glob.glob(SESSIONS_GLOB), key=os.path.getmtime, reverse=True)
    if not matches:
        _fail(f"no request dumps found matching {SESSIONS_GLOB}; pass --dump PATH")
    return matches[0]


def load_body_from_dump(path: str) -> dict:
    """Extract the chat-completions request body from a recorded dump.

    Dumps have drifted in shape over time, so be defensive: accept the body at
    the top level, or nested under a handful of known wrapper keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read dump {path}: {exc}")

    # Dump shapes have drifted: the body may be at the top level, nested under
    # request/request.body/payload, or JSON-encoded in a string. Search
    # recursively for the first dict that carries a 'messages' list.
    def _find_body(node: object, depth: int = 0) -> dict | None:
        if depth > 6:
            return None
        if isinstance(node, dict):
            if isinstance(node.get("messages"), list):
                return node
            for value in node.values():
                found = _find_body(value, depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, str) and node.lstrip()[:1] in "{[":
            try:
                return _find_body(json.loads(node), depth + 1)
            except json.JSONDecodeError:
                return None
        return None

    body = _find_body(raw)
    if body is None:
        _fail(f"dump {path} has no recognizable chat-completions body (no 'messages')")
    return body  # type: ignore[return-value]


def build_body_from_prompt(prompt: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def _looks_malformed_toolcall(text: str, tool_calls: list) -> tuple[bool, str]:
    """Heuristics for the known failure: tool-call syntax leaking into content,
    or tool-call arguments that are not valid JSON."""
    if text and ("<tool_call>" in text or "</tool_call>" in text):
        return True, "literal <tool_call> tags leaked into assistant content"
    for tc in tool_calls or []:
        args = (tc.get("function") or {}).get("arguments")
        if isinstance(args, str) and args.strip():
            try:
                json.loads(args)
            except json.JSONDecodeError:
                name = (tc.get("function") or {}).get("name") or "?"
                return True, f"tool_call '{name}' has non-JSON arguments"
    return False, ""


def stream_generation(endpoint: str, body: dict, client_timeout: float) -> dict:
    """POST a streaming chat-completions request and collect metrics.

    Returns a metrics dict. Never raises on a normal backend error — captures it
    so the scorecard still prints.
    """
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"

    req_body = dict(body)
    req_body["stream"] = True
    # Ask for a usage summary in the final SSE frame (OpenAI + llama.cpp honor this).
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
        "endpoint": url,
        "model": req_body.get("model"),
        "requested_max_tokens": req_body.get("max_tokens"),
        "http_error": None,
        "backend_error": None,
        "t0": None,
        "t_first_token": None,
        "t_end": None,
        "chunk_count": 0,
        "content_deltas": 0,
        "content_chars": 0,
        "server_timings": None,
        "usage": None,
    }
    content_parts: list[str] = []
    tool_calls: list[dict] = []

    t0 = time.monotonic()
    metrics["t0"] = t0
    try:
        resp = urllib.request.urlopen(request, timeout=client_timeout)
    except urllib.error.HTTPError as exc:
        try:
            metrics["http_error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}"
        except Exception:
            metrics["http_error"] = f"HTTP {exc.code}"
        return metrics
    except (urllib.error.URLError, TimeoutError) as exc:
        metrics["http_error"] = f"connection: {exc}"
        return metrics

    with resp:
        for raw_line in resp:
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
                metrics["backend_error"] = obj["error"].get("message") or json.dumps(obj["error"])
                break
            if isinstance(obj.get("timings"), dict):
                metrics["server_timings"] = obj["timings"]
            if isinstance(obj.get("usage"), dict):
                metrics["usage"] = obj["usage"]

            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if metrics["t_first_token"] is None:
                        metrics["t_first_token"] = time.monotonic()
                    content_parts.append(piece)
                    metrics["content_chars"] += len(piece)
                    metrics["content_deltas"] += 1
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    while len(tool_calls) <= idx:
                        tool_calls.append({"function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tool_calls[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                    if metrics["t_first_token"] is None:
                        metrics["t_first_token"] = time.monotonic()
            metrics["chunk_count"] += 1

    metrics["t_end"] = time.monotonic()
    text = "".join(content_parts)
    malformed, reason = _looks_malformed_toolcall(text, tool_calls)
    metrics["malformed"] = malformed
    metrics["malformed_reason"] = reason
    metrics["tool_calls"] = [
        {"name": tc["function"]["name"], "arg_chars": len(tc["function"]["arguments"])}
        for tc in tool_calls
    ]
    return metrics


def parse_corr_log(path: str, tail_lines: int = 4000) -> dict:
    """Extract draft acceptance / commit-per-step from a correlation log tail.

    Best-effort: the corr log format may drift, so we look for a few well-known
    tokens rather than a rigid schema. Returns {} if nothing is found.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-tail_lines:]
    except OSError as exc:
        return {"error": f"could not read corr log: {exc}"}

    commits: list[int] = []
    accepts: list[float] = []
    for line in lines:
        low = line.lower()
        for key in ("commit/step", "commit_per_step", "commits=", "accepted="):
            if key in low:
                for tok in low.replace("=", " ").replace(",", " ").split():
                    try:
                        val = float(tok)
                    except ValueError:
                        continue
                    if "accept" in low and 0.0 <= val <= 1.0:
                        accepts.append(val)
                    elif 0.0 < val < 64.0:
                        commits.append(int(val))
                break
    out: dict = {}
    if commits:
        out["commit_per_step_avg"] = round(sum(commits) / len(commits), 2)
        out["commit_per_step_samples"] = len(commits)
    if accepts:
        out["acceptance_avg"] = round(sum(accepts) / len(accepts), 3)
        out["acceptance_samples"] = len(accepts)
    return out


def scorecard(metrics: dict, corr: dict) -> dict:
    t0 = metrics["t0"]
    t_first = metrics["t_first_token"]
    t_end = metrics["t_end"]

    total_s = (t_end - t0) if (t0 and t_end) else None
    ttft_s = (t_first - t0) if (t0 and t_first) else None
    decode_s = (t_end - t_first) if (t_first and t_end) else None

    timings = metrics.get("server_timings") or {}
    usage = metrics.get("usage") or {}

    out_tokens = (
        usage.get("completion_tokens")
        or timings.get("predicted_n")
        or None
    )
    token_source = "usage/timings"
    if not out_tokens and metrics.get("content_deltas"):
        # Custom server returned no usage block; each streamed content delta is
        # ~1 token for this engine, so use the delta count as a proxy.
        out_tokens = metrics["content_deltas"]
        token_source = "content_deltas (proxy)"
    server_tps = timings.get("predicted_per_second")

    # Detect a buffered (non-incremental) stream: the server withholds all
    # tokens then flushes them in a tiny window, making decode_s meaningless and
    # measured tok/s absurdly high. In that case the only defensible rate is a
    # FLOOR of output_tokens / total_s (which still includes prefill), and the
    # authoritative decode rate must come from the engine's own daemon log.
    buffered = bool(
        out_tokens and out_tokens > 50 and decode_s is not None and decode_s < 1.0
    )

    measured_tps = None
    rate_note = None
    if buffered:
        rate_note = "buffered stream — decode_s unreliable; see daemon decode_tok_s"
        if out_tokens and total_s and total_s > 0:
            measured_tps = round(out_tokens / total_s, 2)  # floor (incl prefill)
    elif out_tokens and decode_s and decode_s > 0:
        measured_tps = round(out_tokens / decode_s, 2)

    tps = server_tps or measured_tps
    projected_finish_s = None
    headroom_s = None
    max_tokens_in_wall = None
    if out_tokens and tps:
        projected_finish_s = round(out_tokens / tps, 1)
        headroom_s = round(WALL_SEC - projected_finish_s, 1)
        max_tokens_in_wall = int(tps * WALL_SEC)

    verdict = "unknown"
    if buffered:
        verdict = "UNKNOWN (buffered stream — read decode_tok_s from daemon log)"
    elif projected_finish_s is not None:
        verdict = "FITS the 600s wall" if projected_finish_s <= WALL_SEC else "OVER the 600s wall"

    return {
        "endpoint": metrics.get("endpoint"),
        "model": metrics.get("model"),
        "http_error": metrics.get("http_error"),
        "backend_error": metrics.get("backend_error"),
        "output_tokens": out_tokens,
        "output_token_source": token_source,
        "buffered_stream": buffered,
        "rate_note": rate_note,
        "decode_tok_per_sec_measured": measured_tps,
        "decode_tok_per_sec_server": server_tps,
        "ttft_s": round(ttft_s, 2) if ttft_s else None,
        "decode_s": round(decode_s, 1) if decode_s else None,
        "total_s": round(total_s, 1) if total_s else None,
        "projected_finish_s": projected_finish_s,
        "headroom_vs_600s": headroom_s,
        "max_tokens_in_600s": max_tokens_in_wall,
        "verdict": verdict,
        "malformed_tool_output": metrics.get("malformed"),
        "malformed_reason": metrics.get("malformed_reason") or None,
        "tool_calls": metrics.get("tool_calls"),
        "chunk_count": metrics.get("chunk_count"),
        "draft_stats": corr or None,
    }


def print_human(card: dict) -> None:
    line = "─" * 60
    print(line)
    print("  DIGEST WALL SCORECARD")
    print(line)
    if card.get("http_error"):
        print(f"  HTTP error : {card['http_error']}")
    if card.get("backend_error"):
        print(f"  backend    : {card['backend_error']}")
    print(f"  endpoint   : {card.get('endpoint')}")
    print(f"  model      : {card.get('model')}")
    print(f"  output tok : {card.get('output_tokens')} "
          f"[{card.get('output_token_source')}]")
    print(f"  tok/s      : measured={card.get('decode_tok_per_sec_measured')} "
          f"server={card.get('decode_tok_per_sec_server')}")
    if card.get("rate_note"):
        print(f"  note       : {card['rate_note']}")
    print(f"  ttft       : {card.get('ttft_s')}s   decode: {card.get('decode_s')}s   "
          f"total: {card.get('total_s')}s")
    print(f"  projected  : {card.get('projected_finish_s')}s to finish "
          f"(headroom vs 600s: {card.get('headroom_vs_600s')}s)")
    print(f"  ceiling    : ~{card.get('max_tokens_in_600s')} tokens fit in 600s "
          f"at this rate")
    print(f"  VERDICT    : {card.get('verdict')}")
    mal = card.get("malformed_tool_output")
    print(f"  malformed  : {mal}"
          + (f"  ({card.get('malformed_reason')})" if mal else ""))
    if card.get("draft_stats"):
        print(f"  draft      : {card['draft_stats']}")
    print(line)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Read-only 600s-wall scorecard for one long generation.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dump", help="path to a recorded request-dump JSON")
    src.add_argument("--auto", action="store_true",
                     help=f"use the newest dump matching {SESSIONS_GLOB}")
    ap.add_argument("--prompt", help="ad-hoc user prompt (instead of a dump)")
    ap.add_argument("--model", default="qwen3.6-27b-autoround",
                    help="model id for --prompt mode (default: %(default)s)")
    ap.add_argument("--force-model",
                    help="override body['model'] in any mode (direct engine hits "
                         "skip the proxy's alias rewrite, so set the served id here)")
    ap.add_argument("--tool-choice",
                    help="set body['tool_choice']; parsed as JSON when possible, "
                         "else used as a string. e.g. 'required' or "
                         "'{\"type\":\"function\",\"function\":{\"name\":\"create_briefing\"}}'")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max_tokens; overrides the dump's value when set. Use a "
                         "small cap (e.g. 2000) for a fast, low-impact rate sample "
                         "that still yields tok/s and the 600s ceiling. "
                         "Defaults to 32768 in --prompt mode.")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080",
                    help="engine base URL; hit the ENGINE directly to bypass the "
                         "600s proxy wall (default: %(default)s)")
    ap.add_argument("--corr-log", help="optional correlation log to parse for draft acceptance")
    ap.add_argument("--client-timeout", type=float, default=1800.0,
                    help="client read timeout in seconds; keep > 600 so the wall "
                         "does not truncate the measurement (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="emit the scorecard as JSON only")
    args = ap.parse_args(argv)

    if args.prompt:
        body = build_body_from_prompt(args.prompt, args.model, args.max_tokens or 32768)
    else:
        dump_path = args.dump or (find_latest_dump() if args.auto else None)
        if not dump_path:
            _fail("provide one of --dump PATH, --auto, or --prompt TEXT")
        body = load_body_from_dump(dump_path)
        if not args.json:
            print(f"# replaying dump: {dump_path}", file=sys.stderr)

    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens
    if args.force_model:
        body["model"] = args.force_model
    if args.tool_choice:
        try:
            body["tool_choice"] = json.loads(args.tool_choice)
        except json.JSONDecodeError:
            body["tool_choice"] = args.tool_choice

    metrics = stream_generation(args.endpoint, body, args.client_timeout)
    corr = parse_corr_log(args.corr_log) if args.corr_log else {}
    card = scorecard(metrics, corr)

    if args.json:
        print(json.dumps(card, indent=2))
    else:
        print_human(card)

    if card.get("http_error") or card.get("backend_error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
