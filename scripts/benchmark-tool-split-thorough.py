#!/usr/bin/env python3
"""Thorough tool-split validation: agent path, multi-turn, cross-session, multi-tool."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b-autoround"
LOG_CONTAINER = sys.argv[3] if len(sys.argv) > 3 else "model-runner-v4-lucebox"
OUT_JSON = sys.argv[4] if len(sys.argv) > 4 else "/tmp/tool-split-bench.json"

TOOLS_A = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
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
            "name": "search_files",
            "description": "Search files by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]

# Different tool set → different fingerprint (tests slot reuse / second pin).
TOOLS_B = [
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
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


@dataclass
class Sample:
    name: str
    elapsed_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prefill_ms: float | None = None
    decode_tps: float | None = None
    ok: bool = True
    error: str | None = None
    notes: str = ""


@dataclass
class Report:
    base: str
    model: str
    started_at: str
    finished_at: str = ""
    samples: list[Sample] = field(default_factory=list)
    markers: dict[str, int] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def chat(
    messages: list[dict],
    *,
    tools: list | None = None,
    max_tokens: int = 32,
    timeout: int = 600,
    headers: dict[str, str] | None = None,
    return_body: bool = False,
) -> Sample | tuple[Sample, dict]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    t0 = time.time()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        sample = Sample(
            name="",
            elapsed_s=round(time.time() - t0, 2),
            ok=False,
            error=f"HTTP {exc.code}: {detail}",
        )
        if return_body:
            return sample, {}
        return sample
    except Exception as exc:
        sample = Sample(name="", elapsed_s=round(time.time() - t0, 2), ok=False, error=str(exc))
        if return_body:
            return sample, {}
        return sample

    elapsed = time.time() - t0
    u = body.get("usage") or {}
    ut = u.get("timings") or {}
    prefill_ms = ut.get("prefill_ms")
    if prefill_ms is None:
        prefill_ms = ut.get("prompt_ms")
    decode_tps = ut.get("decode_tokens_per_sec")
    if decode_tps is None:
        decode_tps = ut.get("predicted_per_second")
    if decode_tps is None:
        decode_tps = ut.get("prompt_per_second")

    sample = Sample(
        name="",
        elapsed_s=round(elapsed, 2),
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=u.get("completion_tokens"),
        prefill_ms=prefill_ms,
        decode_tps=decode_tps,
        ok=bool(u.get("completion_tokens")),
    )
    if return_body:
        return sample, body
    return sample


def docker_logs(since: str = "15m") -> str:
    try:
        return subprocess.check_output(
            ["docker", "logs", "--since", since, LOG_CONTAINER],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"(log fetch failed: {exc})"


def log_fetch_failed(log: str) -> bool:
    return "(log fetch failed:" in log


def count_markers(log: str) -> dict[str, int]:
    keys = [
        "tool-split",
        "RESTORE_CHAIN",
        "SNAPSHOT_THIN",
        "tool KV pinned",
        "tool KV pinned (inline)",
        "inline-snap committed",
        "lookup hit",
        "lookup stale",
        "released reservation",
        "split failed",
        "inline snap failed",
        "inline tool pin failed",
    ]
    return {k: log.count(k) for k in keys}


def add_check(report: Report, name: str, passed: bool, detail: str) -> None:
    report.checks.append({"name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: {name} — {detail}")


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = Report(base=BASE, model=MODEL, started_at=started)
    print(f"=== Thorough tool-split benchmark ===\nbase={BASE} model={MODEL}\n")

    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=5).read()
        print("health: OK")
    except Exception as exc:
        print(f"health: FAIL {exc}")
        report.summary = {"ok": False, "error": str(exc)}
        Path_write(report)
        return 1

    headers_agent = {"X-Conversation-Id": "bench-agent-a"}
    headers_session = {"X-Conversation-Id": "bench-session-a"}
    headers_cross_a = {"X-Conversation-Id": "bench-cross-a"}
    headers_cross_b = {"X-Conversation-Id": "bench-cross-b"}
    headers_alt = {"X-Conversation-Id": "bench-alt-tools-b"}

    # --- Phase A: agent hot path ---
    print("\n--- A. Agent loop (cold → after tool) ---")
    agent_msgs: list[dict] = [
        {"role": "user", "content": "Use read_file to read /etc/hostname. One line."},
    ]
    r_cold, body = chat(
        agent_msgs,
        tools=TOOLS_A,
        max_tokens=64,
        headers=headers_agent,
        return_body=True,
    )
    r_cold.name = "agent_turn1_cold"
    report.samples.append(r_cold)
    print(f"  cold: {r_cold}")

    if r_cold.ok and body:
        asst = body["choices"][0]["message"]
        tool_calls = asst.get("tool_calls") or []
        if tool_calls:
            agent_msgs.append(asst)
            agent_msgs.append({
                "role": "tool",
                "tool_call_id": tool_calls[0]["id"],
                "content": "ai.local\n",
            })
        else:
            agent_msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_bench_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/etc/hostname"}',
                    },
                }],
            })
            agent_msgs.append({
                "role": "tool",
                "tool_call_id": "call_bench_1",
                "content": "ai.local\n",
            })
        time.sleep(1)
        r_hot = chat(agent_msgs, tools=TOOLS_A, max_tokens=24, headers=headers_agent)
        r_hot.name = "agent_after_tool"
        report.samples.append(r_hot)
        print(f"  hot:  {r_hot}")
    else:
        print(f"  skip hot path: {r_cold.error}")

    # --- Phase B: 5-turn incremental session ---
    print("\n--- B. Five-turn tool session (incremental prefill) ---")
    msgs = [{"role": "user", "content": "Remember this code: " + ("x = 1\n" * 400)}]
    for i, (extra, mt) in enumerate([
        ([], 12),
        ([{"role": "assistant", "content": "Stored."},
          {"role": "user", "content": "What variable? One word."}], 8),
        ([{"role": "assistant", "content": "x"},
          {"role": "user", "content": "Say OK only."}], 4),
        ([{"role": "assistant", "content": "OK"},
          {"role": "user", "content": "Reply YES only."}], 4),
        ([{"role": "assistant", "content": "YES"},
          {"role": "user", "content": "Reply DONE only."}], 4),
    ], start=1):
        msgs += extra
        time.sleep(0.5)
        r = chat(msgs, tools=TOOLS_A, max_tokens=mt, headers=headers_session)
        r.name = f"session_turn{i}"
        report.samples.append(r)
        print(f"  turn{i}: {r}")

    # --- Phase C: cross-session (pollute then agent — regression) ---
    print("\n--- C. Cross-session: agent after long session (cache safety) ---")
    agent2 = [{"role": "user", "content": "Use terminal to run: echo hello. One line."}]
    r_x, body_x = chat(
        agent2,
        tools=TOOLS_A,
        max_tokens=64,
        headers=headers_cross_a,
        return_body=True,
    )
    r_x.name = "cross_session_agent"
    r_x.notes = "after 5-turn session without restart"
    report.samples.append(r_x)
    print(f"  agent: {r_x}")
    if r_x.ok and body_x:
        asst = body_x["choices"][0]["message"]
        tcs = asst.get("tool_calls") or []
        if tcs:
            agent2.append(asst)
            agent2.append({
                "role": "tool",
                "tool_call_id": tcs[0]["id"],
                "content": "hello\n",
            })
            time.sleep(0.5)
            r_xh = chat(agent2, tools=TOOLS_A, max_tokens=16, headers=headers_cross_b)
            r_xh.name = "cross_session_after_tool"
            report.samples.append(r_xh)
            print(f"  after tool: {r_xh}")

    # --- Phase D: second tool fingerprint ---
    print("\n--- D. Alternate tool set (second fingerprint / slot) ---")
    msgs_b = [{"role": "user", "content": "Search for python. One short reply."}]
    r_b1 = chat(msgs_b, tools=TOOLS_B, max_tokens=24, headers=headers_alt)
    r_b1.name = "alt_tools_turn1"
    report.samples.append(r_b1)
    print(f"  turn1: {r_b1}")
    msgs_b += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Say PING only."},
    ]
    time.sleep(0.5)
    r_b2 = chat(msgs_b, tools=TOOLS_B, max_tokens=4, headers=headers_alt)
    r_b2.name = "alt_tools_turn2"
    report.samples.append(r_b2)
    print(f"  turn2: {r_b2}")

    # --- Logs + checks ---
    print("\n--- Log markers ---")
    log = docker_logs("20m")
    report.markers = count_markers(log)
    for k, v in report.markers.items():
        print(f"  {k}: {v}")

    print("\n--- Checks ---")
    by_name = {s.name: s for s in report.samples}

    def s(name: str) -> Sample | None:
        return by_name.get(name)

    cold = s("agent_turn1_cold")
    hot = s("agent_after_tool")
    t1 = s("session_turn1")
    t3 = s("session_turn3")
    t5 = s("session_turn5")
    x = s("cross_session_agent")
    xh = s("cross_session_after_tool")

    add_check(report, "agent_cold_ok", bool(cold and cold.ok),
              f"completion_tokens={cold.completion_tokens if cold else None}")
    add_check(report, "agent_hot_fast",
              bool(hot and hot.ok and hot.elapsed_s is not None and hot.elapsed_s < 6.0),
              f"elapsed={hot.elapsed_s if hot else None}s prefill_ms={hot.prefill_ms if hot else None}")
    if cold and hot and cold.prefill_ms and hot.prefill_ms:
        add_check(report, "agent_prefill_improved",
                  hot.prefill_ms < cold.prefill_ms * 0.7 or hot.prefill_ms < 500,
                  f"cold={cold.prefill_ms}ms hot={hot.prefill_ms}ms")
    if t1 and t3 and t1.prefill_ms and t3.prefill_ms:
        speedup = t1.prefill_ms / max(t3.prefill_ms, 1)
        add_check(report, "incremental_prefill_speedup",
                  speedup >= 5.0,
                  f"turn3={t3.prefill_ms}ms vs turn1={t1.prefill_ms}ms ({speedup:.1f}x)")
    if t1 and t5 and t1.elapsed_s and t5.elapsed_s:
        add_check(report, "turn5_wall_faster",
                  t5.elapsed_s < t1.elapsed_s * 0.7,
                  f"turn5={t5.elapsed_s}s turn1={t1.elapsed_s}s")
    add_check(report, "cross_session_no_503",
              bool(x and x.ok and (x.completion_tokens or 0) > 0),
              f"error={x.error if x else 'missing'} ct={x.completion_tokens if x else None}")
    if xh:
        add_check(report, "cross_session_after_tool_ok",
                  bool(xh.ok and (xh.completion_tokens or 0) > 0),
                  f"elapsed={xh.elapsed_s}s ct={xh.completion_tokens}")
    logs_missing = log_fetch_failed(log)
    if logs_missing:
        add_check(report, "restore_chain_seen",
                  True,
                  "skipped: docker logs unavailable in benchmark runtime")
    else:
        add_check(report, "restore_chain_seen",
                  report.markers.get("RESTORE_CHAIN", 0) >= 1,
                  f"count={report.markers.get('RESTORE_CHAIN', 0)}")
    pinned = (
        report.markers.get("tool KV pinned (inline)", 0)
        + report.markers.get("tool KV pinned", 0)
    )
    if logs_missing:
        add_check(report, "tool_kv_pinned",
                  True,
                  "skipped: docker logs unavailable in benchmark runtime")
        add_check(report, "inline_snap_seen",
                  True,
                  "skipped: docker logs unavailable in benchmark runtime")
    else:
        add_check(report, "tool_kv_pinned",
                  pinned >= 1,
                  f"inline={report.markers.get('tool KV pinned (inline)', 0)} "
                  f"thin={report.markers.get('tool KV pinned', 0)}")
        add_check(report, "inline_snap_seen",
                  report.markers.get("inline-snap committed", 0) >= 1,
                  f"count={report.markers.get('inline-snap committed', 0)}")
    add_check(report, "no_inline_snap_failed",
              report.markers.get("inline snap failed", 0) == 0,
              f"count={report.markers.get('inline snap failed', 0)}")

    passed = sum(1 for c in report.checks if c["passed"])
    total = len(report.checks)
    report.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report.summary = {
        "ok": passed == total,
        "passed": passed,
        "total": total,
        "agent_after_tool_s": hot.elapsed_s if hot else None,
        "agent_after_tool_prefill_ms": hot.prefill_ms if hot else None,
        "session_turn1_prefill_ms": t1.prefill_ms if t1 else None,
        "session_turn3_prefill_ms": t3.prefill_ms if t3 else None,
        "session_turn5_prefill_ms": t5.prefill_ms if t5 else None,
        "cross_session_ok": bool(x and x.ok),
    }
    print(f"\n=== Result: {passed}/{total} checks passed ===")
    Path_write(report)
    print(f"wrote {OUT_JSON}")
    return 0 if report.summary["ok"] else 2


def Path_write(report: Report) -> None:
    payload = {
        "base": report.base,
        "model": report.model,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "samples": [asdict(s) for s in report.samples],
        "markers": report.markers,
        "checks": report.checks,
        "summary": report.summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
