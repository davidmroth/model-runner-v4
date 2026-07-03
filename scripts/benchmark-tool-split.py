#!/usr/bin/env python3
"""Benchmark + validate tool-split deploy via ai-platform proxy (:8000)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b-autoround"
LOG_CONTAINER = sys.argv[3] if len(sys.argv) > 3 else "model-runner-v4-lucebox"

TOOLS = [
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


@dataclass
class Sample:
    name: str
    elapsed_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prefill_ms: float | None = None
    decode_tps: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def chat(
    messages: list[dict],
    *,
    tools: list | None = None,
    max_tokens: int = 32,
    timeout: int = 600,
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
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    elapsed = time.time() - t0
    u = body.get("usage") or {}
    ut = u.get("timings") or {}
    sample = Sample(
        name="",
        elapsed_s=round(elapsed, 2),
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=u.get("completion_tokens"),
        prefill_ms=ut.get("prefill_ms"),
        decode_tps=ut.get("decode_tokens_per_sec"),
    )
    if return_body:
        return sample, body
    return sample


def ref_bench() -> Sample:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "from typing import List\n\n"
                "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
                "    for"
            ),
        }],
        "max_tokens": 256,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.load(resp)
    ut = (body.get("usage") or {}).get("timings") or {}
    return Sample(
        name="ref_decode",
        elapsed_s=round(time.time() - t0, 2),
        prefill_ms=ut.get("prefill_ms"),
        decode_tps=ut.get("decode_tokens_per_sec"),
    )


def docker_logs(since: str = "2m") -> str:
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--since", since, LOG_CONTAINER],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        return out
    except Exception as exc:
        return f"(log fetch failed: {exc})"


def count_markers(log: str) -> dict[str, int]:
    keys = [
        "tool-split",
        "RESTORE_CHAIN",
        "SNAPSHOT_THIN",
        "tool KV pinned",
        "restore=true",
        "inline-snap committed",
    ]
    return {k: log.count(k) for k in keys}


def main() -> int:
    print(f"=== Tool-split benchmark ===")
    print(f"base={BASE} model={MODEL} log_container={LOG_CONTAINER}\n")

    # Health
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=5).read()
        print("health: OK /v1/models")
    except Exception as exc:
        print(f"health: FAIL {exc}")
        return 1

    results: list[Sample] = []

    print("\n--- 1. Reference decode (no tools) ---")
    try:
        r = ref_bench()
        r.name = "ref_decode"
        results.append(r)
        print(f"  {r}")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    print("\n--- 2. Agent loop: cold then after tool result (practical hot path) ---")
    agent_msgs: list[dict] = [
        {"role": "user", "content": "Use read_file to read /etc/hostname. One line."},
    ]
    try:
        r_agent_cold, cold_body = chat(
            agent_msgs, tools=TOOLS, max_tokens=64, return_body=True,
        )
        r_agent_cold.name = "agent_turn1_cold"
        results.append(r_agent_cold)
        print(f"  cold: {r_agent_cold}")
        asst = cold_body["choices"][0]["message"]
        tool_calls = asst.get("tool_calls") or []
        if tool_calls:
            agent_msgs.append(asst)
            tc = tool_calls[0]
            agent_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
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
        r_agent_hot = chat(agent_msgs, tools=TOOLS, max_tokens=24)
        r_agent_hot.name = "agent_after_tool"
        results.append(r_agent_hot)
        print(f"  after tool result: {r_agent_hot}")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    time.sleep(1)
    print("\n--- 3. Tool session: turn 1 (cold tool KV) ---")
    msgs = [{"role": "user", "content": "Remember this code: " + ("x = 1\n" * 400)}]
    try:
        r1 = chat(msgs, tools=TOOLS, max_tokens=12)
        r1.name = "tools_turn1"
        results.append(r1)
        print(f"  {r1}")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    time.sleep(1)
    print("\n--- 4. Tool session: turn 2 (RESTORE_CHAIN) ---")
    msgs += [{"role": "assistant", "content": "Stored."}, {"role": "user", "content": "What variable did I assign? One word."}]
    try:
        r2 = chat(msgs, tools=TOOLS, max_tokens=8)
        r2.name = "tools_turn2"
        results.append(r2)
        print(f"  {r2}")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    time.sleep(1)
    print("\n--- 5. Tool session: turn 3 (small follow-up — practical speed) ---")
    msgs += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "Say OK only."}]
    try:
        r3 = chat(msgs, tools=TOOLS, max_tokens=4)
        r3.name = "tools_turn3"
        results.append(r3)
        print(f"  {r3}")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    print("\n--- 6. Log validation (last 3m) ---")
    log = docker_logs("3m")
    markers = count_markers(log)
    for k, v in markers.items():
        print(f"  {k}: {v}")
    interesting = [ln for ln in log.splitlines() if "tool-split" in ln or "RESTORE_CHAIN" in ln or "thin slot" in ln or "restore=true" in ln][-20:]
    if interesting:
        print("  recent lines:")
        for ln in interesting:
            print(f"    {ln}")

    print("\n--- 7. Validation checks ---")
    ok = True
    t1 = next(s for s in results if s.name == "tools_turn1")
    t2 = next(s for s in results if s.name == "tools_turn2")
    t3 = next(s for s in results if s.name == "tools_turn3")
    agent_cold = next((s for s in results if s.name == "agent_turn1_cold"), None)
    agent_hot = next((s for s in results if s.name == "agent_after_tool"), None)
    ref = next(s for s in results if s.name == "ref_decode")

    if t1.prefill_ms and t3.prefill_ms:
        ratio = t3.prefill_ms / t1.prefill_ms
        speedup = t1.prefill_ms / max(t3.prefill_ms, 1)
        print(
            f"  incremental prefill: turn3 {t3.prefill_ms}ms vs "
            f"turn1 {t1.prefill_ms}ms ({speedup:.1f}x)"
        )
    if t1.elapsed_s and t3.elapsed_s:
        wall = t1.elapsed_s / max(t3.elapsed_s, 0.01)
        print(
            f"  incremental wall-clock: turn3 {t3.elapsed_s}s vs "
            f"turn1 {t1.elapsed_s}s ({wall:.1f}x faster)"
        )
    if agent_cold and agent_hot:
        print(
            f"  agent after tool: elapsed={agent_hot.elapsed_s}s "
            f"prefill_ms={agent_hot.prefill_ms} "
            f"(cold was {agent_cold.elapsed_s}s / {agent_cold.prefill_ms}ms)"
        )

    if ref.decode_tps and ref.decode_tps < 50:
        print(f"  WARN: ref decode {ref.decode_tps} tok/s < 50 (expected ~76 with layer_split=0)")
    else:
        print(f"  PASS: ref decode {ref.decode_tps} tok/s")

    if t1.prefill_ms and t3.prefill_ms and (
        t3.prefill_ms < t1.prefill_ms * 0.25 or t3.prefill_ms < 500
    ):
        print(f"  PASS: turn3 incremental prefill {t3.prefill_ms}ms << turn1 {t1.prefill_ms}ms")
    elif t2.prefill_ms and t2.prefill_ms < t1.prefill_ms * 0.25:
        print(f"  PASS: turn2 prefill {t2.prefill_ms}ms << turn1 {t1.prefill_ms}ms")
    elif t2.prefill_ms and t2.prefill_ms < 2000:
        print(f"  PASS: turn2 prefill {t2.prefill_ms}ms (sub-2s)")
    else:
        print(
            f"  WARN: incremental prefill not improved "
            f"(turn3={t3.prefill_ms} turn2={t2.prefill_ms} turn1={t1.prefill_ms} ms)"
        )
        ok = False

    if agent_hot and (
        (agent_hot.elapsed_s is not None and agent_hot.elapsed_s < 4.0)
        or (agent_hot.prefill_ms is not None and agent_hot.prefill_ms < 500)
    ):
        print(f"  PASS: agent after tool feels fast ({agent_hot.elapsed_s}s elapsed)")
    elif agent_hot:
        print(
            f"  WARN: agent after tool still slow "
            f"(elapsed={agent_hot.elapsed_s}s prefill_ms={agent_hot.prefill_ms})"
        )
        ok = False

    if t3.elapsed_s and t1.elapsed_s and t3.elapsed_s < t1.elapsed_s * 0.6:
        print(f"  PASS: turn3 wall-clock {t3.elapsed_s}s < 60% of turn1 {t1.elapsed_s}s")
    elif t3.elapsed_s and t1.elapsed_s:
        print(
            f"  WARN: turn3 wall-clock {t3.elapsed_s}s not much faster than "
            f"turn1 {t1.elapsed_s}s (user-visible)"
        )

    if markers.get("inline-snap committed", 0) >= 1:
        print("  PASS: conv prefix inline-snap committed")
    else:
        print("  WARN: no inline-snap committed (conv cache inactive — check VRAM/OOM logs)")
        ok = False

    if markers.get("tool KV pinned", 0) >= 1 or markers.get("SNAPSHOT_THIN", 0) >= 1:
        print("  PASS: tool KV snapshot seen in logs")
    else:
        print("  WARN: no SNAPSHOT_THIN / tool KV pinned in logs (turn1 may have missed snap)")
        ok = False

    if markers.get("RESTORE_CHAIN", 0) >= 1 or markers.get("restore=true", 0) >= 1:
        print("  PASS: cache restore activity in logs")
    else:
        print("  WARN: no RESTORE_CHAIN / restore=true in logs")
        ok = False

    print("\n--- Summary ---")
    for s in results:
        print(f"  {s.name}: elapsed={s.elapsed_s}s prefill_ms={s.prefill_ms} decode_tps={s.decode_tps} pt={s.prompt_tokens}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
