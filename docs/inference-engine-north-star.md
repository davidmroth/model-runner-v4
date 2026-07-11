# Inference Engine North Star

**Status:** July 2026 · active  
**Owner stack:** `model-runner-v4` → `lucebox-hub` (`test_dflash` + `server_tools.py`) → `ai-platform` proxy  
**Consumers:** Hermes agent, WebUI (must not be “fixed” by disabling features)

This document states the **non-negotiable product goal** for the inference engine.
Tactical gates and deploy steps live in
[engine-certification-plan.md](./engine-certification-plan.md). Program context
lives in [agent-inference-program.md](./agent-inference-program.md).

---

## Goal (one sentence)

Ship a **multimodal**, **session-cached**, **tool-split** inference engine on 2× RTX
3090 that is **stable** (no empty or stuck turns), with **low TTFT** and **high
decode TPS** — proven end-to-end through the ai-platform proxy, not by rolling
back to legacy `dflash_server`.

---

## Success metrics

| Dimension | Target | How we prove it |
|-----------|--------|-----------------|
| **Stability** | Zero stuck streams; valid `tool_calls` on tool turns | WebUI/Hermes sessions complete; cert script exit 0 |
| **TTFT** | Warm prefill @ ~8.5K tokens **≤ 0.5 s**; first SSE chunk **< 2 s** on smoke prompt | Proxy + engine logs (`prefill_ms`, `ttfb_ms`) |
| **TPS** | Decode **≥ 50 tok/s** warm (context-dependent; whitepaper 51–94) | Daemon `[dflash]` / `decode_tok_s` in logs + cert benches |
| **Wall time** | Agent turn after tool result **< 6 s** | `agent_after_tool_s` in certification |
| **Multimodal** | Native mmproj vision path (`DFLASH_MMPROJ`) works through OpenAI API | Vision smoke + image turn in cert |
| **Session cache** | `X-Conversation-Id` scoped prefix cache; `RESTORE_CHAIN` for tool-split | Log lines `[tool-split] RESTORE_CHAIN`; turn-3 prefill ≥ 5× turn-1 |

---

## Required configuration (not optional)

```bash
DFLASH_TOOL_SPLIT_ENABLED=1          # server_tools.py + test_dflash daemon
DFLASH_LAYER_SPLIT=1                 # 2× GPU layer split
DFLASH_PREFILL_MODE=off              # PFlash conv compress breaks tool calls
DFLASH_TOOL_SPLIT_COMPRESS_CONV=0
DFLASH_DRAFT_FEATURE_MIRROR=1
DFLASH_LEGACY_DAEMON=1               # SNAPSHOT_THIN / RESTORE_CHAIN protocol
DFLASH_TOOL_INLINE_SNAP_PIN=1        # Phase 1c: pin tool KV via inline snap= (20K+ schemas)
```

Agent traffic must send `X-Conversation-Id: <uuid>` per session.

---

## What is **not** a solution

| Workaround | Why it fails the goal |
|------------|----------------------|
| `DFLASH_TOOL_SPLIT_ENABLED=0` (legacy `dflash_server` only) | No `RESTORE_CHAIN`, no tool KV pins, no cert path for agent cache |
| Disabling prefix cache globally | Destroys TTFT/TPS on multi-turn agent loops |
| Hermes timeout / retry tweaks | Masks engine hang; user still sees empty/stuck turns |
| Disabling proxy streaming | Hermes and WebUI require streamed completions |

---

## Proof bar (ship checklist)

All must pass on `ai.local` (and production) before calling the engine fixed:

1. **Direct engine stream** — `curl` to `:8080` → role + content + `[DONE]` in < 5 s for `"Reply pong"`.
2. **Proxy stream** — same through `qwen3.6-27b-autoround` on `:8000`.
3. **Agent-shaped** — tools + ~6K system prompt + `X-Conversation-Id`; no 180 s stale provider loops.
4. **Certification** — `scripts/run-engine-certification.sh` exit 0 through proxy.
5. **Multimodal smoke** — image + text turn when mmproj is enabled (see [vision-integration.md](./vision-integration.md)).

---

## Architecture (reference)

```
Hermes / WebUI
    → ai-platform-proxy:8000  (alias qwen3.6-27b-autoround, forwards X-Conversation-Id)
    → model-runner-v4-lucebox:8080
          server_tools.py  (FastAPI, tool parse, prefix cache, RESTORE_CHAIN)
          ↔ test_dflash --daemon --stream-fd=<pipe>  (layer-split, DFlash decode)
```

The **pipe protocol** is int32 little-endian token ids per committed decode token,
optional `-1` sentinel (compress/error acks; generate may omit sentinel on
layer-split path — Python reader must not block waiting for it).

**Linux pitfall:** `os.pipe()` fds are non-inheritable by default (PEP 446). The
spawn path must call `os.set_inheritable(w_pipe, True)` before `Popen(pass_fds=…)`
or the daemon writes to a stale fd and every stream hangs after the role chunk.
