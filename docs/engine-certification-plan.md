# Engine Certification & Optimization Plan

**Program overview:** [agent-inference-program.md](./agent-inference-program.md) — what
we're trying to achieve, architecture, and sign-off criteria.

**North star (metrics + no-hack policy):** [inference-engine-north-star.md](./inference-engine-north-star.md)

**Operations SOP:** [anonymous-feedback-loop-sop.md](./anonymous-feedback-loop-sop.md)

**Iteration template:** [anonymous-feedback-loop-scorecard.md](./anonymous-feedback-loop-scorecard.md)

**Goal:** Restore Hermes agent tool-calling stability and hit the speed targets
documented in [whitepaper-agent-inference-cache.md](./whitepaper-agent-inference-cache.md)
on 2×RTX 3090 (`ai.local` and production Lightsail).

**Scope:** Inference engine only (`lucebox-hub` + `model-runner-v4`). Hermes,
WebUI, and ai-platform proxy are consumers — they must forward
`X-Conversation-Id` but do not own KV cache behavior.

---

## Problem statement (July 2026)

Production WebUI session `deea23ff-5d7d-4a4b-8504-2d36a423b34e` showed:

| Symptom | Engine cause |
|---------|----------------|
| `Empty response from model — retrying` (0 completion tokens) | Stale prefix-cache KV restore → `empty prompt` in daemon log |
| Raw `<function=web>…` in chat instead of structured `tool_calls` | `parse_tool_calls()` missed bare `<function=` blocks (no `<tool_call>` wrapper) |
| Intermittent success (`terminal` tool worked; web search failed first turn) | Cache pollution + parser gap, not Hermes regression |

Root causes are documented in:

- [feedback-prefix-cache-regression.md](./feedback-prefix-cache-regression.md) — global LRU slot reuse
- [feedback-pflash-agent-regression.md](./feedback-pflash-agent-regression.md) — PFlash conv compression

---

## Speed targets (whitepaper claims)

| Metric | Cold / naive | Target (warm tool-split) | Gate |
|--------|--------------|--------------------------|------|
| Prefill @ ~8.5K tokens | ~11 s | **≤ 0.5 s** (~20×) | `agent_after_tool_prefill_ms < 500` or `< 70%` of cold |
| Decode throughput | 7–8 tok/s | **51–94 tok/s** (context-dependent) | Logged in bench samples |
| Wall time after tool result | 30–45 s/turn | **< 6 s** | `agent_after_tool_s < 6.0` |
| Multi-turn prefill (turn 3 vs 1) | — | **≥ 5×** speedup | `incremental_prefill_speedup` check |

**Required stack configuration:**

```bash
DFLASH_TOOL_SPLIT_ENABLED=1
DFLASH_PREFILL_MODE=off
DFLASH_TOOL_SPLIT_COMPRESS_CONV=0
DFLASH_LAYER_SPLIT=1
DFLASH_DRAFT_FEATURE_MIRROR=1   # eliminates ~700ms/step cross-GPU copy
```

Agent traffic should include `X-Conversation-Id: <session-uuid>` so scoped
prefix cache can reuse conversation snapshots without cross-session pollution.

---

## Stability targets

| Check | Pass criteria |
|-------|---------------|
| No cache pollution | Benchmark probe → agent tool call returns `completion_tokens > 0`, no `empty prompt` |
| Cross-session isolation | Unrelated session after agent turn: no 503, no HumanEval benchmark text |
| Tool parse | Bare `<function=name>` and `<tool_call>` wrappers both → `tool_calls[]` |
| Daemon health | `/health` returns `{"status":"ok"}` (daemon process alive) |
| Snapshot protocol | Logs contain `RESTORE_CHAIN` and `inline-snap committed`; zero `inline snap failed` |
| Zero-token guard | No sustained `completion_tokens=0` on agent prompts with tools |

---

## Certification pipeline

See also: [restore-chain-phases-1-3.md](./restore-chain-phases-1-3.md) (RESTORE_CHAIN recovery plan).

Run on any host with `model-runner-v4-lucebox` + `ai-platform-proxy` on
network `ai-inference`:

```bash
cd /media/data/projects/model-runner-v4
PROXY_URL=http://ai-platform-proxy:8000 \
  bash scripts/run-engine-certification.sh
```

### Phase 1 — Unit tests (seconds)

| Test file | What it validates |
|-----------|-------------------|
| `test_prefix_cache_slot_depth.py` | `resolve_cache_scope()`, scoped LRU, stale-slot eviction |
| `test_parse_tool_calls.py` | Bare `<function=` blocks (Flock regression), wrapped `<tool_call>` |

### Phase 2 — Cache pollution E2E (~2 min)

`scripts/test_cache_pollution.py`:

1. Send HumanEval-style benchmark prompt (pollutes global cache in old builds)
2. Send agent tool-call turn with `X-Conversation-Id`
3. Assert non-empty completion, structured tool call or text

### Phase 3 — Thorough tool-split bench (~15–30 min)

`scripts/benchmark-tool-split-thorough.py` — 10 automated checks:

- `agent_cold_ok`, `agent_hot_fast`, `agent_prefill_improved`
- `incremental_prefill_speedup`, `turn5_wall_faster`
- `cross_session_no_503`, `cross_session_after_tool_ok`
- `restore_chain_seen`, `inline_snap_seen`, `no_inline_snap_failed`

Artifacts land in `/tmp/engine-cert-YYYYMMDD_HHMMSS/`.

---

## Deploy checklist (tool-split path)

### Patch mount — full file set

Copy to `model-runner-v4/lucebox-patch/dflash/scripts/`:

```
_prefill_hook.py          # required import for server_tools prefill path
prefix_cache.py           # session-scoped cache keys
server_tools.py           # OpenAI API + parse_tool_calls fix
tool_split/               # qwen3 adapter plugins
test_prefix_cache_slot_depth.py
test_parse_tool_calls.py
```

### Entrypoint

`scripts/entrypoint-tool-split-serve.sh` must:

1. Set `LD_LIBRARY_PATH` for:
   - `dflash-build/deps/llama.cpp/tools/mtmd` (`libmtmd.so.0`)
   - `dflash-build/bin`
   - `dflash-build/deps/llama.cpp/ggml/src` (+ `ggml-cuda`)

2. Pass **integer** GPU ids to `--target-gpus` (`0,1`), not `cuda:0,cuda:1`
   (test_dflash rejects the latter with `bad --target-gpus value`).

Without mtmd on the loader path the daemon exits immediately → `/health` 503.
With wrong GPU syntax the daemon dies on the first inference request.

### Enable tool-split

```bash
# model-runner-v4/.env
DFLASH_TOOL_SPLIT_ENABLED=1
```

```bash
docker compose --profile serve up -d --force-recreate lucebox
# Wait ~30–120s for model load; verify:
curl -sf http://127.0.0.1:8080/health
docker logs model-runner-v4-lucebox 2>&1 | grep -E 'tool-split = on|libmtmd'
```

### Proxy header forwarding

`ai-platform/services/proxy/alias_proxy.py` forwards `X-Conversation-Id`.
WebUI / Hermes gateway should set this on every chat completion request.

---

## Optimization roadmap (priority order)

### P0 — Stability (done / in flight)

1. **Session-scoped prefix cache** — `resolve_cache_scope()` in `prefix_cache.py`
2. **Bare function parser** — `_parse_function_block()` in `server_tools.py`
3. **Complete patch deploy** — all files above + `_prefill_hook.py`
4. **libmtmd loader fix** — `entrypoint-tool-split-serve.sh` LD_LIBRARY_PATH

### P1 — Speed validation

5. Run certification suite; archive JSON in `model-runner-v4/bench-results/`
6. Tune `DFLASH_PREFIX_CACHE_SLOTS` (default 4) if cross-toolset thrash seen
7. Confirm `DFLASH_DRAFT_FEATURE_MIRROR=1` in production `.env`

### P2 — Production parity

8. Deploy same patch + env to Lightsail inference stack
9. Re-run Flock camera WebUI E2E (`What is Flock camera?` → `web` tool)
10. Monitor lucebox logs for 24h: `empty prompt`, `inline snap failed`

### P3 — Hardening (optional)

11. When `parse_tool_calls` sees partial/corrupt tool syntax after cache miss,
    return HTTP 502 + retry hint instead of leaking raw markup to Hermes
12. Add ai-platform proxy metric: `completion_tokens=0` rate per model
13. CI gate: run unit tests + pollution test on PRs touching `server_tools.py`
    or `prefix_cache.py`

---

## Rollback

If certification fails or agent traffic degrades:

```bash
DFLASH_TOOL_SPLIT_ENABLED=0   # falls back to dflash_server HTTP path
docker compose --profile serve up -d --force-recreate lucebox
```

This disables RESTORE_CHAIN speedups but restores the pre-regression stable path
that ran for months before tool-split was enabled without the scoped-cache fix.

---

## Success criteria (sign-off)

- [ ] `run-engine-certification.sh` exits 0 (all bench checks pass)
- [ ] Unit tests: 12/12 pass (9 prefix_cache + 3 parse_tool_calls)
- [ ] Pollution test: benchmark → agent turn succeeds
- [ ] `agent_after_tool_s < 6` and `agent_after_tool_prefill_ms < 500` on ai.local
- [ ] WebUI Flock camera prompt: structured `web` tool call, no empty retries
- [ ] No `empty prompt` in lucebox logs during 30-minute agent session

---

## References

- [whitepaper-agent-inference-cache.md](./whitepaper-agent-inference-cache.md) — architecture + benchmarks
- [feedback-prefix-cache-regression.md](./feedback-prefix-cache-regression.md)
- [deployment-flow.md](./deployment-flow.md) — git pull / compose lifecycle
- `scripts/run-engine-certification.sh` — one-command cert runner
