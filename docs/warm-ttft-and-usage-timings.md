# Warm TTFT deepen + usage.timings (WebUI metrics)

**Status:** Deployed on `ai.local` (2026-07-14) · `feat/native-mmproj-multi-request`  
**Repos:** `model-runner-v4` (Python patch), `lucebox-hub` (RESTORE_CHAIN ack fields)

Related: [inference-engine-north-star.md](./inference-engine-north-star.md),
[anonymous-feedback-loop-sop.md](./anonymous-feedback-loop-sop.md),
[nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md).

---

## 1. WebUI “no TTFT / t/s” under assistant messages

### Symptom

Chat shows only copy/retry under assistant bubbles — no READING / TTFT / t/s strip —
even though the engine is healthy.

### Cause

`MessagePane` only renders stats when **decode** timings are present
(`predicted_ms` / equivalent). Prefill-only payloads are hidden on purpose.

Dual-GPU layer-split logs decode as:

```text
[target-split-dflash] decode tokens=35 time=4.225 s speed=8.28 tok/s
```

`DaemonStdoutBus` used to match only `decode tokens=(\d+)` and **dropped**
`time=` / `speed=`. Result: `usage.timings` often had `prefill_ms` but never
`decode_ms` → proxy could build prompt-side timings → **UI showed nothing**.

Single-GPU `[dflash] generated N tokens in …` was fine; prod is layer-split.

### Fix

In `lucebox-patch/dflash/scripts/prefix_cache.py`:

- Parse full `[target-split-dflash] decode tokens=N time=S speed=T` into
  `decode_ms` + `decode_tokens_per_sec`.
- Also scrape `prefill_s` / `decode_s` / `decode_tok_s` from daemon `ok …`
  lines (including RESTORE_CHAIN).

Lucebox adapter (`ai-platform` `engine_adapters/lucebox.py`) already maps those
to `prompt_ms` / `predicted_ms` / `predicted_per_second` / `ttft_ms`.

### Verify

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-Id: metrics-check-$(date +%s)" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16,"stream":false}' \
  | python3 -c 'import sys,json; t=(json.load(sys.stdin).get("usage") or {}).get("timings") or {};
print(t); assert t.get("prompt_ms") and (t.get("predicted_ms") or t.get("predicted_per_second"))'
```

Expect both prefill and decode fields. WebUI then shows the stats strip again.

---

## 2. Warm TTFT ~30s despite `RESTORE_CHAIN thick=0` (~80% cached)

### Symptom

Log: `RESTORE_CHAIN thick=0 thin=[…]`, UI `~20K cached`, but TTFT ≈ 30s at ~170 t/s
on the uncached remainder. Gate elsewhere expects warm `prefill_s <= 5`.

### Measurement (instrumented daemon)

RESTORE_CHAIN now reports `restore_s`, `prefill_s`, `suffix_n` on the `ok` line
and `[restore-chain] …` on stderr. Finding on `ai.local`:

| Component | Wall time |
|-----------|-----------|
| Thick+thin H2D (`restore_s`) | ~0.05–0.5 s |
| Suffix prefill (`prefill_s`) | Dominates (~ suffix_n / 170–250 tok/s) |

Phase 1 multi-request scaffolding did **not** change restore semantics. The
30s path was real suffix FLOPs (and/or re-prefill after a shallow thick snap),
not “broken restore.”

### Cause (shallow inline cut)

`prepare_inline_snap` default cut is `candidates[-2]` (start of the current
user role). A large tool dump / pasted body in that turn never entered the
thick snapshot. Next turn looked up miss or shallow hit → `thick=-1` or huge
`suffix_n` again.

Post-gen `SNAPSHOT` of `prompt + raw gen tokens` **does not** hash-match the
next turn’s chat-templated assistant message — do not use that deepen path.

### Fix

When `len(prompt_ids) - candidates[-2] >= DFLASH_PREFIX_DEEPEN_TAIL` (default
**256**), snap at **`candidates[-1]`** — the trailing generation-prompt
`<|im_start|>assistant` boundary. That cut:

- Includes the large completed user body.
- Is stable with the next turn’s templated history (shared role opener).

Also: `lookup()` considers `_slot_prefix_len` cuts that fall between
role-opening boundaries so deepened snaps are findable.

### Probe (ai.local)

| Turn | Behavior | TTFT (usage) |
|------|----------|--------------|
| Large dump (pay once) | ~17k suffix | ~74 s |
| Short follow-up **before** fix | Re-prefill ~18k | ~79 s |
| Short follow-up **after** | `thick=0`, `suffix_n≈24` | **~1.2 s** |

The *first* turn that introduces a large uncached body still pays once; later
turns must not.

### Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `DFLASH_PREFIX_DEEPEN_TAIL` | `256` | Min tokens past role cut before deepen to last role-start; `0` disables |

---

## 3. Code map

| Area | Files |
|------|--------|
| Decode / ok parsing | `lucebox-patch/dflash/scripts/prefix_cache.py` (`DaemonStdoutBus`) |
| Deepen policy + lookup | `prefix_cache.py` (`prepare_inline_snap`, `lookup`) |
| Tests | `lucebox-patch/dflash/scripts/test_prefix_cache_slot_depth.py` |
| RESTORE_CHAIN ack | `lucebox-hub` `daemon_loop.cpp`, `layer_split_backend.cpp`, `model_backend.h` |
| UI gate | `webui` `MessagePane.svelte` (`generatedMs` required) |
| Proxy map | `ai-platform` `engine_adapters/lucebox.py` |

---

## 4. Ops checklist

After patch or daemon rebuild:

1. Recreate `model-runner-v4-lucebox` (Python is bind-mounted; still restart to reload).
2. Curl check §1 (decode + prefill present).
3. Two chat turns with `X-Conversation-Id` and tools: turn-2+ `RESTORE_CHAIN thick≥0`;
   large-then-short: second turn `suffix_n` small and `prefill_s` (or `prompt_ms`) ≤ 5 s
   when history was deepened.
4. Confirm WebUI shows READING / TTFT / t/s under a fresh assistant reply.
