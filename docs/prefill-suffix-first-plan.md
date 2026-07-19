# Prefill Improvement Plan — Suffix-First (Stay on Lucebox)

**Status:** Phase 0 done · Phase A in progress (slot thrash fix deployed)  
**Date:** 2026-07-17  
**Scope:** `model-runner-v4` / lucebox dflash on `ai.local` (Qwen3.6-27B Q4_K_M, 2×3090 layer-split)  
**Repos:** engine Python patch + daemon logs; no Hermes overlays; no stack switch

Related: [warm-ttft-and-usage-timings.md](./warm-ttft-and-usage-timings.md),
[inference-engine-north-star.md](./inference-engine-north-star.md),
[restore-chain-phases-1-3.md](./restore-chain-phases-1-3.md),
[agent-inference-program.md](./agent-inference-program.md),
[feedback-pflash-agent-regression.md](./feedback-pflash-agent-regression.md).

---

## 1. Goal (plain English)

Make warm agent turns feel fast by **prefilling less leftover context** each
turn — not by chasing blog-tier prompt tok/s on a different engine.

Cold / large-suffix GPU rate stays a **secondary** lever (~tens of percent at
best on this path).

---

## 2. Non-goals

| Out of scope | Why |
|---|---|
| `VLLM_MARLIN_INPUT_DTYPE=int8` / Marlin W4A8 | vLLM-only; no-op on GGUF/lucebox |
| Switching to vLLM / Ollama / stock llama-server | Explicitly keep lucebox |
| Matching “1.2k–2.2k tok/s for 27B on 3090” blog tables | Wrong benchmark class vs layer-split agent path |
| Enabling `DFLASH_PREFILL_MODE` without `agent_safe` canary | Known agent/cron quality regression |
| Hermes-side concurrency hacks for prefill | Prefer engine cache correctness |

Already true on Ampere Q4: ggml **MMQ** uses int8 tensor cores. There is no
missing “flip int8 on” switch for this stack.

---

## 3. Baseline (ai.local, 48h ending 2026-07-17)

From `docker logs model-runner-v4-lucebox` `[restore-chain]` lines (**n=512**):

| Metric | p50 | p90 | Notes |
|---|---:|---:|---|
| `suffix_n` (tokens still prefilled) | **583** | **3,570** | max ~49k |
| `prefill_s` | **~4.4 s** | **~21 s** | max ~276 s |
| Implied rate `suffix_n/prefill_s` | ~130 tok/s | ~184 tok/s | Short suffixes look slower (overhead) |
| `thick=-1` rate | **7.8%** | — | Full conversation snap miss |

Large-suffix cold-ish rate (when `suffix_n` is big) is still ~**170–250 tok/s**
(~200 typical), consistent with prior docs.

**Interpretation:** usual warm turn still pays ~600 leftover tokens (~4 s). One
in ten pays ~3.5k+ (~20 s). Cache/suffix tax dominates; raw rate is secondary.

---

## 4. Strategy

```text
TTFT ≈ restore_s + suffix_n / ~200 tok/s
```

- Cutting `suffix_n` 10× beats raising tok/s 20%.
- Phase A = make restore deep and reliable.
- Phase B = optional ubatch / layer-split A/B for unavoidable large suffixes.
- Phase C = only if A+B plateau and product still needs more (separate decision).

---

## 5. Target bands

Same hardware, stay on lucebox:

| Metric | Today | Near-term (Phase A healthy) | Stretch (A working as designed) |
|---|---:|---:|---:|
| p50 `suffix_n` | ~580 | **100–200** | **30–80** |
| p90 `suffix_n` | ~3,500 | **500–1,000** | **200–400** |
| p50 `prefill_s` | ~4.4 s | **1–2 s** | **0.5–1 s** |
| p90 `prefill_s` | ~21 s | **5–8 s** | **2–4 s** |
| `thick=-1` rate | ~8% | **&lt;2%** | **&lt;1%** |

First-time huge blobs (new tool pin, first large paste) still pay once. Success
means **follow-ups do not pay again**.

Phase B alone might move large-suffix rate ~200 → ~220–280 tok/s (~0–30%). That
is not enough to hit near-term `prefill_s` bands without Phase A.

---

## 6. Phases

### Phase 0 — Scorecard (measure before changing)

**Status:** Done (2026-07-17) — `scripts/prefill_suffix_scorecard.py`

**Deliverable:** parse recent lucebox logs into:

- n, p50/p90/p95/max for `suffix_n`, `prefill_s`, `tok/s`
- `% thick=-1`
- split: follow-ups with `suffix_n < 8k` vs `≥ 8k`
- optional `--check-gates` vs Phase A near-term exit criteria

**Verify:** 48h run on `ai.local` (2026-07-17) matched §3 baseline (n=512,
p50 `suffix_n`=583, p90=3570, `thick=-1`=7.8%). Gates correctly **FAIL**.

```bash
# On ai.local (or SSH + pipe):
docker logs model-runner-v4-lucebox --since 48h 2>&1 \
  | python3 scripts/prefill_suffix_scorecard.py --check-gates

# JSON artifact:
docker logs model-runner-v4-lucebox --since 48h 2>&1 \
  | python3 scripts/prefill_suffix_scorecard.py --json /tmp/prefill-score.json
```

**Gate to Phase A:** met — script checked in; prod baseline re-confirmed.

### Phase A — Suffix tax (primary)

Ordered work:

| Step | Work | Success signal |
|---|---|---|
| A1 | Confirm **prefix deepen** live (`DFLASH_PREFIX_DEEPEN_TAIL=256`) after large user/tool bodies | Large-then-short: turn-2 `suffix_n` tens–low hundreds, not ~equal to dump size |
| A2 | Confirm **tool-split pins** healthy on agent toolsets | Turn-2+ `RESTORE_CHAIN` with thin hit; no systematic re-pin every turn |
| A3 | Drive down **`thick=-1`** | Rate &lt;2% over 48h; grep `deferred conv snap failed` / shallow cuts |
| A3a | **(2026-07-17)** Runtime-reclaim unused PFlash slots into prefix while mode=off; keep `PREFILL_CACHE_SLOTS≥2` in config for future enable | Startup: reclaim log + `slot budget` with higher prefix; `.env` still has prefill≥2 |
| A4 | Multi-slot / scope hygiene | No cross-session pin thrash; `[corr]` logs used when debugging |
| A5 | Optional: agent prompt hygiene (huge tool results) | Fewer ≥8k first-pay suffixes — product/ops, not kernel |

**Exit criteria (48h window, n≥200 restore-chain samples):**

- p50 `suffix_n` ≤ 200  
- p90 `suffix_n` ≤ 1,000  
- p50 `prefill_s` ≤ 2.0 s  
- p90 `prefill_s` ≤ 8.0 s  
- `thick=-1` &lt; 2%

If deepen probe-class behavior is consistent (suffix ≈ tens of tokens on short
follow-ups), stretch bands become the next bar — do not block Phase B on stretch.

### Phase B — Rate tax (secondary, after A exit or clear plateau)

Fixed-prompt A/B only (same cold 8k and 16k prompts, empty competing load):

| Experiment | Change | Measure |
|---|---|---|
| B1 | Ubatch / batch if daemon exposes knobs | `prefill_s` / tok/s on fixed prompt |
| B2 | Layer split balance (`32,32` → `30,34` / `33,31`) | Same |
| B3 | Confirm no `FORCE_CUBLAS`; leave `GGML_CUDA_NO_GRAPHS=1` | Decode-oriented; do not “fix prefill” by toggling blindly |

**Exit criteria:** documented A/B table; adopt only if ≥10% large-suffix tok/s
gain with no quality / OOM / decode regression on cert smoke.

**Do not** expect Phase B to deliver §5 near-term bands alone.

### Phase C — Explicit non-default options (decision required)

Only if product still needs more after A+B:

- `DFLASH_PREFILL_MODE=agent_safe` canary (see existing agent-safe PFlash plan)
- Topology experiment: short-ctx single-GPU pp rate vs long-ctx layer-split
  (diagnostic, not a silent prod flip)
- Stack change (vLLM Marlin W4A8) — **out of this plan** unless leadership
  reopens “keep lucebox”

---

## 7. How to read success (p50 / p90)

- **p50** = typical turn (half of turns are this good or better).  
- **p90** = annoying-but-common turn (nine of ten are this good or better).

Optimize both. A great demo turn with `suffix_n=24` does not count as done if
p90 is still 3,500.

---

## 8. Ops checklist

After any Phase A patch or compose change:

1. Recreate `model-runner-v4-lucebox` if needed (Python bind-mount still restart).
2. Smoke: two tool turns with stable `X-Conversation-Id` → turn-2+
   `RESTORE_CHAIN thick≥0`, thin hit.
3. Large-then-short probe → turn-2 `suffix_n` small (`prefill_s` ideally ≤ 5 s).
4. Re-run scorecard on ≥24h traffic (or synthetic soak if quiet):

   ```bash
   docker logs model-runner-v4-lucebox --since 48h 2>&1 \
     | python3 scripts/prefill_suffix_scorecard.py --check-gates
   ```

5. Keep L0 tooling gate green (queue / keepalives / no 503 flood) — orthogonal
   but must not regress.

---

## 9. Work log

| Date | Note |
|---|---|
| 2026-07-17 | Plan written. Baseline §3 from 48h `[restore-chain]` logs. Deepen documented as deployed (2026-07-14) in warm-ttft doc; prod p50/p90 show remaining suffix tax. |
| 2026-07-17 | Phase 0: `scripts/prefill_suffix_scorecard.py` landed; 48h ai.local re-run matches baseline; `--check-gates` FAIL as expected. |
| 2026-07-17 | **Phase A triage** (`scripts/prefill_suffix_triage.py`): of 48 lookup misses, **30 cold first-seen**, **18 miss-after-prior-commit** (LRU thrash). Top thrash scope = long Qwen chat (9×). Cron unique scopes also collide. `thick≥0` suffix buckets show **no 0–99 band** — agent turns typically append 100–1200+ tokens even on hit. |
| 2026-07-17 | **Phase A fix:** with PFlash off, unused `prefill_cache_slots` were still reserved in the 8-slot budget while prod ran `PREFIX=2`+`PREFILL=2`. Added `reclaim_prefill_slots_when_pflash_off()` in `server_tools.py` (**runtime fold only** — keep `PREFILL_CACHE_SLOTS≥2` in config for a future PFlash enable). Prod `.env` `PREFIX=4` + `PREFILL=2` → runtime `prefix=6 prefill=0` while mode=off. Re-scorecard after ≥24h traffic. |
