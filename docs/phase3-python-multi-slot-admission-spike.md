# Phase 3 spike — Python multi-slot admission

**Status:** M3a green · M3b demux/cold/warm side-binary green · warm admit PASSED · **HTTP overlap smoke PASSED** (2026-07-14) · **EOS/SCHED one-shot fix** (early-stop clears remaining; skip SCHED when remaining=0) · compose **prod stays N=1** until post-fix smokes green  
**Branch:** `feat/native-mmproj-multi-request`  
**Parent:** [nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md)  
**Prereq:** Phase 2 M2b green ([phase2-layer-split-multi-slot-spike.md](./phase2-layer-split-multi-slot-spike.md))

---

## Verdict

Engine multi-slot (`--target-cache-slots=N`, `SLOT k`, `err slot_required`) is proven
on layer-split. Prod compose still runs **N=1** with exclusive `PriorityDaemonLock`.

Phase 3 makes Python speak that protocol and admit more than one in-flight request.
Ship as two slices so we never enable compose `N=2` without demux.

| Slice | Deliverable | Compose |
|-------|-------------|---------|
| **M3a** | Env/CLI plumb, `SLOT k` on snap/restore/generate, sticky slot lease pool | **N=1** |
| **M3b** | `--stream-tagged` demux + drop exclusive lock when N>1; chat∩cron | **N=2** after green |

---

## Regression (2026-07-14): N=2 short completions + phantom SCHED_DRAIN

**Symptom:** Under `SLOTS=2 TAGGED=1 DROP=1`, agent turns returned HTTP 200 with only
~8–16 completion tokens after long TTFB; daemon logged
`SCHED_DRAIN steps≈32760 in ~140ms` after a first quantum of `gen=8`.

**Cause:** Admit kept `remaining = max_tokens − emitted` after the first quantum even
when generation had already hit EOS (or finished early). `CONTINUE` re-seeded on the
EOS last-token and SCHED burned the leftover budget one phantom step at a time.
Demux already stopped on EOS → short answers. HTTP `finally` also awaited the long
SCHED kick.

**Fix (one-shot):**
1. **C++** — After each quantum, if `produced < requested` or last token is EOS (or
   `continue_generate` sees an EOS seed), set `remaining=0` and emit DONE.
2. **Python** — Parse `remaining=` from `ok RESTORE_CHAIN_ADMIT`; **skip SCHED_DRAIN**
   when `remaining<=0`; do not hold the request lock for a long SCHED await.

**Gate:** `phase3_warm_restore_admit_smoke.py` includes
`gate_large_max_tokens_no_phantom_drain` (`total_gen=64000`). Keep prod on **N=1**
until that + HTTP overlap smokes pass on a temporary N=2 recreate.

---

## M3a — first shippable gate

**Status: PASSED** (`c601431`, unit smoke `test_target_cache_admission`)

1. Plumb `DFLASH_TARGET_CACHE_SLOTS` / `DFLASH_STREAM_TAGGED` (defaults **1** / off)
2. Emit `SLOT k` when N>1
3. Sticky `TargetCacheSlotPool`
4. Admission: N=1 exclusive; N>1 lease + keep exclusive unless drop flag
5. Unit smoke green
6. **Do not** set compose `N=2` until M3b demux is green

---

## M3b — overlap

| Step | Detail | Status |
|------|--------|--------|
| Demux module | `tagged_stream_demux.py` | done |
| Cold overlap smoke | `phase3_multi_slot_overlap_smoke.py` START+SCHED_DRAIN | PASSED |
| Warm admit (C++) | `RESTORE_CHAIN … total quantum` → `ok RESTORE_CHAIN_ADMIT` + CONTINUE; SCHED_* completes | **PASSED** (`phase3_warm_restore_admit_smoke.py`, 24 tagged toks, req 1+2) |
| HTTP wire | Prefix `REQ <id>` + demux collect for chat | chat path (default tagged off) |
| HTTP SCHED driver | After warm admit, emit `SCHED_DRAIN`; quantum must survive `SLOT`/`REQ` prefixes | **PASSED** (`phase3_http_overlap_smoke.py`) |
| Drop exclusive + deploy N=2 | After HTTP overlap smoke green | **ready** |

**Protocol (warm):**
```text
REQ <id> SLOT <k> RESTORE_CHAIN <thick> <thin> <prompt> <total_gen> <quantum>
→ ok … (RESTORE_CHAIN …)   # classic ack after first quantum
→ ok RESTORE_CHAIN_ADMIT req=… slot=… emitted=… remaining=…
→ SCHED_DRAIN / SCHED_STEP / CONTINUE
```
Omitting `<quantum>` keeps legacy blocking full generate (prod N=1).

---

## Blockers / notes

| Finding | Impact |
|---------|--------|
| Python historically emitted bare `RESTORE_CHAIN` | Flipping daemon N>1 without SLOT → `err slot_required` (fixed M3a) |
| `DFLASH_LEGACY_DAEMON=1` | Affects **single-GPU** only; layer-split uses `daemon_loop` |
| Exclusive lock + dual lease | Capacity alone does not multiplex tokens; demux + START/SCHED is M3b |
| Tool thin pins | Process-global; shared across live slots (M2b certified) |
| `append_restore_chain_quantum` + `SLOT` prefix | Prefixed lines skipped quantum → blocking restore + hung ADMIT wait (fixed) |
| Tagged demux + warmup dual-read | Warmup `iter_pipe_tokens` raced demux on `r_pipe` (fixed) |
| Demux `stop_ids` `continue` | EOS skipped → hang waiting for DONE (fixed: break + idle) |
| ContextVar reset in SSE teardown | Slot lease leak → 503 slot wait (fixed) |
| EOS first quantum + huge `max_tokens` | `remaining` kept alive → phantom `SCHED_DRAIN` ~32k steps; demux already stopped → 8–16 tok answers (fixed: early-stop clears remaining; skip SCHED when remaining=0) |
| HTTP returns after first quantum under N=2 | Live (`b9d8c6a0-…`): `gen=8` then `completion_tokens≈10–16` while `SCHED_DRAIN` may continue after inline-snap — repro `scripts/phase3_http_quantum_truncation_smoke.py` (`GATE_PHASE3_HTTP_QUANTUM_TRUNC=1`) |
| Background `max_tokens=64000` storm | Hindsight/extractors (ephemeral) fill both slots → scoped chat 503 — repro `scripts/phase3_http_background_storm_smoke.py` (`GATE_PHASE3_HTTP_BG_STORM=1`) |

---

## Order of work (model-runner-v4)

1. Spike doc + parent plan pointer
2. `target_cache_admission.py`
3. Plumb env/CLI/entrypoint/compose (default N=1)
4. Wire lease + SLOT into `server_tools` / `daemon_bridge` / orchestrator
5. Unit smoke `test_target_cache_admission.py`
6. `tagged_stream_demux.py` + `phase3_multi_slot_overlap_smoke.py`
7. HTTP demux + drop exclusive + warm-overlap path
8. `phase3_http_overlap_smoke.py` behind `GATE_PHASE3_HTTP_OVERLAP=1` — **PASSED**
9. Deploy compose `N=2` + tagged + drop-exclusive; manual chat∩cron check
