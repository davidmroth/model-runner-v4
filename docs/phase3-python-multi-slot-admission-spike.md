# Phase 3 spike — Python multi-slot admission

**Status:** M3a green (2026-07-14) · M3b demux smoke PASSED (2026-07-14) · HTTP wire next · compose stays **N=1**  
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
| Demux module | `tagged_stream_demux.py` — `TaggedFrameBuffer` + `TaggedStreamDemux` | done |
| Overlap smoke | `scripts/phase3_multi_slot_overlap_smoke.py` — dual `REQ`/`SLOT`/`START` + `SCHED_DRAIN` | **PASSED** (24 tagged tokens, req_ids 1+2) |
| HTTP wire | Prefix `REQ <id>` when tagged; demux → chat collect via `_generate_via_daemon`; stdin mutex; drop exclusive only with demux | chat path wired (default tagged off) |
| Drop exclusive | Only with demux + START/SCHED (blocking `RESTORE_CHAIN` cannot interleave) | after HTTP |
| Deploy | `DFLASH_TARGET_CACHE_SLOTS=2` + `DFLASH_STREAM_TAGGED=1` after smoke | blocked on HTTP |

**Blocker for warm chat∩cron:** daemon `START` cold-prefills; prod warm path is
`RESTORE_CHAIN`. Overlap smoke proves demux + scheduler; warm restore→quantum
may need a C++ follow-up.

---

## Blockers / notes

| Finding | Impact |
|---------|--------|
| Python historically emitted bare `RESTORE_CHAIN` | Flipping daemon N>1 without SLOT → `err slot_required` (fixed M3a) |
| `DFLASH_LEGACY_DAEMON=1` | Affects **single-GPU** only; layer-split uses `daemon_loop` |
| Exclusive lock + dual lease | Capacity alone does not multiplex tokens; demux + START/SCHED is M3b |
| Tool thin pins | Process-global; shared across live slots (M2b certified) |

---

## Order of work (model-runner-v4)

1. Spike doc + parent plan pointer
2. `target_cache_admission.py`
3. Plumb env/CLI/entrypoint/compose (default N=1)
4. Wire lease + SLOT into `server_tools` / `daemon_bridge` / orchestrator
5. Unit smoke `test_target_cache_admission.py`
6. `tagged_stream_demux.py` + `phase3_multi_slot_overlap_smoke.py`
7. HTTP demux + drop exclusive + warm-overlap path
