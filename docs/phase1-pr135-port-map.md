# Phase 0: PR #135 → our-tree port map

**Status:** Done (2026-07-13)  
**PR tip:** `refs/tmp/pr135` @ `561b0ac` (`javierpazo/lucebox-hub`)  
**Our tip:** `feat/native-mmproj-multi-request` @ `0cbf153`  
**Parent plan:** [nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md)

Do **not** `git merge` #135. Paths and ownership differ (`dflash/test/test_dflash.cpp` monolithic vs extracted `daemon_loop` + `ModelBackend`).

---

## Concept map

| PR #135 concept | Our tree target | Notes |
|-----------------|-----------------|-------|
| `--target-cache-slots=N` | CLI → `Qwen35DaemonArgs` / `Qwen35Config` → `Qwen35Backend::init` | Clamp 1–16; default **1** |
| Extra `TargetCache` alloc | `create_target_cache(w, …, slot.cache)` after primary `cache_` | Shared `TargetWeights`; KV/SSM only per slot |
| `DaemonSlotState` | New `server/src/common/daemon_slots.h` **or** private in `Qwen35Backend` | Holds cache + StepGraphs + feature_mirror + first_iter |
| `ActiveDaemonSlot` swap | Backend `activate_slot(id)` RAII | Swap into primary `cache_`/`sg_` working set |
| `DaemonRequestState` / `PendingQuantum` | `daemon_scheduler` state beside `run_daemon` | Transport-agnostic; pipe is one client |
| `REQ` / `SLOT` prefixes | Start of stdin parse in `daemon_loop.cpp` | Before existing command table |
| `START` / `CONTINUE` / `CANCEL` | New cmds in `daemon_loop` → backend admit/run | Epoch bump on cancel |
| `SCHED_STEP` / `SCHED_DRAIN` | New cmds; single-seq first | `SCHED_BATCH_*` **deferred** past Phase 1 gate |
| `--stream-tagged` | `DaemonIO::emit` + `DaemonLoopArgs.stream_tagged` | Frames `[-2, req_id, tok]`; `-4` CONTINUE; `-1` DONE |
| `LIST_TARGET_CACHE_SLOTS` | New cmd (≠ today’s `LIST_SLOTS`) | Live caches vs PrefixSnapshot slots |
| `RESTORE_CHAIN` vs busy slot | Refuse overwrite when slot has active req | Match #135 intent |
| Layer-split multi-slot | **Phase 2** — `LayerSplitBackend` / shards | Out of Phase 1 |
| Python demux / drop lock | **Phase 3** — `server_tools.py` | Out of Phase 1 |

---

## Do not confuse

| Today | Phase 1 target |
|-------|----------------|
| PrefixSnapshot slots (`LIST_SLOTS`, `kMaxSlots=64`) | Live **target-cache** slots (concurrent KV) |
| Layer-split per-GPU `TargetCache` (one request, sharded layers) | N concurrent requests |
| Blocking generate until EOS | Quantum-sized generate + scheduler |

---

## Phase 1 order of attack

1. CLI + config plumbing (`--target-cache-slots`, `--stream-tagged`)
2. Allocate extra caches in `Qwen35Backend::init` (single-GPU only)
3. `SLOT` / `LIST_TARGET_CACHE_SLOTS` + activate/swap
4. Tagged stream frames
5. `REQ` + `START` / `CONTINUE` / `CANCEL` + `SCHED_STEP` / `SCHED_DRAIN`
6. Gate tests on `ai.local` (side binary; compose stays `N=1`)

---

## Gate reminder (enter Phase 2)

1. `N=1` smoke green — **done** (generate + SNAPSHOT/`RESTORE`; vision /
   `RESTORE_CHAIN` deferred to Phase 2 layer-split)
2. `N=2` two short concurrent gens; demux OK; no crash — **done**
   (`scripts/phase1_multi_request_smoke.py` → `test_dflash.phase1`)
3. Scheduler API not hard-wired to “stdin is special”
4. Busy-slot RESTORE policy documented + tested — **done** (`err slot_busy`)
5. Production compose still layer-split `N=1` — **unchanged**
