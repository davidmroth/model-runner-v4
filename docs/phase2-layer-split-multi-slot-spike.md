# Phase 2 spike — Layer-split multi-slot

**Status:** M2a smoke PASSED on ai.local (2026-07-14) · M2b next  
**Branch:** `feat/native-mmproj-multi-request`  
**Parent:** [nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md)

---

## Verdict

Phase 1 protocol/scheduler (`daemon_loop`, `daemon_scheduler.h`) is reusable.
Prod path (`LayerSplitBackend` + `Qwen35LayerSplitAdapter`) now allocates **N
live `TargetCache`s per GPU shard** when `--target-cache-slots=N` (fail init on
OOM; **refuse N>1 with kvflash** until pager reattach exists).

---

## M2a — first shippable gate (ai.local, side binary)

1. ~~Plumb `--target-cache-slots` / `--stream-tagged`~~ done.
2. ~~Allocate N partial live caches per shard; fail init on OOM~~ done.
3. ~~`LayerSplitBackend` overrides: slot count / activate / busy / `continue_generate`~~ done.
4. ~~Smoke: `scripts/phase2_layer_split_multi_request_smoke.py`~~ passed
   (`--target-gpus=0,1`, N=2 tagged START+SCHED_DRAIN, `slot_busy`; AR-only, no kvflash).

**M2b (exit gate):** tool pin + `RESTORE_CHAIN` on slot 0 while slot 1 idle
(and inverse); N=1 certify still green. **Do not** set compose `N=2` until M2b.

---

## Order of work (lucebox-hub)

1. ~~Flag plumb~~
2. ~~Allocate N live caches — `Qwen35LayerSplitAdapter::allocate_extra_live_slots`~~
3. ~~Activate/swap per shard — `activate_target_cache_slot` / `swap_live_slot_state`~~
4. ~~`LayerSplitBackend` slot façade + `continue_generate`~~
5. RESTORE_CHAIN — `SLOT k` then restore into live slot `k` (M2b harden)
6. Smoke — `model-runner-v4/scripts/phase2_layer_split_multi_request_smoke.py`

---

## Risks (short)

| Risk | Mitigation |
|------|------------|
| VRAM ×2 live KV per GPU | Tiny `max_ctx` for spike; OOM fails at init |
| Wrong active slot on restore | Real `target_cache_slot_busy`; explicit `SLOT` before RESTORE_CHAIN |
| Shared thin tool pins | Keep tool restore serialized onto the target live slot |
| kvflash / feature ring | Multi-slot **refuses kvflash**; feature ring parked per slot when dflash |

Phase 3 (Python admission / drop exclusive lock) stays blocked until M2b.
