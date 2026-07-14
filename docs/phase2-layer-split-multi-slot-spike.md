# Phase 2 spike — Layer-split multi-slot

**Status:** Survey done (2026-07-14) · Implementation next (M2a)  
**Branch:** `feat/native-mmproj-multi-request`  
**Parent:** [nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md)

---

## Verdict

Phase 1 protocol/scheduler (`daemon_loop`, `daemon_scheduler.h`) is reusable.
Prod path (`LayerSplitBackend` + `Qwen35LayerSplitAdapter`) still has **one live
`TargetCache` per GPU shard** and ignores `--target-cache-slots` /
`--stream-tagged`. Phase 2 is almost entirely **lucebox-hub**.

---

## M2a — first shippable gate (ai.local, side binary)

1. Plumb `--target-cache-slots` / `--stream-tagged` into `LayerSplitDaemonConfig`
   → adapter + `DaemonLoopArgs.stream_tagged`.
2. Allocate N partial live caches per shard; **fail init on OOM**.
3. `LayerSplitBackend` overrides: slot count / activate / busy / `continue_generate`.
4. Smoke: `--target-gpus=0,1 --target-cache-slots=2 --stream-tagged` (small
   `--max-ctx`); `LIST_TARGET_CACHE_SLOTS`; two `START` + `SCHED_DRAIN`; demux
   `req_ids=[1,2]`.

**M2b (exit gate):** tool pin + `RESTORE_CHAIN` on slot 0 while slot 1 idle
(and inverse); N=1 certify still green. **Do not** set compose `N=2` until M2b.

---

## Order of work (lucebox-hub)

1. Flag plumb — `test_dflash.cpp` → `LayerSplitDaemonConfig` →
   `layer_split_daemon_loop.cpp` + `BackendArgs` / adapter config
2. Allocate N live caches — `Qwen35LayerSplitAdapter::init` (mirror
   `Qwen35Backend` `extra_slots_`)
3. Activate/swap per shard — `activate_live_slot`; reattach draft/kvflash carefully
4. `LayerSplitBackend` slot façade + `continue_generate`
5. RESTORE_CHAIN — `SLOT k` then restore into live slot `k`; thin PrefixSnapshots
   stay process-global
6. Smoke — `model-runner-v4/scripts/phase2_layer_split_multi_request_smoke.py`

---

## Risks (short)

| Risk | Mitigation |
|------|------------|
| VRAM ×2 live KV per GPU | Tiny `max_ctx` for spike; OOM fails at init |
| Wrong active slot on restore | Real `target_cache_slot_busy`; explicit `SLOT` before RESTORE_CHAIN |
| Shared thin tool pins | Keep tool restore serialized onto the target live slot |
| kvflash / feature ring | Prefer AR / kvflash-off for first smoke |

Phase 3 (Python admission / drop exclusive lock) stays blocked until M2b.
