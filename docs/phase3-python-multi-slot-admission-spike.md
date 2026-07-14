# Phase 3 spike — Python multi-slot admission

**Status:** M3a in progress · compose stays **N=1**  
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

1. Plumb `DFLASH_TARGET_CACHE_SLOTS` / `DFLASH_STREAM_TAGGED` (defaults **1** / off)
   through compose → `entrypoint-tool-split-serve.sh` → `server_tools` `extra_daemon`.
2. Emit `SLOT k` on `RESTORE` / `RESTORE_CHAIN` / `SNAPSHOT*` / generate when N>1
   (`format_slot_command` + ContextVar from the lease).
3. Sticky `TargetCacheSlotPool`: conversation / cache_scope → live slot; free-list + wait.
4. Admission:
   - **N=1:** exclusive `PriorityDaemonLock` (unchanged).
   - **N>1:** acquire a slot lease for capacity; keep exclusive pipe lock unless
     `DFLASH_MULTI_SLOT_DROP_EXCLUSIVE=1` (smoke / M3b only — garbles streams
     without demux).
5. Unit smoke: SLOT formatting, dual lease admit/release, N=1 exclusive path intact.
6. **Do not** set compose `N=2` until M3b demux is green.

### M3a exit gate

- `python -m unittest test_target_cache_admission` green (Docker).
- With `DFLASH_TARGET_CACHE_SLOTS=1` (prod default): command lines have **no** `SLOT`
  prefix; daemon still gets no `--target-cache-slots` (or `=1`).
- With slots=2 in unit tests: `RESTORE_CHAIN` / `SNAPSHOT_THIN` lines are
  `SLOT k …`; pool admits two concurrent leases and waits/times out on a third.
- Prod recreate with defaults still healthy (N=1).

---

## M3b — overlap (next)

- Demux tagged frames `[-2, req_id, tok]` → per-HTTP SSE.
- Speak `REQ` / `SCHED_*` (or equivalent) so two generates share one stdin safely.
- Set `DFLASH_MULTI_SLOT_DROP_EXCLUSIVE=1` (or remove the gate) once demux works.
- Deploy `DFLASH_TARGET_CACHE_SLOTS=2` + `DFLASH_STREAM_TAGGED=1`.
- Manual: chat stream + cron completion overlapping.

---

## Blockers / notes

| Finding | Impact |
|---------|--------|
| Python historically emitted bare `RESTORE_CHAIN` | Flipping daemon N>1 without SLOT → `err slot_required` |
| `DFLASH_LEGACY_DAEMON=1` | Affects **single-GPU** path only; layer-split (`--target-gpus`) uses `daemon_loop` where multi-slot lives |
| Exclusive lock + dual lease | Capacity accounting alone does not multiplex tokens; demux is M3b |
| Tool thin pins | Process-global; shared across live slots (M2b certified) |

---

## Order of work (model-runner-v4)

1. Spike doc (this file) + parent plan pointer
2. `target_cache_admission.py` — config helpers, `format_slot_command`, `TargetCacheSlotPool`
3. Plumb env/CLI/entrypoint/compose (default N=1)
4. Wire lease + SLOT into `server_tools` / `daemon_bridge` / orchestrator
5. Unit smoke `test_target_cache_admission.py`
6. M3b (demux) — separate gate
