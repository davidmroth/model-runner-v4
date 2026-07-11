# Anonymous Feedback Loop SOP

Status: Approved  
Date: 2026-07-11  
Scope: Lucebox tool-split + prefix cache + RESTORE_CHAIN in `model-runner-v4`

## Goal
Use a repeatable anonymous feedback loop to validate, fix, and deploy reliability improvements without architecture rollbacks.

## Non-Negotiable Guardrails
- Keep tool-split enabled.
- Keep prefix cache enabled.
- Keep RESTORE_CHAIN enabled.
- Do not mask engine defects by only increasing client timeouts/retries.
- Deploy through git only: commit, push, pull on `ai.local`, then recreate `lucebox`.

## Mandatory Loop (Every Iteration)
1. Probe and validate using WebUI Lightsail instrumentation.
2. Patch/fix in `model-runner-v4` patch layer.
3. Test/validate in Docker context.
4. Commit with evidence-linked message.
5. Push to `origin/feat/vision`.
6. Pull on `ai.local`.
7. Redeploy `lucebox`.
8. Re-probe from WebUI and re-validate with `ai.local` logs.

## Quantifiable Goals (Release Gate)
- Deployment parity: `ai.local` deployed SHA equals latest approved `feat/vision` SHA.
- Cold path bound: after restart, turn-1 cold prefill is allowed once and should be <= 120s.
- Deferred snap efficiency: deferred conversation snap must use tail-only payload, not full prompt replay.
- Deferred tail bound: `tail_len <= DFLASH_DEFERRED_CONV_SNAP_MAX_TAIL` (default 8192).
- Warm path latency: turn-2 (within 30s) shows `lookup hit slot=0` and `RESTORE_CHAIN thick=0`, with `prefill_s <= 5`.
- Lock contention: zero occurrences of scoped `daemon_lock wait timed out after 60s` during validation windows.
- User-visible reliability: zero stuck/empty turns across a 20-turn mixed validation set.
- Certification gate: `scripts/run-engine-certification.sh` exits 0.
- Stability confidence: D1 + F1 + G1 pass for 3 consecutive loop iterations.

## Checkpoint Runbook

### B0: Baseline Snapshot
- Confirm local branch SHA.
- Confirm `ai.local` SHA and container status.
- Capture baseline lucebox logs for: cache scope, deferred snap, RESTORE_CHAIN, daemon_lock, prefill_s, decode_tok_s, 503.
- Verify critical env knobs on `ai.local`:
  - `DFLASH_TOOL_SPLIT_ENABLED=1`
  - `DFLASH_TOOL_INLINE_SNAP_PIN=1`
  - `DFLASH_LEGACY_DAEMON=1`
  - `DFLASH_SCOPED_LOCK_WAIT_SEC`
  - `DFLASH_EPHEMERAL_LOCK_WAIT_SEC`
  - `DFLASH_DEFERRED_CONV_SNAP_MAX_TAIL`
  - `DFLASH_REQUEST_WALL_TIMEOUT_SEC`

Pass criteria:
- Baseline evidence captured.
- Runtime knobs match mission profile.

### C1: Deploy Candidate
- Pull latest `feat/vision` on `ai.local`.
- Recreate `lucebox` with profile serve.
- Verify deployed markers exist:
  - `_DeferredConvSnapJob`
  - `deferred conv snap queued`
  - `tail_bin` slicing logic
  - `RESTORE_CHAIN -1`
  - `deferred-conv-snap` lock label
- Verify `X-Conversation-Id` forwarding path remains intact.

Pass criteria:
- Deployed code and scope-forwarding path are confirmed.

### D1: Probe Pair (Primary Validation)
- Run turn 1 on the target conversation via WebUI instrumentation.
- Correlate `ai.local` logs by timestamp and conversation scope.
- Run turn 2 within about 30s.
- Correlate warm-path markers.

Pass criteria:
- One cold prefill only.
- Deferred snap queued with `tail_len`.
- Lock release occurs before deferred background execution.
- Warm lookup hit in slot 0.
- `RESTORE_CHAIN thick=0 thin=[tool_slot]` appears.
- No repeated full replay prefill.

### E1: Fix Cycle (If D1 Fails)
- Apply minimal targeted patch.
- Run focused tests:
  - `test_prefix_cache_slot_depth.py`
  - `test_server_handler_reliability.py`
- Commit -> push -> pull -> redeploy.
- Re-run D1.

Pass criteria:
- Failed condition no longer reproduces in the same scenario.

### F1: Stability Soak
- Run scoped chat requests with overlapping ephemeral probes.
- Verify scoped traffic is not starved by deferred jobs.
- Verify stream completion and valid `tool_calls`.

Pass criteria:
- No scoped lock timeout under normal single-user flow.
- No stuck/empty visible turn.

### G1: Certification
- Run `scripts/run-engine-certification.sh` through proxy.
- Archive artifacts and summarize outcomes.

Pass criteria:
- Stability gates pass.
- Performance-only deficits, if present, are tracked as a separate workstream.

### H1: Exit Reporting
- Publish before/after matrix:
  - cold prefill
  - deferred tail_len
  - warm prefill
  - lock wait outcomes
  - stream completion
  - tool_calls validity
- Record deployed SHAs for `model-runner-v4` and `ai-platform`.
- Mark north-star status.

## Anonymous Telemetry Rules
- Keep identifiers hashed or anonymous where possible.
- Keep only required trace fields:
  - `trace_id`
  - hashed conversation id
  - turn index
  - request start/end timestamps
  - engine timing extracts
- Avoid storing full raw user payload unless required for regression reproduction.

## Escalation Rules
- If D1 intermittently fails due to turn-2 race, measure frequency over multiple runs before lock-policy changes.
- If stability passes but performance remains low, open a separate daemon/C++ optimization track.
- Do not disable core architecture features to pass validation.

## End State
This SOP is the default operating model for Lucebox tool-split validation and deployment cycles until superseded by a newer approved revision.
