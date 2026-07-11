# Anonymous Feedback Loop Scorecard

Use this template for each loop iteration. One section equals one full iteration.

## Iteration Header
- Iteration ID:
- Date/Time (UTC):
- Operator:
- Local SHA (`model-runner-v4`):
- Deployed SHA (`ai.local`):
- Conversation ID (hashed/reference):
- WebUI run link:

## Checkpoint Status
- B0 Baseline Snapshot: PASS / FAIL
- C1 Deploy Candidate: PASS / FAIL
- D1 Probe Pair: PASS / FAIL
- E1 Fix Cycle (if used): PASS / FAIL / N/A
- F1 Stability Soak: PASS / FAIL
- G1 Certification: PASS / FAIL
- H1 Exit Reporting: PASS / FAIL

## Quantitative Metrics
- Cold turn prefill_s:
- Deferred queued tail_len:
- Deferred daemon N (should align to tail work, not full prompt):
- Warm turn prefill_s:
- Warm marker seen (`lookup hit slot=0`): YES / NO
- Warm marker seen (`RESTORE_CHAIN thick=0`): YES / NO
- Scoped lock timeout count (`60s`):
- Stuck/empty turns in 20-turn set:
- Certification exit code:

## Required Log Markers
- Cache scope marker seen: YES / NO
- Deferred queued marker seen: YES / NO
- Deferred background marker seen: YES / NO
- Full replay duplicate prefill seen: YES / NO
- `[DONE]` stream completion seen: YES / NO

## Decisions and Changes
- Fix applied:
- Files changed:
- Test commands run:
- Test results summary:
- Commit hash:
- Deploy command summary:

## Outcome
- Iteration result: PASS / FAIL
- Failure checkpoint (if any):
- Next action:

---

## Stability Declaration Block (After Consecutive Passes)
- Consecutive passing iterations (D1 + F1 + G1):
- Stability declaration: APPROVED / NOT APPROVED
- Notes:
