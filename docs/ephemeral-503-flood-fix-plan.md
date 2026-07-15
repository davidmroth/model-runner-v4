# Ephemeral 503 Flood — Fix Plan

**Status:** Implemented  

**Scope:** `lucebox-patch/dflash/scripts/handler_reliability.py`, `lucebox-patch/dflash/scripts/server_tools.py`, `services/proxy/alias_proxy.py`

---

## Problem Summary

Under normal scoped-conversation load, ephemeral (no-`X-Conversation-Id`) requests
produce a log flood that looks like this:

```
[pc] cache scope ephemeral (no X-Conversation-Id)
[handler] ephemeral yield — scoped waiter ahead (chat)
INFO: 172.22.0.4:38042 - "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable
… (repeated 70-140 times over ~28 seconds)
```

The flood is caused by two independent failures working together:

1. **Server-side** — The ephemeral yield pre-check issues an **immediate 503 with no
   wait**, bypassing the existing `DFLASH_EPHEMERAL_LOCK_WAIT_SEC=5s` hold mechanism.
2. **Client-side** — The upstream caller (Hermes agent / proxy) retries 503 responses
   with effectively no backoff (~200ms between attempts).

Together they produce one log triplet per ~200ms for the entire duration of the scoped
lock hold, typically 20–340 s per turn.

---

## Background: Concurrency Models

### llama.cpp (N-slot continuous batching)

llama.cpp allocates `--parallel N` independent KV-cache regions (slots). Multiple
conversations share the same GPU forward pass via `llama_decode()` continuous batching —
they are genuinely parallel because Transformer attention is independent per sequence
position. Incoming requests queue internally and are assigned to a free slot; 503 is
only returned if the queue is also full.

### lucebox / dflash (single-daemon serial)

The dflash daemon is architecturally serial:

- One stdin/stdout pipe; one draft→verify→replay loop at a time.
- SNAPSHOT_THIN / RESTORE_CHAIN restores a global state snapshot, not per-slot
  independent blocks.
- Running N conversations in parallel would require N daemon processes each with
  their own full GPU allocation (~8 GB each) — a different product.

For this architecture, **some form of 503 on overload is unavoidable** and correct.
The question is how quickly to emit it and how clients should behave.

---

## Root Cause (Server Side)

`server_tools.py` calls `_acquire_daemon_lock()` which has this guard before any lock
attempt:

```python
# server_tools.py — _acquire_daemon_lock()
if not scoped and use_priority and daemon_lock.scoped_waiting:
    print(
        "  [handler] ephemeral yield — scoped waiter ahead "
        f"({label})",
        flush=True,
    )
    raise DaemonBusyError(label)        # ← instant 503, no wait
```

`PriorityDaemonLock.acquire()` has the same check mirrored:

```python
# handler_reliability.py — PriorityDaemonLock.acquire()
if not scoped and self._high:
    raise DaemonBusyError("ephemeral-yields-to-scoped")   # ← instant 503
```

Both checks fire before the ephemeral request is ever placed into the `_low` wait
queue. They were added to prevent ephemeral traffic from "sneaking in" ahead of a
queued scoped request.

**The checks are redundant.** The priority invariant is already enforced by the
queue machinery in two places:

1. `PriorityDaemonLock.release()` drains `_high` (scoped) completely before it
   ever touches `_low` (ephemeral). A queued ephemeral cannot run while any scoped
   waiter exists.
2. `PriorityDaemonLock._drain_low_waiters()` is called inside `acquire()` whenever
   a new scoped request enqueues (`if scoped: self._drain_low_waiters()`). This
   cancels any ephemeral that is currently waiting in `_low` the moment a new
   scoped request arrives.

Removing the two early-bail checks changes the observable behavior only in
log volume: instead of a 503 every ~200 ms, the ephemeral receives a 503 at most
once every `DFLASH_EPHEMERAL_LOCK_WAIT_SEC` seconds (default 5 s).

---

## Root Cause (Client Side)

`alias_proxy.py` forwards 503 responses from the lucebox backend transparently with
no added retry delay. The upstream caller (Hermes agent) retries inference 503s with
no back-off, creating the tight retry loop.

The lucebox server does set `Retry-After` in `_busy_response()`:

```python
headers["Retry-After"] = str(max(1, int(wait_sec)))
```

However, this is **never reached** in the ephemeral yield path because the
`DaemonBusyError` is raised before `_busy_response()` is called. Even after the
server fix, the proxy should explicitly re-attach the header to prevent future
regressions when new 503 paths are added.

---

## Planned Changes

### Change 1 — `handler_reliability.py`: Remove early bail from `PriorityDaemonLock.acquire()`

**File:** `lucebox-patch/dflash/scripts/handler_reliability.py`

Remove the two lines that immediately raise `DaemonBusyError` before the wait:

```python
# REMOVE these two lines from PriorityDaemonLock.acquire():
if not scoped and self._high:
    raise DaemonBusyError("ephemeral-yields-to-scoped")
```

After removal, `acquire()` falls through to the normal `_low` queue path. The
ephemeral future is appended to `_low`, waits up to `max_wait` seconds, and is
cancelled by `_drain_low_waiters()` the moment any new scoped request enqueues.

No change to the priority semantics: `release()` still drains `_high` before `_low`.
An ephemeral that is waiting when the lock releases will only run if `_high` is empty.

**Update the `_drain_low_waiters` docstring** to reflect that it fires both when a
new scoped request enqueues AND (implicitly) keeps ephemerals from running after a
scoped request has been drained from `_high` by `release()`.

---

### Change 2 — `server_tools.py`: Remove ephemeral yield pre-check from `_acquire_daemon_lock()`

**File:** `lucebox-patch/dflash/scripts/server_tools.py`

Remove the redundant guard block from `_acquire_daemon_lock()`:

```python
# REMOVE from _acquire_daemon_lock():
if not scoped and use_priority and daemon_lock.scoped_waiting:
    print(
        "  [handler] ephemeral yield — scoped waiter ahead "
        f"({label})",
        flush=True,
    )
    raise DaemonBusyError(label)
```

After removal, ephemeral requests flow into the standard `daemon_lock.acquire(scoped=False, max_wait=ephemeral_lock_wait_seconds())` path and wait the
configured `DFLASH_EPHEMERAL_LOCK_WAIT_SEC` seconds (default 5 s) before timing out.

The `_busy_response()` call that follows the `DaemonBusyError` catch will then
correctly set `Retry-After: 5` on every ephemeral 503 — previously this header was
absent on the yield path.

**Log volume impact:** For a 28 s scoped lock hold the ephemeral caller receives
≈6 503s (one every 5 s) instead of ≈140 (one every ~200 ms).

---

### Change 3 — `alias_proxy.py`: Forward `Retry-After` from backend 503 to caller

**File:** `services/proxy/alias_proxy.py`

When the proxy receives a 503 from the backend (lucebox) that includes a `Retry-After`
header, copy that header to the response returned to the upstream caller.

The proxy currently preserves backend headers only for streaming responses. For the
non-streaming error path (`_backend_status_response` and the slot-saturation fallback)
add a utility function:

```python
def _copy_retry_after(backend_response: httpx.Response, headers: dict) -> None:
    """Copy Retry-After from backend 503 to caller response, if present."""
    val = backend_response.headers.get("retry-after")
    if val:
        headers["Retry-After"] = val
```

Call this in the error branches that forward backend 503s:

```python
# Inside the inference proxy path, after detecting a 503 backend response:
out_headers: dict[str, str] = {}
_copy_retry_after(backend_response, out_headers)
return JSONResponse(
    status_code=503,
    content=backend_body,
    headers=out_headers,
)
```

This ensures well-behaved HTTP clients (OpenAI Python SDK, httpx with retry
transport, etc.) honor the server's declared back-off window automatically.

---

### Change 4 — `handler_reliability.py`: Rate-limit repeated ephemeral log messages

**File:** `lucebox-patch/dflash/scripts/handler_reliability.py`

Even after Changes 1–2, a client with no back-off will still produce one log entry
per `DFLASH_EPHEMERAL_LOCK_WAIT_SEC` seconds. With the default of 5 s and a typical
scoped hold of 30 s that is 6 entries — acceptable. No code change strictly required.

However, add an optional debounce for the `daemon_lock busy — queueing` message to
guard against future regressions or operators who set `DFLASH_EPHEMERAL_LOCK_WAIT_SEC`
very low. Add a module-level `_last_ephemeral_log: float = 0.0` guard and only print
if the last ephemeral log was more than `DFLASH_EPHEMERAL_LOG_DEBOUNCE_SEC` (default
5 s) ago:

```python
_EPHEMERAL_LOG_DEBOUNCE_SEC = float(
    os.environ.get("DFLASH_EPHEMERAL_LOG_DEBOUNCE_SEC", "5")
)
_last_ephemeral_log: float = 0.0


def _should_log_ephemeral_busy() -> bool:
    global _last_ephemeral_log
    import time
    now = time.monotonic()
    if now - _last_ephemeral_log >= _EPHEMERAL_LOG_DEBOUNCE_SEC:
        _last_ephemeral_log = now
        return True
    return False
```

Use this guard inside `_acquire_daemon_lock()` around the `daemon_lock busy` print.

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| 503 log lines per 28 s scoped hold | ~140 (1 per ~200 ms) | ~6 (1 per 5 s) |
| `Retry-After` header present on ephemeral 503 | No | Yes (5 s) |
| Scoped priority invariant preserved | Yes | Yes (unchanged) |
| `_drain_low_waiters` still cancels ephemerals on scoped enqueue | Yes | Yes (unchanged) |
| Release order: scoped before ephemeral | Yes | Yes (unchanged) |

---

## Testing

### Unit tests (`tests/` in model-runner-v4)

Add cases to `test_server_handler_reliability.py`:

1. **Ephemeral waits instead of instant-503 when scoped waiter present**:
   Acquire lock with scoped=True (hold it). Enqueue one scoped waiter. Fire an
   ephemeral acquire. Assert the ephemeral does NOT raise immediately; assert it
   eventually raises `DaemonBusyError` after `max_wait` expires (not before).

2. **Ephemeral is cancelled when new scoped enqueues while it waits**:
   Acquire lock. Start an ephemeral with `max_wait=10`. Then enqueue a scoped
   waiter. Assert `_drain_low_waiters` cancels the ephemeral within the asyncio
   event loop tick, well before `max_wait` expires.

3. **Scoped still runs before ephemeral after lock release**:
   Hold lock. Enqueue one ephemeral then one scoped. Release lock. Assert scoped
   acquires first; ephemeral acquires only after scoped releases.

4. **`Retry-After` header present on ephemeral 503 response**:
   Integration-style test using `httpx.AsyncClient` against the FastAPI test
   client. Assert that a 503 response on an ephemeral request while a scoped
   request is queued includes `Retry-After` with a value of
   `DFLASH_EPHEMERAL_LOCK_WAIT_SEC`.

### Manual smoke

```bash
# Hold the daemon busy with a scoped conversation turn, then fire a
# concurrent ephemeral probe and observe: only ~1 503 per 5 s, each
# with Retry-After: 5, and no more than 6 total for a 30 s hold.
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-Conversation-Id: smoke-test-$(date +%s)" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"count to 500"}],"stream":false}' &

sleep 2

for i in $(seq 1 10); do
  curl -sS -w "\nHTTP %{http_code} Retry-After: %header{retry-after}\n" \
    -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"hi"}],"stream":false}' \
    -o /dev/null
  sleep 1
done
```

Expected: most of the 10 probe calls either wait up to 5 s and return 503, or (if
the scoped turn finishes first) succeed. No more than 2 distinct 503 responses in
the 10-second window.

---

## Rollback

Set `DFLASH_SCOPED_LOCK_PRIORITY=0` to disable priority scheduling entirely (legacy
FIFO). This restores the old behavior without reverting code. The flag is already
wired into `scoped_lock_priority_enabled()` in `handler_reliability.py`.

---

## Slow lane: `/v1e` (follow-on)

Explicit background path on the **same daemon** as `/v1`:

| Path | Lane | Admission |
|------|------|-----------|
| `/v1/chat/completions` | fast | scoped sticky; long hold; may use all live slots |
| `/v1e/chat/completions` | slow | always ephemeral; short/medium wait; **cannot take reserved fast slot** |

Knobs:

- `DFLASH_RESERVED_FAST_SLOTS` — auto `1` when `N>=2`, else `0`
- `DFLASH_SLOW_LANE_LOCK_WAIT_SEC` — default `30`
- `DFLASH_SLOW_LANE_MAX_TOKENS` — defaults to `DFLASH_EPHEMERAL_MAX_TOKENS`

Point Hermes aux (e.g. `auxiliary.title_generation.base_url`) at the host with
path prefix `/v1e` (OpenAI client appends `/chat/completions`). Proxy
`MODEL_PATHS` includes `/v1e/chat/completions` so alias rewrite still applies.

---

## Files to Change

```
lucebox-patch/dflash/scripts/handler_reliability.py   (Changes 1, 4 + /v1e waits)
lucebox-patch/dflash/scripts/server_tools.py          (Change 2 + /v1e routes)
lucebox-patch/dflash/scripts/target_cache_admission.py (reserved fast slots)
services/proxy/alias_proxy.py                         (Change 3 + /v1e path)
tests/test_server_handler_reliability.py              (new test cases)
```
