"""Handler reliability helpers for server_tools (no heavy runtime deps)."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Callable


# Uvicorn access-log paths suppressed when ``DFLASH_QUIET_ACCESS_LOGS=1`` (default).
_QUIET_ACCESS_LOG_PATHS = ("/health", "/v1/models", "/v1e/models")

_EPHEMERAL_LOG_DEBOUNCE_DEFAULT = 5.0
_last_ephemeral_log: float = 0.0


def should_log_ephemeral_busy() -> bool:
    """Rate-limit ephemeral ``daemon_lock busy`` log lines.

    Returns True at most once per ``DFLASH_EPHEMERAL_LOG_DEBOUNCE_SEC``
    (default 5s) so a no-backoff client cannot flood the handler log.
    """
    global _last_ephemeral_log
    raw = os.environ.get("DFLASH_EPHEMERAL_LOG_DEBOUNCE_SEC", "5")
    try:
        debounce = max(0.0, float(raw))
    except ValueError:
        debounce = _EPHEMERAL_LOG_DEBOUNCE_DEFAULT
    now = time.monotonic()
    if now - _last_ephemeral_log >= debounce:
        _last_ephemeral_log = now
        return True
    return False


class _QuietAccessLogFilter(logging.Filter):
    """Drop uvicorn access lines for high-frequency probe endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not quiet_access_logs_enabled():
            return True
        msg = record.getMessage()
        for path in _QUIET_ACCESS_LOG_PATHS:
            if f"{path} HTTP" in msg:
                return False
        return True


def quiet_access_logs_enabled() -> bool:
    """Hide ``GET /health`` and ``GET /v1/models`` uvicorn access lines (default on)."""
    raw = os.environ.get("DFLASH_QUIET_ACCESS_LOGS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def install_quiet_access_log_filter() -> None:
    """Attach filter to uvicorn access logger (idempotent)."""
    logger = logging.getLogger("uvicorn.access")
    for existing in logger.filters:
        if isinstance(existing, _QuietAccessLogFilter):
            return
    logger.addFilter(_QuietAccessLogFilter())


class DaemonBusyError(Exception):
    """Raised when the single-flight daemon lock cannot be acquired in time."""

    def __init__(self, label: str):
        self.label = label
        super().__init__(label)


class PriorityDaemonLock:
    """Single-flight lock with priority (/v1) priority over slow (/v1e).

    Queues (release order: high → mid → low):

    - **high** — all ``/v1`` traffic (scoped or not)
    - **mid** — legacy leftover (unused for new admits)
    - **low** — slow lane ``/v1e``

    Any priority enqueue drains low waiters and bumps an in-flight slow holder.
    """

    def __init__(self) -> None:
        self._held = False
        self._holder_lane: str = "priority"
        self._high: deque[asyncio.Future[None]] = deque()
        self._mid: deque[asyncio.Future[None]] = deque()
        self._low: deque[asyncio.Future[None]] = deque()
        self._bump_slow: Callable[[], None] | None = None

    def set_bump_slow_callback(self, cb: Callable[[], None] | None) -> None:
        """Called when a priority waiter needs the lock held by a slow request."""
        self._bump_slow = cb

    def locked(self) -> bool:
        return self._held

    @property
    def scoped_waiting(self) -> int:
        return len(self._high)

    @property
    def holder_lane(self) -> str:
        return self._holder_lane if self._held else ""

    async def __aenter__(self) -> PriorityDaemonLock:
        await self.acquire(scoped=True, max_wait=float("inf"))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    @staticmethod
    def _tier(*, scoped: bool, lane: str) -> str:
        # All /v1 (priority) shares high priority over /v1e (slow). Scoped vs
        # unscoped /v1 is FIFO within high; only /v1e sits on low.
        if lane == "slow":
            return "low"
        return "high"

    def _queue_for(self, tier: str) -> deque[asyncio.Future[None]]:
        if tier == "high":
            return self._high
        if tier == "mid":
            return self._mid
        return self._low

    def _drain_queue(self, queue: deque[asyncio.Future[None]], *, reason: str) -> int:
        drained = 0
        while queue:
            fut = queue.popleft()
            if not fut.done():
                fut.cancel()
                drained += 1
        if drained:
            print(
                f"  [lock] drained {drained} waiter(s) — {reason}",
                flush=True,
            )
        return drained

    def _maybe_bump_slow_holder(self, *, waiter_lane: str) -> None:
        if waiter_lane == "slow":
            return
        if self._held and self._holder_lane == "slow" and self._bump_slow is not None:
            self._bump_slow()

    async def acquire(
        self,
        *,
        scoped: bool,
        max_wait: float,
        lane: str = "priority",
    ) -> None:
        lane = "slow" if lane == "slow" else "priority"
        tier = self._tier(scoped=scoped, lane=lane)
        if not self._held:
            self._held = True
            self._holder_lane = lane
            return

        # L0 (harness): keep /v1e queued (no fail-fast / no cancel-on-/v1).
        # release() still wakes high before low. Only bump in-flight slow work.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._queue_for(tier).append(fut)
        if tier in ("high", "mid"):
            self._maybe_bump_slow_holder(waiter_lane=lane)
        try:
            if max_wait == float("inf"):
                await fut
            else:
                await asyncio.wait_for(fut, timeout=max_wait)
        except asyncio.TimeoutError:
            self._drop_waiter(fut, tier=tier)
            raise
        except asyncio.CancelledError:
            self._drop_waiter(fut, tier=tier)
            raise

    def _drop_waiter(self, fut: asyncio.Future[None], *, tier: str) -> None:
        queue = self._queue_for(tier)
        try:
            queue.remove(fut)
        except ValueError:
            pass
        if not fut.done():
            fut.cancel()

    def release(self) -> None:
        if not self._held:
            raise RuntimeError("release on unlocked PriorityDaemonLock")
        self._held = False
        self._holder_lane = "priority"
        for tier, queue in (
            ("high", self._high),
            ("mid", self._mid),
            ("low", self._low),
        ):
            while queue:
                fut = queue.popleft()
                if fut.cancelled():
                    continue
                self._held = True
                self._holder_lane = "slow" if tier == "low" else "priority"
                fut.set_result(None)
                return


class SlowLaneBumpRegistry:
    """Tracks in-flight ``/v1e`` work so ``/v1`` can preempt it.

    Each slow admission registers an ``asyncio.Event``; ``bump_all`` sets
    every active event. Generators watch the event and CANCEL + abort.
    """

    def __init__(self) -> None:
        self._events: dict[int, asyncio.Event] = {}
        self._next_id = 1

    def register(self) -> tuple[int, asyncio.Event]:
        eid = self._next_id
        self._next_id += 1
        ev = asyncio.Event()
        self._events[eid] = ev
        return eid, ev

    def unregister(self, eid: int) -> None:
        self._events.pop(eid, None)

    def bump_all(self) -> int:
        n = 0
        for ev in list(self._events.values()):
            if not ev.is_set():
                ev.set()
                n += 1
        if n:
            print(
                f"  [lock] bumping {n} in-flight slow-lane (/v1e) request(s)",
                flush=True,
            )
        return n

    @property
    def inflight(self) -> int:
        return len(self._events)


def scoped_lock_priority_enabled() -> bool:
    raw = os.environ.get("DFLASH_SCOPED_LOCK_PRIORITY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def daemon_lock_wait_seconds() -> float:
    """Max seconds to wait for the single-flight daemon lock.

    Defaults to ``DFLASH_REQUEST_WALL_TIMEOUT_SEC`` (same bound as inference).
    Set ``DFLASH_DAEMON_LOCK_WAIT_SEC=0`` for an unbounded queue (no 503 on busy).
    """
    raw = os.environ.get("DFLASH_DAEMON_LOCK_WAIT_SEC")
    if raw is None:
        return request_wall_timeout_seconds()
    try:
        val = float(raw)
    except ValueError:
        return request_wall_timeout_seconds()
    if val <= 0:
        return float("inf")
    return max(1.0, val)


def is_ephemeral_cache_scope(cache_scope: str) -> bool:
    """True when the request has no conversation id (benchmark / one-off probes)."""
    return cache_scope.startswith("ephemeral:")


def scoped_lock_wait_cap_seconds() -> float:
    """Cap lock wait for scoped (conversation-id) chat when global wait is unbounded.

    Default ``0`` → use ``DFLASH_REQUEST_WALL_TIMEOUT_SEC`` so Hermes-facing
    ``/v1`` traffic queues through a peer turn instead of 503'ing early.
    Set a positive value to force a shorter cap (legacy 180s behavior).
    """
    raw = os.environ.get("DFLASH_SCOPED_LOCK_WAIT_SEC", "0")
    try:
        val = float(raw)
    except ValueError:
        return request_wall_timeout_seconds()
    if val <= 0:
        return request_wall_timeout_seconds()
    return max(5.0, val)


def sse_keepalive_seconds() -> float:
    """SSE comment interval while waiting on daemon generate (0 disables).

    Keeps Hermes / proxies from treating long quantum gaps as a dead stream.
    """
    raw = os.environ.get("DFLASH_SSE_KEEPALIVE_SEC", "15").strip()
    try:
        val = float(raw)
    except ValueError:
        return 15.0
    if val <= 0:
        return 0.0
    return max(1.0, val)


def sse_live_emit_enabled() -> bool:
    """Emit OpenAI chat SSE content deltas as tokens arrive (not one end burst).

    Default on. When enabled, the chat-stream path detokenizes and yields
    ``delta.content`` per token so clients that track activity on real stream
    chunks (e.g. Hermes cron idle watchdog) see progress during long generates.
    SSE ``: keepalive`` comments alone are not enough for those clients.

    Set ``DFLASH_SSE_LIVE_EMIT=0`` to restore collect-all-then-burst.
    """
    raw = os.environ.get("DFLASH_SSE_LIVE_EMIT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def ephemeral_lock_wait_seconds() -> float:
    """Max lock wait for ephemeral traffic before returning 503 when the lock is busy."""
    raw = os.environ.get("DFLASH_EPHEMERAL_LOCK_WAIT_SEC", "5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def slow_lane_lock_wait_seconds() -> float:
    """Max lock/slot wait for explicit slow-lane traffic (``/v1e``).

    Defaults to 30s (longer than accidental ``/v1`` ephemeral) so title-gen and
    extractors can ride brief priority-lane occupancy without hammering. Override
    with ``DFLASH_SLOW_LANE_LOCK_WAIT_SEC``.
    """
    raw = os.environ.get("DFLASH_SLOW_LANE_LOCK_WAIT_SEC", "30")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 30.0


def chat_stream_lock_wait_seconds(*, scoped: bool, lane: str = "priority") -> float:
    """Lock wait for chat/chat-stream before 503.

    Slow lane (``/v1e``) uses ``DFLASH_SLOW_LANE_LOCK_WAIT_SEC``. Accidental
    ephemeral on ``/v1`` uses the short ephemeral cap. Scoped traffic uses
    ``DFLASH_SCOPED_LOCK_WAIT_SEC`` when ``DFLASH_DAEMON_LOCK_WAIT_SEC=0``.
    """
    if lane == "slow":
        return slow_lane_lock_wait_seconds()
    if not scoped:
        return ephemeral_lock_wait_seconds()
    base = daemon_lock_wait_seconds()
    cap = scoped_lock_wait_cap_seconds()
    if base == float("inf"):
        return cap
    return min(base, cap)


def tool_snapshot_max_kv_tokens() -> int:
    """Skip SNAPSHOT_THIN above this KV depth (daemon may crash on huge thin snaps)."""
    raw = os.environ.get("DFLASH_TOOL_SNAPSHOT_MAX_KV", "16384")
    try:
        return max(0, int(raw))
    except ValueError:
        return 16384


def tool_inline_snap_pin_enabled() -> bool:
    """Pin tool KV via inline ``snap=`` on cold prefill (Phase 1c; avoids SNAPSHOT_THIN)."""
    raw = os.environ.get("DFLASH_TOOL_INLINE_SNAP_PIN", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def deferred_conv_snap_max_tail() -> int:
    """Skip post-turn deferred conv snap when tail after tool KV exceeds this (tokens).

    Long multi-turn prompts would otherwise replay the full prompt bin (~25K) while
    holding the daemon lock. Turn 2+ builds the thick slot on demand instead.
    """
    raw = os.environ.get("DFLASH_DEFERRED_CONV_SNAP_MAX_TAIL", "8192")
    try:
        return max(0, int(raw))
    except ValueError:
        return 8192


def request_wall_timeout_seconds() -> float:
    raw = os.environ.get("DFLASH_REQUEST_WALL_TIMEOUT_SEC", "600")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def request_hard_ceiling_seconds() -> float | None:
    """Absolute safety cap for a whole request, independent of token progress.

    The token collectors enforce a *progress-aware* wall (cancel only after
    ``DFLASH_REQUEST_WALL_TIMEOUT_SEC`` with no new token), so healthy long
    generations are never guillotined by elapsed time. This hard ceiling is a
    belt-and-suspenders backstop for a truly wedged coroutine that somehow
    never yields and never stalls.

    Default ``0`` disables it (returns ``None``) — the collectors already
    guarantee termination. Set ``DFLASH_REQUEST_HARD_CEILING_SEC`` to a large
    value (e.g. ``3600``) to re-enable an absolute cap.
    """
    raw = os.environ.get("DFLASH_REQUEST_HARD_CEILING_SEC", "0")
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None
