"""Handler reliability helpers for server_tools (no heavy runtime deps)."""
from __future__ import annotations

import asyncio
import logging
import os
from collections import deque


# Uvicorn access-log paths suppressed when ``DFLASH_QUIET_ACCESS_LOGS=1`` (default).
_QUIET_ACCESS_LOG_PATHS = ("/health", "/v1/models")


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
    """Single-flight lock with scoped (conversation) priority over ephemeral traffic.

    Scoped requests jump ahead of ephemeral waiters when the lock is free.
    Ephemeral acquire fails immediately while any scoped request is queued.
    """

    def __init__(self) -> None:
        self._held = False
        self._high: deque[asyncio.Future[None]] = deque()
        self._low: deque[asyncio.Future[None]] = deque()

    def locked(self) -> bool:
        return self._held

    @property
    def scoped_waiting(self) -> int:
        return len(self._high)

    async def __aenter__(self) -> PriorityDaemonLock:
        await self.acquire(scoped=True, max_wait=float("inf"))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    async def acquire(self, *, scoped: bool, max_wait: float) -> None:
        if not scoped and self._high:
            raise DaemonBusyError("ephemeral-yields-to-scoped")

        if not self._held:
            self._held = True
            return

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        (self._high if scoped else self._low).append(fut)
        try:
            if max_wait == float("inf"):
                await fut
            else:
                await asyncio.wait_for(fut, timeout=max_wait)
        except asyncio.TimeoutError:
            self._drop_waiter(fut, scoped=scoped)
            raise
        except asyncio.CancelledError:
            self._drop_waiter(fut, scoped=scoped)
            raise

    def _drop_waiter(self, fut: asyncio.Future[None], *, scoped: bool) -> None:
        queue = self._high if scoped else self._low
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
        while self._high:
            fut = self._high.popleft()
            if fut.cancelled():
                continue
            self._held = True
            fut.set_result(None)
            return
        while self._low:
            fut = self._low.popleft()
            if fut.cancelled():
                continue
            self._held = True
            fut.set_result(None)
            return


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
    """Cap lock wait for scoped (conversation-id) chat when global wait is unbounded."""
    raw = os.environ.get("DFLASH_SCOPED_LOCK_WAIT_SEC", "60")
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 60.0


def ephemeral_lock_wait_seconds() -> float:
    """Max lock wait for ephemeral traffic before returning 503 when the lock is busy."""
    raw = os.environ.get("DFLASH_EPHEMERAL_LOCK_WAIT_SEC", "5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def chat_stream_lock_wait_seconds(*, scoped: bool) -> float:
    """Lock wait for chat/chat-stream before 503.

    Ephemeral/benchmark traffic gets a short cap so user conversations are not
    stuck behind multi-minute cold prefills. Scoped traffic uses
    ``DFLASH_SCOPED_LOCK_WAIT_SEC`` when ``DFLASH_DAEMON_LOCK_WAIT_SEC=0``.
    """
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


def request_wall_timeout_seconds() -> float:
    raw = os.environ.get("DFLASH_REQUEST_WALL_TIMEOUT_SEC", "600")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0
