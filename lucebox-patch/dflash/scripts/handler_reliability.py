"""Handler reliability helpers for server_tools (no heavy runtime deps)."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque


# Uvicorn access-log paths suppressed when ``DFLASH_QUIET_ACCESS_LOGS=1`` (default).
_QUIET_ACCESS_LOG_PATHS = ("/health", "/v1/models")

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
    """Single-flight lock with scoped (conversation) priority over ephemeral traffic.

    Scoped requests jump ahead of ephemeral waiters when the lock is free.
    Ephemeral waiters join ``_low`` and wait up to ``max_wait``; they are
    cancelled when a scoped request enqueues, and ``release()`` always drains
    ``_high`` before ``_low``.
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
        if not self._held:
            self._held = True
            return

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        (self._high if scoped else self._low).append(fut)
        if scoped:
            self._drain_low_waiters()
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

    def _drain_low_waiters(self) -> None:
        """Cancel queued ephemeral waiters when a scoped request enqueues.

        Fires when a new scoped request joins ``_high``. Combined with
        ``release()`` draining ``_high`` before ``_low``, this keeps
        ephemerals from running while any scoped waiter exists. The current
        lock *holder* (if any) is not affected — only queued-but-not-running
        ephemerals are cancelled.
        """
        drained = 0
        while self._low:
            fut = self._low.popleft()
            if not fut.done():
                fut.cancel()
                drained += 1
        if drained:
            print(
                f"  [lock] drained {drained} ephemeral waiter(s) — scoped enqueued",
                flush=True,
            )

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
    """Cap lock wait for scoped (conversation-id) chat when global wait is unbounded.

    Default 180s: cold prefill of 20K+ tokens takes ~90s without PFlash, so
    the cap must exceed the longest expected inference to avoid a scoped request
    timing out before it can acquire the lock.
    """
    raw = os.environ.get("DFLASH_SCOPED_LOCK_WAIT_SEC", "180")
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 180.0


def ephemeral_lock_wait_seconds() -> float:
    """Max lock wait for ephemeral traffic before returning 503 when the lock is busy."""
    raw = os.environ.get("DFLASH_EPHEMERAL_LOCK_WAIT_SEC", "5")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def ephemeral_max_tokens() -> int:
    """Hard cap on completion tokens for ephemeral (no conversation id) traffic.

    Background extractors often send ``max_tokens=64000`` which can hold a live
    slot for minutes and starve scoped chat under N=2. Default 2048.
    """
    raw = os.environ.get("DFLASH_EPHEMERAL_MAX_TOKENS", "2048")
    try:
        return max(16, min(int(raw), 65536))
    except ValueError:
        return 2048


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
