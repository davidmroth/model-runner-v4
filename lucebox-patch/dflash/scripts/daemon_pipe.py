"""Helpers for reading int32 LE token streams from the test_dflash daemon pipe.

The legacy inline daemon loop always emits a ``-1`` sentinel after each
command.  The layer-split ``run_daemon()`` path (``daemon_loop.cpp``) streams
committed decode tokens but does **not** emit ``-1`` on successful generate —
only on error/compress ack paths.  Readers must stop after the daemon-reported
decode count (via ``DaemonStdoutBus``), after ``n_gen`` tokens, on ``-1``,
EOF, or a post-token idle gap — not by blocking until ``n_gen`` is exhausted.
"""
from __future__ import annotations

import asyncio
import os
import struct
import sys
import time
from typing import TYPE_CHECKING, AsyncIterator, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

try:
    import select
except ImportError:  # pragma: no cover - Windows
    select = None  # type: ignore

if TYPE_CHECKING:
    from prefix_cache import DaemonStdoutBus


def drain_pipe_residual(r: int) -> None:
    """Drop unread bytes from ``r`` (non-blocking). Best-effort cleanup."""
    if sys.platform == "win32" or fcntl is None:
        return
    try:
        flags = fcntl.fcntl(r, fcntl.F_GETFL)
    except OSError:
        return
    try:
        fcntl.fcntl(r, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        while True:
            try:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
            except BlockingIOError:
                break
    finally:
        try:
            fcntl.fcntl(r, fcntl.F_SETFL, flags)
        except OSError:
            pass


def _daemon_reported_completion(bus: "DaemonStdoutBus | None") -> int | None:
    if bus is None:
        return None
    raw = bus.request_timings().get("completion_tokens")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def iter_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
    *,
    bus: "DaemonStdoutBus | None" = None,
    wall_timeout: float = 600.0,
    post_token_idle: float = 3.0,
) -> Iterator[int]:
    """Sync generator over one daemon decode stream."""
    stops = stop_ids or frozenset()
    generated = 0
    started = time.monotonic()
    last_token_at: float | None = None
    use_select = select is not None and sys.platform != "win32"

    while generated < n_gen:
        # Progress-aware wall: measure the gap since the last real token (or
        # start, before the first token) rather than total elapsed, so a healthy
        # long generation is never guillotined by an absolute cap.
        wall_ref = last_token_at if last_token_at is not None else started
        if time.monotonic() - wall_ref > wall_timeout:
            break
        reported = _daemon_reported_completion(bus)
        if reported is not None and generated >= reported:
            break
        if last_token_at is not None and time.monotonic() - last_token_at > post_token_idle:
            break

        if use_select:
            if last_token_at is None:
                wait = min(30.0, wall_timeout - (time.monotonic() - started))
            else:
                wait = min(
                    post_token_idle - (time.monotonic() - last_token_at),
                    wall_timeout - (time.monotonic() - last_token_at),
                )
            if wait <= 0:
                break
            ready, _, _ = select.select([r], [], [], wait)
            if not ready:
                if last_token_at is not None:
                    break
                continue

        b = os.read(r, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stops:
            continue
        generated += 1
        last_token_at = time.monotonic()
        yield tok_id


async def async_iter_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
    *,
    bus: "DaemonStdoutBus | None" = None,
    wall_timeout: float = 600.0,
    post_token_idle: float = 3.0,
) -> AsyncIterator[int]:
    """Async generator: one 4-byte read per worker-thread hop."""
    stops = stop_ids or frozenset()
    generated = 0
    started = time.monotonic()
    last_token_at: float | None = None
    while generated < n_gen:
        # Progress-aware wall: gap since last real token (or start), not total.
        wall_ref = last_token_at if last_token_at is not None else started
        if time.monotonic() - wall_ref > wall_timeout:
            break
        reported = _daemon_reported_completion(bus)
        if reported is not None and generated >= reported:
            break
        if last_token_at is not None and time.monotonic() - last_token_at > post_token_idle:
            break
        b = await asyncio.to_thread(os.read, r, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stops:
            continue
        generated += 1
        last_token_at = time.monotonic()
        yield tok_id


def collect_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
    *,
    bus: "DaemonStdoutBus | None" = None,
    wall_timeout: float = 600.0,
    post_token_idle: float = 3.0,
) -> list[int]:
    return list(
        iter_pipe_tokens(
            r,
            n_gen,
            stop_ids,
            bus=bus,
            wall_timeout=wall_timeout,
            post_token_idle=post_token_idle,
        )
    )
