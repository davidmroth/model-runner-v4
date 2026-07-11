"""Helpers for reading int32 LE token streams from the test_dflash daemon pipe.

The legacy inline daemon loop always emits a ``-1`` sentinel after each
command.  The layer-split ``run_daemon()`` path (``daemon_loop.cpp``) streams
committed decode tokens but does **not** emit ``-1`` on successful generate —
only on error/compress ack paths.  Readers must stop after ``n_gen`` tokens (or
EOF) instead of blocking forever waiting for a sentinel.
"""
from __future__ import annotations

import asyncio
import os
import struct
import sys
from typing import AsyncIterator, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore


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


def iter_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
) -> Iterator[int]:
    """Sync generator over one daemon decode stream."""
    stops = stop_ids or frozenset()
    generated = 0
    while generated < n_gen:
        b = os.read(r, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stops:
            continue
        generated += 1
        yield tok_id


async def async_iter_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
) -> AsyncIterator[int]:
    """Async generator: one 4-byte read per worker-thread hop."""
    stops = stop_ids or frozenset()
    generated = 0
    while generated < n_gen:
        b = await asyncio.to_thread(os.read, r, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stops:
            continue
        generated += 1
        yield tok_id


def collect_pipe_tokens(
    r: int,
    n_gen: int,
    stop_ids: set[int] | frozenset[int] | None = None,
) -> list[int]:
    return list(iter_pipe_tokens(r, n_gen, stop_ids))
