"""Tagged token-stream demux for ``--stream-tagged`` daemon frames.

Frame wire format (little-endian int32), matching lucebox
``daemon_scheduler.h`` / ``DaemonIO::emit``:

+ ``[-2, req_id, tok]`` — tagged token (``tok >= 0``), DONE (``tok == -1``),
  or CONTINUE (``tok == -4``)
+ bare ``[tok]`` — legacy untagged path

A process-wide :class:`TaggedStreamDemux` owns the stream fd and routes
frames into per-``req_id`` queues so multiple HTTP handlers can share one pipe.
"""
from __future__ import annotations

import asyncio
import os
import struct
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

STREAM_TAG_MARKER = -2
STREAM_DONE_SENTINEL = -1
STREAM_CONTINUE_SENTINEL = -4


@dataclass(frozen=True)
class StreamFrame:
    """One demuxed frame from the daemon token pipe."""

    kind: str  # "tag" | "done" | "cont" | "tok"
    value: int
    req_id: int | None


class TaggedFrameBuffer:
    """Byte buffer → demuxed :class:`StreamFrame` list (sync, reentrant-safe)."""

    def __init__(self) -> None:
        self._buf = b""

    def clear(self) -> None:
        self._buf = b""

    def push(self, chunk: bytes) -> list[StreamFrame]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[StreamFrame] = []
        while len(self._buf) >= 4:
            (v,) = struct.unpack_from("<i", self._buf, 0)
            if v == STREAM_TAG_MARKER:
                if len(self._buf) < 12:
                    break
                _, req_id, tok = struct.unpack_from("<iii", self._buf, 0)
                self._buf = self._buf[12:]
                if tok == STREAM_DONE_SENTINEL:
                    kind = "done"
                elif tok == STREAM_CONTINUE_SENTINEL:
                    kind = "cont"
                else:
                    kind = "tag"
                out.append(StreamFrame(kind=kind, value=tok, req_id=req_id))
            else:
                self._buf = self._buf[4:]
                if v == STREAM_DONE_SENTINEL:
                    kind = "done"
                elif v == STREAM_CONTINUE_SENTINEL:
                    kind = "cont"
                else:
                    kind = "tok"
                out.append(StreamFrame(kind=kind, value=v, req_id=None))
        return out


def pack_tagged_frame(req_id: int, tok: int) -> bytes:
    """Pack a tagged ``[-2, req_id, tok]`` frame (tests / synthetic inject)."""
    return struct.pack("<iii", STREAM_TAG_MARKER, int(req_id), int(tok))


def pack_bare_token(tok: int) -> bytes:
    return struct.pack("<i", int(tok))


class TaggedStreamDemux:
    """Background drain of ``stream_fd`` with per-request asyncio queues.

    Always drains the pipe (even when no subscribers) so the daemon cannot
    block on a full write. Untagged frames fan out to ``req_id=0`` when that
    queue is registered, otherwise they are dropped (legacy exclusive path
    should not enable this demux).
    """

    def __init__(self, stream_fd: int) -> None:
        self._fd = int(stream_fd)
        self._parser = TaggedFrameBuffer()
        self._queues: dict[int, asyncio.Queue[StreamFrame | None]] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._next_req_id = 1
        self._req_lock = threading.Lock()

    def alloc_req_id(self) -> int:
        with self._req_lock:
            rid = self._next_req_id
            self._next_req_id += 1
            return rid

    async def start(self) -> None:
        if self._task is not None:
            return
        self._closed = False
        self._task = asyncio.create_task(self._pump(), name="tagged-stream-demux")

    async def stop(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        async with self._lock:
            for q in self._queues.values():
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._queues.clear()

    async def register(self, req_id: int) -> asyncio.Queue[StreamFrame | None]:
        q: asyncio.Queue[StreamFrame | None] = asyncio.Queue(maxsize=0)
        async with self._lock:
            if req_id in self._queues:
                raise ValueError(f"req_id {req_id} already registered")
            self._queues[req_id] = q
        return q

    async def unregister(self, req_id: int) -> None:
        async with self._lock:
            q = self._queues.pop(req_id, None)
        if q is not None:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def _route(self, frame: StreamFrame) -> None:
        rid = frame.req_id if frame.req_id is not None else 0
        q = self._queues.get(rid)
        if q is None:
            return
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    async def _pump(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                chunk = await loop.run_in_executor(None, os.read, self._fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            frames = self._parser.push(chunk)
            if not frames:
                continue
            async with self._lock:
                for fr in frames:
                    self._route(fr)
        async with self._lock:
            for q in self._queues.values():
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    async def iter_tokens(
        self,
        req_id: int,
        n_gen: int,
        stop_ids: set[int] | frozenset[int] | None = None,
        *,
        wall_timeout: float = 600.0,
    ) -> AsyncIterator[int]:
        """Yield vocab tokens for ``req_id`` until DONE, n_gen, or timeout.

        CONTINUE frames are skipped (scheduler will emit more tokens later).
        """
        stops = stop_ids or frozenset()
        q = await self.register(req_id)
        generated = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wall_timeout
        try:
            while generated < n_gen:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    break
                if item.kind == "cont":
                    continue
                if item.kind == "done":
                    break
                tok = item.value
                if tok in stops:
                    continue
                if tok < 0:
                    continue
                generated += 1
                yield tok
        finally:
            await self.unregister(req_id)


def format_req_command(line: str, req_id: int) -> str:
    """Prefix ``REQ <id> `` onto a daemon command (preserves trailing newline)."""
    nl = line.endswith("\n")
    body = line[:-1] if nl else line
    body = body.strip()
    if body.upper().startswith("REQ "):
        out = body
    else:
        out = f"REQ {int(req_id)} {body}"
    return out + ("\n" if nl else "")


def iter_frames_from_bytes(data: bytes) -> Iterator[StreamFrame]:
    """Parse a complete byte blob into frames (unit-test helper)."""
    buf = TaggedFrameBuffer()
    yield from buf.push(data)
