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

from request_correlation import OrphanFrameMeter

STREAM_TAG_MARKER = -2
STREAM_DONE_SENTINEL = -1
STREAM_CONTINUE_SENTINEL = -4

# End reasons for :meth:`TaggedStreamDemux.iter_tokens` (Phase A visibility).
END_DONE = "done"
END_STOP = "stop"
END_N_GEN = "n_gen"
END_IDLE = "idle"
END_CONTINUE_IDLE = "continue_idle"
END_WALL = "wall"
END_PIPE = "pipe_closed"


@dataclass(frozen=True)
class StreamFrame:
    """One demuxed frame from the daemon token pipe."""

    kind: str  # "tag" | "done" | "cont" | "tok"
    value: int
    req_id: int | None


@dataclass
class StreamCollectOutcome:
    """Side-channel result for an ``iter_tokens`` collect (truncation forensics)."""

    end_reason: str = "unknown"
    generated: int = 0
    awaiting_continue: bool = False
    admit_hold: bool = False
    admit_remaining: int | None = None

    @property
    def is_truncated(self) -> bool:
        """True when the stream stopped early while more decode was expected."""
        if self.end_reason in (END_DONE, END_STOP, END_N_GEN):
            return False
        if self.end_reason not in (
            END_IDLE,
            END_CONTINUE_IDLE,
            END_WALL,
            END_PIPE,
            "unknown",
        ):
            return False
        if self.awaiting_continue:
            return True
        if self.admit_hold and self.admit_remaining is not None:
            return self.admit_remaining > 0
        # Admit-hold without a remaining sample: any non-clean end is truncation.
        return bool(self.admit_hold)


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
        # Frames for unregistered req_ids (orphan decode after CANCEL / mix).
        self._orphan_meter = OrphanFrameMeter()

    @property
    def orphan_meter(self) -> OrphanFrameMeter:
        return self._orphan_meter

    def mark_after_collect_cancel(self, *, window_sec: float = 5.0) -> None:
        """Arm orphan-after-CANCEL window (Phase A: demux_orphan_after_collect)."""
        self._orphan_meter.mark_after_collect(window_sec=window_sec)

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
            # Untagged legacy frames use req_id=None → rid=0; ignore those
            # when exclusive path has no subscriber. Tagged orphans matter.
            if frame.req_id is not None:
                self._orphan_meter.note(int(frame.req_id), frame.kind)
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
        post_token_idle: float = 30.0,
        continue_idle: float = 60.0,
        admit_hold: bool = False,
        outcome: StreamCollectOutcome | None = None,
        queue: asyncio.Queue[StreamFrame | None] | None = None,
    ) -> AsyncIterator[int]:
        """Yield vocab tokens for ``req_id`` until DONE, stop, n_gen, or timeout.

        ``wall_timeout`` is a *progress-aware* bound: it caps the gap between
        real tokens, not total request time. Each yielded token pushes the
        deadline forward, so an arbitrarily long but healthy generation runs to
        completion; only a stream that produces no real token for
        ``wall_timeout`` seconds (even one emitting CONTINUE heartbeats) is
        cancelled.

        When ``admit_hold`` is True (quantum-admit path with live remaining),
        short ``continue_idle`` / ``post_token_idle`` exits are disabled: the
        collector holds until DONE, stop, ``n_gen``, or the progress-aware wall.
        Slow SCHED between quanta must not silently end the HTTP stream.

        When ``admit_hold`` is False (legacy / non-admit): CONTINUE replaces
        ``post_token_idle`` with the longer ``continue_idle`` so a slow SCHED
        kick is tolerated briefly without hanging until ``wall_timeout`` if
        SCHED never returns.

        Stop ids end the stream (same as a soft EOS) — do not ``continue`` past
        them or we hang waiting for DONE when the engine already finished.

        Pass ``queue`` from a prior :meth:`register` so the caller can write the
        daemon command before frames arrive (avoids a subscribe race).
        Pass ``outcome`` to capture ``end_reason`` / truncation for Phase A logs.
        """
        stops = stop_ids or frozenset()
        own_reg = queue is None
        q = queue if queue is not None else await self.register(req_id)
        generated = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wall_timeout
        last_token_at: float | None = None
        # After CONTINUE (non-admit): wait up to continue_idle for next burst.
        awaiting_continue = False
        continue_deadline: float | None = None
        end_reason = END_N_GEN
        try:
            while generated < n_gen:
                now = loop.time()
                remaining = deadline - now
                if remaining <= 0:
                    end_reason = END_WALL
                    break
                if not admit_hold:
                    if awaiting_continue and continue_deadline is not None:
                        cont_left = continue_deadline - now
                        if cont_left <= 0:
                            end_reason = END_CONTINUE_IDLE
                            break
                        remaining = min(remaining, cont_left)
                    elif last_token_at is not None and post_token_idle > 0:
                        idle_left = post_token_idle - (now - last_token_at)
                        if idle_left <= 0:
                            end_reason = END_IDLE
                            break
                        remaining = min(remaining, idle_left)
                try:
                    item = await asyncio.wait_for(q.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    # Distinguish wall vs short-idle using the same rules as above.
                    now = loop.time()
                    if now >= deadline:
                        end_reason = END_WALL
                    elif (
                        not admit_hold
                        and awaiting_continue
                        and continue_deadline is not None
                        and now >= continue_deadline
                    ):
                        end_reason = END_CONTINUE_IDLE
                    elif not admit_hold and last_token_at is not None:
                        end_reason = END_IDLE
                    else:
                        end_reason = END_WALL
                    break
                if item is None:
                    end_reason = END_PIPE
                    break
                if item.kind == "cont":
                    # More decode is scheduled.
                    awaiting_continue = True
                    if admit_hold:
                        # Hold on the progress-aware wall only — no short idle.
                        continue_deadline = None
                    else:
                        cont_budget = (
                            continue_idle if continue_idle > 0 else post_token_idle
                        )
                        continue_deadline = loop.time() + max(
                            cont_budget, post_token_idle
                        )
                        last_token_at = None
                    continue
                if item.kind == "done":
                    end_reason = END_DONE
                    break
                tok = item.value
                if tok < 0:
                    continue
                if tok in stops:
                    end_reason = END_STOP
                    break
                generated += 1
                awaiting_continue = False
                continue_deadline = None
                last_token_at = loop.time()
                # Progress-aware wall: reset the deadline on each *real* token so
                # a healthy (even slow) stream is never guillotined by an absolute
                # cap. CONTINUE heartbeats do NOT reset it, so a scheduled-but-
                # starved stream is still cancelled ``wall_timeout`` after its last
                # real token — guaranteeing termination and freeing the slot.
                deadline = last_token_at + wall_timeout
                yield tok
            else:
                end_reason = END_N_GEN
        finally:
            if outcome is not None:
                outcome.end_reason = end_reason
                outcome.generated = generated
                outcome.awaiting_continue = awaiting_continue
                outcome.admit_hold = admit_hold
            if own_reg:
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
