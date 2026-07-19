"""Progress-aware wall behavior for the token collectors.

The request wall must cap the gap *between real tokens*, not total request
time. A healthy (even slow) generation that keeps producing tokens must run to
completion; a stream that produces no real token for ``wall_timeout`` seconds —
even one kept alive by CONTINUE heartbeats — must still be cancelled so the slot
is freed. These repros exercise both the tagged-demux path (production) and the
legacy pipe path.
"""
from __future__ import annotations

import asyncio
import os
import struct
import threading
import time
import unittest

from daemon_pipe import collect_pipe_tokens
from tagged_stream_demux import (
    STREAM_CONTINUE_SENTINEL,
    STREAM_DONE_SENTINEL,
    StreamFrame,
    TaggedStreamDemux,
)


def _demux() -> TaggedStreamDemux:
    r, w = os.pipe()
    os.close(w)
    return TaggedStreamDemux(r)


class TaggedDemuxProgressWallTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_stream_survives_past_wall_timeout(self) -> None:
        """Tokens spread over > wall_timeout, each within it, all delivered."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)
        wall = 0.3
        n = 8

        async def producer() -> None:
            for v in range(1, n + 1):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
                await asyncio.sleep(0.08)  # < wall, but n*gap > wall
            await q.put(
                StreamFrame(kind="done", value=STREAM_DONE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        started = asyncio.get_running_loop().time()
        got: list[int] = []
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=wall,
            post_token_idle=1.0,
            queue=q,
        ):
            got.append(t)
        elapsed = asyncio.get_running_loop().time() - started
        await task
        await demux.unregister(req)
        # All tokens delivered even though total time (~0.64s) exceeds the wall.
        self.assertEqual(got, list(range(1, n + 1)))
        self.assertGreater(elapsed, wall)

    async def test_heartbeating_stall_cancelled_at_wall_since_last_token(self) -> None:
        """CONTINUE heartbeats without real tokens still hit the wall."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)
        wall = 0.4

        async def producer() -> None:
            for v in (1, 2, 3):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            # Heartbeat CONTINUE frequently (< continue_idle) but never send a
            # real token — must NOT keep the stream alive past the wall.
            for _ in range(40):
                await q.put(
                    StreamFrame(
                        kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req
                    )
                )
                await asyncio.sleep(0.05)

        task = asyncio.create_task(producer())
        started = asyncio.get_running_loop().time()
        got: list[int] = []
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=wall,
            post_token_idle=5.0,
            continue_idle=5.0,
            queue=q,
        ):
            got.append(t)
        elapsed = asyncio.get_running_loop().time() - started
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3])
        # Cancelled ~wall after the last real token, not hanging on heartbeats.
        self.assertLess(elapsed, wall + 0.5)

    async def test_new_token_after_gap_resets_the_wall(self) -> None:
        """A real token pushes the deadline forward for the next quantum."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)
        wall = 0.3

        async def producer() -> None:
            await q.put(StreamFrame(kind="tag", value=1, req_id=req))
            await asyncio.sleep(0.2)  # < wall since token 1
            await q.put(StreamFrame(kind="tag", value=2, req_id=req))
            await asyncio.sleep(0.2)  # < wall since token 2 (but > wall since t1)
            await q.put(StreamFrame(kind="tag", value=3, req_id=req))
            await q.put(
                StreamFrame(kind="done", value=STREAM_DONE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        got: list[int] = []
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=wall,
            post_token_idle=1.0,
            queue=q,
        ):
            got.append(t)
        await task
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3])


class LegacyPipeProgressWallTests(unittest.TestCase):
    def test_slow_stream_survives_past_wall_timeout(self) -> None:
        r, w = os.pipe()
        wall = 0.3
        n = 6

        def writer() -> None:
            for tok in range(1, n + 1):
                os.write(w, struct.pack("<i", tok))
                time.sleep(0.08)  # < wall, but n*gap > wall
            os.close(w)

        threading.Thread(target=writer, daemon=True).start()
        started = time.monotonic()
        try:
            tokens = collect_pipe_tokens(
                r, n, wall_timeout=wall, post_token_idle=1.0,
            )
        finally:
            os.close(r)
        elapsed = time.monotonic() - started
        self.assertEqual(tokens, list(range(1, n + 1)))
        self.assertGreater(elapsed, wall)

    def test_no_first_token_still_bounded_by_wall(self) -> None:
        r, w = os.pipe()
        wall = 0.3

        def writer() -> None:
            time.sleep(wall + 0.5)  # never produce a token before the wall
            os.close(w)

        threading.Thread(target=writer, daemon=True).start()
        started = time.monotonic()
        try:
            tokens = collect_pipe_tokens(
                r, 8, wall_timeout=wall, post_token_idle=1.0,
            )
        finally:
            os.close(r)
        elapsed = time.monotonic() - started
        self.assertEqual(tokens, [])
        self.assertLess(elapsed, wall + 0.4)


if __name__ == "__main__":
    unittest.main()
