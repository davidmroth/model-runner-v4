"""Unit repros for demux idle behavior across CONTINUE quanta.

Live N=2 signature: HTTP returns after first quantum (~8 toks) while
SCHED_DRAIN may still have remaining. ``post_token_idle`` is one truncation
path when next tokens arrive more than idle after CONTINUE.
"""
from __future__ import annotations

import asyncio
import os
import unittest

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


class DemuxIdleBetweenQuantaTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_then_tokens_within_idle_accepts_next_quantum(self) -> None:
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (10, 11, 12, 13):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req)
            )
            await asyncio.sleep(0.05)  # < post_token_idle
            for v in (20, 21, 22, 23, 24, 25):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="done", value=STREAM_DONE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        got: list[int] = []
        async for t in demux.iter_tokens(
            req, n_gen=64, wall_timeout=5.0, post_token_idle=0.25, queue=q,
        ):
            got.append(t)
        await task
        await demux.unregister(req)
        self.assertEqual(got, [10, 11, 12, 13, 20, 21, 22, 23, 24, 25])

    async def test_continue_then_gap_beyond_idle_still_accepts_next_quantum(self) -> None:
        """After CONTINUE, normal idle must not truncate — SCHED may be slow."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (10, 11, 12, 13):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req)
            )
            await asyncio.sleep(0.40)  # > post_token_idle, < continue_idle
            for v in (20, 21, 22, 23):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="done", value=STREAM_DONE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        got: list[int] = []
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=5.0,
            post_token_idle=0.15,
            continue_idle=1.0,
            queue=q,
        ):
            got.append(t)
        await task
        await demux.unregister(req)
        self.assertEqual(got, [10, 11, 12, 13, 20, 21, 22, 23])

    async def test_continue_without_followup_stops_after_continue_idle(self) -> None:
        """CONTINUE must not hang until wall_timeout when SCHED never returns."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (1, 2, 3, 4):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req)
            )
            # No further tokens/DONE.

        task = asyncio.create_task(producer())
        started = asyncio.get_running_loop().time()
        got: list[int] = []
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=5.0,
            post_token_idle=0.5,
            continue_idle=0.2,
            queue=q,
        ):
            got.append(t)
        elapsed = asyncio.get_running_loop().time() - started
        await task
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3, 4])
        self.assertLess(elapsed, 1.5)

    async def test_idle_without_continue_stops_after_first_quantum(self) -> None:
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (1, 2, 3, 4, 5, 6, 7, 8):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))

        task = asyncio.create_task(producer())
        got: list[int] = []
        async for t in demux.iter_tokens(
            req, n_gen=64, wall_timeout=5.0, post_token_idle=0.15, queue=q,
        ):
            got.append(t)
        await task
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3, 4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
