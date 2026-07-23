"""Unit repros for demux idle behavior across CONTINUE quanta.

Live N=2 signature: HTTP returns after first quantum (~8 toks) while
SCHED_DRAIN may still have remaining. ``post_token_idle`` is one truncation
path when next tokens arrive more than idle after CONTINUE.

Admit-hold (Phase B): quantum-admit collects must ignore short
``continue_idle`` / ``post_token_idle`` and wait on the progress-aware wall.
"""
from __future__ import annotations

import asyncio
import os
import unittest

from tagged_stream_demux import (
    END_CONTINUE_IDLE,
    END_DONE,
    END_WALL,
    STREAM_CONTINUE_SENTINEL,
    STREAM_DONE_SENTINEL,
    StreamCollectOutcome,
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
        outcome = StreamCollectOutcome()
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=5.0,
            post_token_idle=0.5,
            continue_idle=0.2,
            outcome=outcome,
            queue=q,
        ):
            got.append(t)
        elapsed = asyncio.get_running_loop().time() - started
        await task
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3, 4])
        self.assertLess(elapsed, 1.5)
        self.assertEqual(outcome.end_reason, END_CONTINUE_IDLE)
        self.assertTrue(outcome.is_truncated)

    async def test_admit_hold_survives_gap_beyond_continue_idle(self) -> None:
        """Phase B: admitted streams wait past continue_idle for the next quantum."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (10, 11, 12, 13):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req)
            )
            # Far beyond legacy continue_idle (0.15), still under wall.
            await asyncio.sleep(0.55)
            for v in (20, 21, 22, 23):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="done", value=STREAM_DONE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        got: list[int] = []
        outcome = StreamCollectOutcome()
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=2.0,
            post_token_idle=0.1,
            continue_idle=0.15,
            admit_hold=True,
            outcome=outcome,
            queue=q,
        ):
            got.append(t)
        await task
        await demux.unregister(req)
        self.assertEqual(got, [10, 11, 12, 13, 20, 21, 22, 23])
        self.assertEqual(outcome.end_reason, END_DONE)
        self.assertFalse(outcome.is_truncated)

    async def test_admit_hold_without_followup_waits_until_wall(self) -> None:
        """Admit-hold: no short continue_idle exit — wall is the ceiling."""
        demux = _demux()
        req = demux.alloc_req_id()
        q = await demux.register(req)

        async def producer() -> None:
            for v in (1, 2, 3, 4):
                await q.put(StreamFrame(kind="tag", value=v, req_id=req))
            await q.put(
                StreamFrame(kind="cont", value=STREAM_CONTINUE_SENTINEL, req_id=req)
            )

        task = asyncio.create_task(producer())
        started = asyncio.get_running_loop().time()
        got: list[int] = []
        outcome = StreamCollectOutcome()
        wall = 0.45
        async for t in demux.iter_tokens(
            req,
            n_gen=64,
            wall_timeout=wall,
            post_token_idle=0.05,
            continue_idle=0.05,
            admit_hold=True,
            outcome=outcome,
            queue=q,
        ):
            got.append(t)
        elapsed = asyncio.get_running_loop().time() - started
        await task
        await demux.unregister(req)
        self.assertEqual(got, [1, 2, 3, 4])
        self.assertEqual(outcome.end_reason, END_WALL)
        self.assertTrue(outcome.awaiting_continue)
        self.assertTrue(outcome.is_truncated)
        self.assertGreaterEqual(elapsed, wall - 0.05)
        self.assertLess(elapsed, wall + 0.4)

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


class StreamCollectOutcomeTests(unittest.TestCase):
    def test_done_not_truncated(self) -> None:
        o = StreamCollectOutcome(end_reason=END_DONE, admit_hold=True)
        self.assertFalse(o.is_truncated)

    def test_continue_idle_truncated(self) -> None:
        o = StreamCollectOutcome(
            end_reason=END_CONTINUE_IDLE, awaiting_continue=True
        )
        self.assertTrue(o.is_truncated)

    def test_admit_hold_wall_with_remaining(self) -> None:
        o = StreamCollectOutcome(
            end_reason=END_WALL,
            admit_hold=True,
            admit_remaining=32640,
            awaiting_continue=True,
        )
        self.assertTrue(o.is_truncated)


if __name__ == "__main__":
    unittest.main()
