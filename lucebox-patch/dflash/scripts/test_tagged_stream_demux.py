"""Unit tests for Phase 3 M3b tagged stream demux."""
from __future__ import annotations

import asyncio
import os
import struct
import unittest

from tagged_stream_demux import (
    STREAM_CONTINUE_SENTINEL,
    STREAM_DONE_SENTINEL,
    TaggedFrameBuffer,
    TaggedStreamDemux,
    format_req_command,
    pack_bare_token,
    pack_tagged_frame,
)


class TaggedFrameBufferTests(unittest.TestCase):
    def test_tagged_token_done_cont(self) -> None:
        buf = TaggedFrameBuffer()
        blob = (
            pack_tagged_frame(1, 101)
            + pack_tagged_frame(1, STREAM_CONTINUE_SENTINEL)
            + pack_tagged_frame(2, 202)
            + pack_tagged_frame(2, STREAM_DONE_SENTINEL)
        )
        frames = buf.push(blob)
        self.assertEqual(
            [(f.kind, f.value, f.req_id) for f in frames],
            [
                ("tag", 101, 1),
                ("cont", STREAM_CONTINUE_SENTINEL, 1),
                ("tag", 202, 2),
                ("done", STREAM_DONE_SENTINEL, 2),
            ],
        )

    def test_partial_tagged_frame_waits(self) -> None:
        buf = TaggedFrameBuffer()
        partial = struct.pack("<ii", -2, 7)  # missing tok
        self.assertEqual(buf.push(partial), [])
        frames = buf.push(struct.pack("<i", 55))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].req_id, 7)
        self.assertEqual(frames[0].value, 55)

    def test_bare_legacy_frames(self) -> None:
        buf = TaggedFrameBuffer()
        frames = buf.push(pack_bare_token(9) + pack_bare_token(-1))
        self.assertEqual(frames[0].kind, "tok")
        self.assertEqual(frames[1].kind, "done")
        self.assertIsNone(frames[0].req_id)

    def test_format_req_command(self) -> None:
        self.assertEqual(
            format_req_command("SLOT 0 RESTORE_CHAIN -1 4 /p 8\n", 3),
            "REQ 3 SLOT 0 RESTORE_CHAIN -1 4 /p 8\n",
        )
        self.assertEqual(
            format_req_command("REQ 9 START /p 8 4\n", 3),
            "REQ 9 START /p 8 4\n",
        )


class TaggedStreamDemuxAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_two_req_ids(self) -> None:
        r, w = os.pipe()
        demux = TaggedStreamDemux(r)
        await demux.start()
        try:
            q1 = await demux.register(1)
            q2 = await demux.register(2)
            os.write(
                w,
                pack_tagged_frame(1, 11)
                + pack_tagged_frame(2, 22)
                + pack_tagged_frame(1, STREAM_DONE_SENTINEL)
                + pack_tagged_frame(2, STREAM_DONE_SENTINEL),
            )
            f1 = await asyncio.wait_for(q1.get(), timeout=2.0)
            f2 = await asyncio.wait_for(q2.get(), timeout=2.0)
            self.assertEqual((f1.kind, f1.value), ("tag", 11))
            self.assertEqual((f2.kind, f2.value), ("tag", 22))
            d1 = await asyncio.wait_for(q1.get(), timeout=2.0)
            d2 = await asyncio.wait_for(q2.get(), timeout=2.0)
            self.assertEqual(d1.kind, "done")
            self.assertEqual(d2.kind, "done")
        finally:
            await demux.stop()
            os.close(w)
            try:
                os.close(r)
            except OSError:
                pass

    async def test_iter_tokens_skips_continue(self) -> None:
        r, w = os.pipe()
        demux = TaggedStreamDemux(r)
        await demux.start()

        async def writer() -> None:
            await asyncio.sleep(0.05)
            os.write(
                w,
                pack_tagged_frame(5, 1)
                + pack_tagged_frame(5, STREAM_CONTINUE_SENTINEL)
                + pack_tagged_frame(5, 2)
                + pack_tagged_frame(5, STREAM_DONE_SENTINEL),
            )
            os.close(w)

        try:
            asyncio.create_task(writer())
            tokens = [
                t
                async for t in demux.iter_tokens(5, n_gen=8, wall_timeout=2.0)
            ]
            self.assertEqual(tokens, [1, 2])
        finally:
            await demux.stop()
            try:
                os.close(r)
            except OSError:
                pass

    async def test_iter_tokens_stops_on_stop_id(self) -> None:
        r, w = os.pipe()
        demux = TaggedStreamDemux(r)
        await demux.start()

        async def writer() -> None:
            await asyncio.sleep(0.05)
            os.write(
                w,
                pack_tagged_frame(7, 10)
                + pack_tagged_frame(7, 99)  # stop
                + pack_tagged_frame(7, 11)
                + pack_tagged_frame(7, STREAM_DONE_SENTINEL),
            )
            os.close(w)

        try:
            asyncio.create_task(writer())
            tokens = [
                t
                async for t in demux.iter_tokens(
                    7, n_gen=8, stop_ids={99}, wall_timeout=2.0
                )
            ]
            self.assertEqual(tokens, [10])
        finally:
            await demux.stop()
            try:
                os.close(r)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
