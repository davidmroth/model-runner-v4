"""Tests for daemon pipe token readers (no live GPU/daemon)."""
import os
import struct
import threading
import time
import unittest

from daemon_pipe import (
    async_iter_pipe_tokens,
    collect_pipe_tokens,
    drain_pipe_residual,
    iter_pipe_tokens,
)


class DaemonPipeTests(unittest.TestCase):
    def test_iter_stops_at_n_gen_without_sentinel(self):
        r, w = os.pipe()

        def writer():
            for tok in (101, 102, 103, 999):
                os.write(w, struct.pack("<i", tok))
            os.close(w)

        threading.Thread(target=writer, daemon=True).start()
        try:
            tokens = collect_pipe_tokens(r, 3)
        finally:
            os.close(r)
        self.assertEqual(tokens, [101, 102, 103])

    def test_iter_honors_sentinel_before_n_gen(self):
        r, w = os.pipe()

        def writer():
            for tok in (10, 11, -1):
                os.write(w, struct.pack("<i", tok))
            os.close(w)

        threading.Thread(target=writer, daemon=True).start()
        try:
            tokens = collect_pipe_tokens(r, 10)
        finally:
            os.close(r)
        self.assertEqual(tokens, [10, 11])

    def test_iter_stops_when_bus_reports_completion(self):
        from unittest.mock import MagicMock

        r, w = os.pipe()

        def writer():
            for tok in (201, 202):
                os.write(w, struct.pack("<i", tok))
            # Daemon stops without closing pipe or sentinel (layer-split path).
            time.sleep(0.5)

        bus = MagicMock()
        bus.request_timings.side_effect = [
            {},
            {},
            {"completion_tokens": 2},
            {"completion_tokens": 2},
        ]

        threading.Thread(target=writer, daemon=True).start()
        try:
            tokens = collect_pipe_tokens(
                r, 4096, bus=bus, wall_timeout=5.0, post_token_idle=0.2,
            )
        finally:
            os.close(r)
            os.close(w)
        self.assertEqual(tokens, [201, 202])
        r, w = os.pipe()
        os.write(w, struct.pack("<iii", 1, 2, 3))
        drain_pipe_residual(r)
        try:
            import fcntl

            flags = fcntl.fcntl(r, fcntl.F_GETFL)
            fcntl.fcntl(r, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            with self.assertRaises(BlockingIOError):
                os.read(r, 4)
        finally:
            os.close(r)
            os.close(w)

    def test_async_iter_matches_sync(self):
        import asyncio

        r, w = os.pipe()

        def writer():
            for tok in (7, 8):
                os.write(w, struct.pack("<i", tok))
            os.close(w)

        threading.Thread(target=writer, daemon=True).start()

        async def collect():
            out = []
            async for tok in async_iter_pipe_tokens(r, 5):
                out.append(tok)
            return out

        try:
            tokens = asyncio.run(collect())
        finally:
            os.close(r)
        self.assertEqual(tokens, [7, 8])


if __name__ == "__main__":
    unittest.main()
