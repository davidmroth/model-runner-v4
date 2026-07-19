#!/usr/bin/env python3
"""Unit tests for pflash-off prefill→prefix slot reclaim."""
from __future__ import annotations

import unittest
from pathlib import Path


def _load_reclaim():
    src = (Path(__file__).resolve().parent / "server_tools.py").read_text(
        encoding="utf-8"
    )
    start = src.index("def reclaim_prefill_slots_when_pflash_off")
    end = src.index("\ndef main():", start)
    ns: dict = {}
    exec(compile(src[start:end], "server_tools.py", "exec"), ns)
    return ns["reclaim_prefill_slots_when_pflash_off"]


reclaim_prefill_slots_when_pflash_off = _load_reclaim()


class TestReclaimPrefillSlots(unittest.TestCase):
    def test_reclaim_when_pflash_off(self):
        p, f, r = reclaim_prefill_slots_when_pflash_off(
            pflash_enabled=False, prefix_slots=2, prefill_slots=2
        )
        self.assertEqual((p, f, r), (4, 0, 2))

    def test_noop_when_pflash_on(self):
        p, f, r = reclaim_prefill_slots_when_pflash_off(
            pflash_enabled=True, prefix_slots=2, prefill_slots=2
        )
        self.assertEqual((p, f, r), (2, 2, 0))

    def test_noop_when_prefill_already_zero(self):
        p, f, r = reclaim_prefill_slots_when_pflash_off(
            pflash_enabled=False, prefix_slots=4, prefill_slots=0
        )
        self.assertEqual((p, f, r), (4, 0, 0))


if __name__ == "__main__":
    unittest.main()
