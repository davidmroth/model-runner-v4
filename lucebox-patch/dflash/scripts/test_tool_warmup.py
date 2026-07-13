"""Unit tests: tools snapshot persist + protected tool-slot retention."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_split.orchestrator import ToolSlotCache
from tool_split.tools_snapshot import (
    ToolsSnapshot,
    load_tools_snapshot,
    save_tools_snapshot,
    tool_pin_protect_enabled,
    tool_warmup_enabled,
)


class ToolsSnapshotTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tool-warmup.json"
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Run a command",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            out = save_tools_snapshot(
                "abc123fingerprint",
                tools,
                tool_prefix_len=20590,
                path=path,
            )
            self.assertEqual(out, path)
            snap = load_tools_snapshot(path)
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.fingerprint, "abc123fingerprint")
            self.assertEqual(snap.tool_prefix_len, 20590)
            self.assertEqual(snap.tools[0]["function"]["name"], "terminal")
            raw = json.loads(path.read_text())
            self.assertIn("saved_at", raw)

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(load_tools_snapshot(Path(td) / "nope.json"))

    def test_from_dict_rejects_bad(self):
        self.assertIsNone(ToolsSnapshot.from_dict({}))
        self.assertIsNone(ToolsSnapshot.from_dict({"fingerprint": "x", "tools": []}))

    def test_env_flags_default_on(self):
        with patch.dict(os.environ, {}, clear=True):
            # clear=True removes DFLASH_* ; defaults should be on
            os.environ.pop("DFLASH_TOOL_WARMUP", None)
            os.environ.pop("DFLASH_TOOL_PIN_PROTECT", None)
            self.assertTrue(tool_warmup_enabled())
            self.assertTrue(tool_pin_protect_enabled())
        with patch.dict(os.environ, {"DFLASH_TOOL_WARMUP": "0"}):
            self.assertFalse(tool_warmup_enabled())
        with patch.dict(os.environ, {"DFLASH_TOOL_PIN_PROTECT": "off"}):
            self.assertFalse(tool_pin_protect_enabled())


class ToolSlotProtectTests(unittest.TestCase):
    def test_ephemeral_cannot_evict_protected(self):
        cache = ToolSlotCache(pinned_slots=2, slot_base=4)
        s0 = cache.reserve("hot-a")
        self.assertEqual(s0, 4)
        cache.confirm("hot-a", 4, protect=True)
        s1 = cache.reserve("hot-b")
        self.assertEqual(s1, 5)
        cache.confirm("hot-b", 5, protect=True)

        blocked = cache.reserve("ephemeral-c", allow_evict_protected=False)
        self.assertIsNone(blocked)
        self.assertEqual(cache.pinned_slot("hot-a"), 4)
        self.assertEqual(cache.pinned_slot("hot-b"), 5)
        self.assertTrue(cache.is_protected("hot-a"))

    def test_scoped_can_evict_protected_lru(self):
        cache = ToolSlotCache(pinned_slots=1, slot_base=4)
        cache.reserve("old")
        cache.confirm("old", 4, protect=True)
        slot = cache.reserve("new", allow_evict_protected=True)
        self.assertEqual(slot, 4)
        cache.confirm("new", 4, protect=True)
        self.assertIsNone(cache.pinned_slot("old"))
        self.assertFalse(cache.is_protected("old"))
        self.assertEqual(cache.pinned_slot("new"), 4)
        self.assertTrue(cache.is_protected("new"))

    def test_unprotected_evicted_before_protected(self):
        cache = ToolSlotCache(pinned_slots=2, slot_base=4)
        cache.reserve("probe")
        cache.confirm("probe", 4, protect=False)
        cache.reserve("hot")
        cache.confirm("hot", 5, protect=True)
        # LRU is probe (unprotected); ephemeral may take its slot.
        slot = cache.reserve("other", allow_evict_protected=False)
        self.assertEqual(slot, 4)
        cache.confirm("other", 4, protect=False)
        self.assertIsNone(cache.pinned_slot("probe"))
        self.assertEqual(cache.pinned_slot("hot"), 5)


if __name__ == "__main__":
    unittest.main()
