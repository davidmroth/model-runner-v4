"""Unit tests: Phase 1c inline tool KV pin helpers."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from handler_reliability import tool_inline_snap_pin_enabled
from tool_split.daemon_bridge import (
    append_inline_snap,
    tool_snap_prep_from_pending,
)


class ToolInlineSnapPinTests(unittest.TestCase):
    def test_inline_pin_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(tool_inline_snap_pin_enabled())

    def test_inline_pin_disabled_env(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_INLINE_SNAP_PIN": "0"}):
            self.assertFalse(tool_inline_snap_pin_enabled())

    def test_tool_snap_prep_from_pending(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_INLINE_SNAP_PIN": "1"}):
            self.assertEqual(tool_snap_prep_from_pending((4, 20590)), (4, 20590))

    def test_tool_snap_prep_defers_to_snapshot_thin_below_max(self):
        with patch.dict(
            os.environ,
            {"DFLASH_TOOL_INLINE_SNAP_PIN": "1", "DFLASH_TOOL_SNAPSHOT_MAX_KV": "16384"},
        ):
            self.assertIsNone(tool_snap_prep_from_pending((4, 310)))

    def test_tool_snap_prep_disabled(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_INLINE_SNAP_PIN": "0"}):
            self.assertIsNone(tool_snap_prep_from_pending((4, 20590)))

    def test_tool_snap_prep_zero_depth(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_INLINE_SNAP_PIN": "1"}):
            self.assertIsNone(tool_snap_prep_from_pending((4, 0)))

    def test_append_inline_snap(self):
        self.assertEqual(
            append_inline_snap("/tmp/p.bin 64", (4, 20590)),
            "/tmp/p.bin 64 snap=20590:4",
        )
        self.assertEqual(append_inline_snap("/tmp/p.bin 64", None), "/tmp/p.bin 64")


class FinishToolInlineSnapTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirm_on_matching_ack(self):
        from tool_split.daemon_bridge import finish_tool_inline_snap
        from tool_split.orchestrator import ToolSlotCache, ToolSplitConfig, ToolSplitOrchestrator
        from tool_split.base import ToolSplitAdapter, PromptSplit

        class _Adapter(ToolSplitAdapter):
            profile_name = "test"

            @classmethod
            def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
                return True

            def split_prompt(self, tokenizer, messages, tools, **kwargs) -> PromptSplit:
                return PromptSplit(
                    full_ids=[1, 2],
                    tool_prefix_ids=[1],
                    conversation_ids=[2],
                    tool_prefix_len=1,
                    profile="test",
                )

        orch = ToolSplitOrchestrator(
            adapter=_Adapter(),
            config=ToolSplitConfig(enabled=True, profile="test", pinned_tool_slots=2),
        )
        orch.tool_slots = ToolSlotCache(pinned_slots=2, slot_base=4)
        fp = "abc"
        orch.tool_slots.reserve(fp)

        bus = MagicMock()
        bus.drain_inline_snap = AsyncMock()
        bus.inline_snap_slot = MagicMock(return_value=4)

        ok = await finish_tool_inline_snap(
            orchestrator=orch,
            bus=bus,
            fingerprint=fp,
            tool_snap_prep=(4, 20590),
        )
        self.assertTrue(ok)
        self.assertEqual(orch.tool_slots.pinned_slot(fp), 4)

    async def test_release_on_ack_mismatch(self):
        from tool_split.daemon_bridge import finish_tool_inline_snap
        from tool_split.orchestrator import ToolSlotCache, ToolSplitConfig, ToolSplitOrchestrator
        from tool_split.base import ToolSplitAdapter, PromptSplit

        class _Adapter(ToolSplitAdapter):
            profile_name = "test"

            @classmethod
            def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
                return True

            def split_prompt(self, tokenizer, messages, tools, **kwargs) -> PromptSplit:
                return PromptSplit(
                    full_ids=[1, 2],
                    tool_prefix_ids=[1],
                    conversation_ids=[2],
                    tool_prefix_len=1,
                    profile="test",
                )

        orch = ToolSplitOrchestrator(
            adapter=_Adapter(),
            config=ToolSplitConfig(enabled=True, profile="test", pinned_tool_slots=2),
        )
        orch.tool_slots = ToolSlotCache(pinned_slots=2, slot_base=4)
        fp = "abc"
        orch.tool_slots.reserve(fp)

        bus = MagicMock()
        bus.drain_inline_snap = AsyncMock()
        bus.inline_snap_slot = MagicMock(return_value=None)

        ok = await finish_tool_inline_snap(
            orchestrator=orch,
            bus=bus,
            fingerprint=fp,
            tool_snap_prep=(4, 20590),
        )
        self.assertFalse(ok)
        self.assertIsNone(orch.tool_slots.pinned_slot(fp))


if __name__ == "__main__":
    unittest.main()
