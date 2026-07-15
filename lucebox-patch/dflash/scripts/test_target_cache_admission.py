"""Unit tests for Phase 3 M3a target-cache slot admission."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from target_cache_admission import (
    TargetCacheSlotPool,
    append_restore_chain_quantum,
    format_slot_command,
    multi_slot_drop_exclusive,
    parse_restore_chain_admit_remaining,
    set_active_live_slot,
    stream_tagged_enabled,
    target_cache_slots,
)
from tool_split.daemon_bridge import snapshot_thin
from tool_split.orchestrator import ToolSplitOrchestrator, ToolSplitPlan


class ConfigHelpersTests(unittest.TestCase):
    def test_target_cache_slots_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DFLASH_TARGET_CACHE_SLOTS", None)
            self.assertEqual(target_cache_slots(), 1)

    def test_target_cache_slots_clamp(self) -> None:
        with patch.dict(os.environ, {"DFLASH_TARGET_CACHE_SLOTS": "99"}):
            self.assertEqual(target_cache_slots(), 16)
        with patch.dict(os.environ, {"DFLASH_TARGET_CACHE_SLOTS": "0"}):
            self.assertEqual(target_cache_slots(), 1)

    def test_stream_tagged_and_drop_exclusive_flags(self) -> None:
        with patch.dict(os.environ, {"DFLASH_STREAM_TAGGED": "1"}):
            self.assertTrue(stream_tagged_enabled())
        with patch.dict(os.environ, {"DFLASH_MULTI_SLOT_DROP_EXCLUSIVE": "yes"}):
            self.assertTrue(multi_slot_drop_exclusive())
        with patch.dict(
            os.environ,
            {"DFLASH_STREAM_TAGGED": "0", "DFLASH_MULTI_SLOT_DROP_EXCLUSIVE": "0"},
        ):
            self.assertFalse(stream_tagged_enabled())
            self.assertFalse(multi_slot_drop_exclusive())


class FormatSlotCommandTests(unittest.TestCase):
    def test_n1_unchanged(self) -> None:
        line = "RESTORE_CHAIN -1 4 /tmp/p.bin 8\n"
        self.assertEqual(format_slot_command(line, slot=0, slots=1), line)

    def test_n2_prefixes_slot(self) -> None:
        line = "RESTORE_CHAIN -1 4 /tmp/p.bin 8\n"
        self.assertEqual(
            format_slot_command(line, slot=1, slots=2),
            "SLOT 1 RESTORE_CHAIN -1 4 /tmp/p.bin 8\n",
        )

    def test_uses_active_context_slot(self) -> None:
        tok = set_active_live_slot(0)
        try:
            out = format_slot_command("SNAPSHOT_THIN 4 0 10\n", slots=2)
            self.assertEqual(out, "SLOT 0 SNAPSHOT_THIN 4 0 10\n")
        finally:
            from target_cache_admission import reset_active_live_slot

            reset_active_live_slot(tok)

    def test_n2_without_slot_raises(self) -> None:
        with self.assertRaises(ValueError):
            format_slot_command("RESTORE 0 /tmp/p.bin 4\n", slots=2)

    def test_append_restore_chain_quantum(self) -> None:
        from target_cache_admission import append_restore_chain_quantum

        base = "RESTORE_CHAIN -1 4 /tmp/p.bin 12\n"
        self.assertEqual(
            append_restore_chain_quantum(base, quantum=4),
            "RESTORE_CHAIN -1 4 /tmp/p.bin 12 4\n",
        )
        already = "RESTORE_CHAIN -1 4 /tmp/p.bin 12 8\n"
        self.assertEqual(append_restore_chain_quantum(already, quantum=4), already)
        with_snap = "RESTORE_CHAIN -1 4 /tmp/p.bin 12 snap=10:0\n"
        self.assertEqual(
            append_restore_chain_quantum(with_snap, quantum=4),
            "RESTORE_CHAIN -1 4 /tmp/p.bin 12 4 snap=10:0\n",
        )
        slotted = "SLOT 1 RESTORE_CHAIN -1 4 /tmp/p.bin 12\n"
        self.assertEqual(
            append_restore_chain_quantum(slotted, quantum=4),
            "SLOT 1 RESTORE_CHAIN -1 4 /tmp/p.bin 12 4\n",
        )
        req_slot = "REQ 3 SLOT 0 RESTORE_CHAIN -1 4 /tmp/p.bin 12 snap=10:0\n"
        self.assertEqual(
            append_restore_chain_quantum(req_slot, quantum=8),
            "REQ 3 SLOT 0 RESTORE_CHAIN -1 4 /tmp/p.bin 12 8 snap=10:0\n",
        )
        from target_cache_admission import is_restore_chain_command

        self.assertTrue(is_restore_chain_command(slotted))
        self.assertTrue(is_restore_chain_command(req_slot))
        self.assertFalse(is_restore_chain_command("SLOT 0 /tmp/p.bin 8\n"))
        self.assertEqual(
            parse_restore_chain_admit_remaining(
                "ok RESTORE_CHAIN_ADMIT req=1 slot=0 emitted=8 remaining=0 "
                "total_gen=64000 quantum=8"
            ),
            0,
        )
        self.assertEqual(
            parse_restore_chain_admit_remaining(
                "ok RESTORE_CHAIN_ADMIT req=2 slot=1 emitted=4 remaining=20 "
                "total_gen=24 quantum=4"
            ),
            20,
        )
        self.assertIsNone(parse_restore_chain_admit_remaining("ok RESTORE_CHAIN"))
        plan = ToolSplitPlan(
            prompt_bin_path="/tmp/p.bin",
            prompt_token_count=10,
            tool_slot=4,
            conv_restore_slot=-1,
            conv_restore_prefix_len=0,
            use_restore_chain=True,
            thin_slot_ids=[4],
            inline_snap=None,
            compression_fired=False,
            started_in_thinking=False,
            tools_fingerprint="fp",
            pending_tool_snap=None,
            tool_prefix_len=8,
        )
        # ToolSplitOrchestrator.format_daemon_command needs a real object — call helper path.
        orch = object.__new__(ToolSplitOrchestrator)
        with patch.dict(os.environ, {"DFLASH_TARGET_CACHE_SLOTS": "2"}):
            tok = set_active_live_slot(1)
            try:
                line = ToolSplitOrchestrator.format_daemon_command(orch, plan, 4)
                self.assertTrue(line.startswith("SLOT 1 RESTORE_CHAIN "), line)
            finally:
                from target_cache_admission import reset_active_live_slot

                reset_active_live_slot(tok)


class SlotPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_dual_admit_and_third_waits(self) -> None:
        pool = TargetCacheSlotPool(2)
        a = await pool.acquire("conv:a", scoped=True)
        b = await pool.acquire("conv:b", scoped=True)
        self.assertEqual({a.slot, b.slot}, {0, 1})

        third = asyncio.create_task(pool.acquire("conv:c", scoped=True, max_wait=0.05))
        with self.assertRaises(asyncio.TimeoutError):
            await third

        pool.release(a)
        c = await pool.acquire("conv:c", scoped=True, max_wait=1.0)
        self.assertEqual(c.slot, a.slot)
        pool.release(b)
        pool.release(c)

    async def test_sticky_affinity(self) -> None:
        pool = TargetCacheSlotPool(2)
        a1 = await pool.acquire("conv:sticky", scoped=True)
        other = await pool.acquire("conv:other", scoped=True)
        slot = a1.slot
        self.assertNotEqual(other.slot, slot)
        pool.release(a1)
        a2 = await pool.acquire("conv:sticky", scoped=True)
        self.assertEqual(a2.slot, slot)
        pool.release(other)
        pool.release(a2)

    async def test_ephemeral_fails_fast_when_scoped_waiting(self) -> None:
        pool = TargetCacheSlotPool(2)
        a = await pool.acquire("conv:a", scoped=True)
        b = await pool.acquire("conv:b", scoped=True)
        waiter = asyncio.create_task(
            pool.acquire("conv:c", scoped=True, max_wait=2.0)
        )
        await asyncio.sleep(0.05)  # let scoped join _high
        with self.assertRaises(asyncio.TimeoutError):
            await pool.acquire("ephemeral:bg", scoped=False, max_wait=1.0)
        pool.release(a)
        c = await waiter
        self.assertEqual(c.slot, a.slot)
        pool.release(b)
        pool.release(c)

    async def test_slow_lane_cannot_take_last_reserved_slot(self) -> None:
        pool = TargetCacheSlotPool(2)
        fast = await pool.acquire("conv:a", scoped=True, lane="priority")
        # One free remains, reserved=1 → slow must not grant.
        with self.assertRaises(asyncio.TimeoutError):
            await pool.acquire("ephemeral:slow", scoped=False, max_wait=0.05, lane="slow")
        # Fast may still take the reserved slot.
        fast2 = await pool.acquire("conv:b", scoped=True, max_wait=0.05, lane="priority")
        self.assertNotEqual(fast.slot, fast2.slot)
        pool.release(fast)
        # Now free=1 again (reserved) — still no slow.
        with self.assertRaises(asyncio.TimeoutError):
            await pool.acquire("ephemeral:slow2", scoped=False, max_wait=0.05, lane="slow")
        pool.release(fast2)
        # Both free → slow can take one, leave one reserved for fast.
        slow = await pool.acquire("ephemeral:ok", scoped=False, max_wait=0.05, lane="slow")
        with self.assertRaises(asyncio.TimeoutError):
            await pool.acquire("ephemeral:deny", scoped=False, max_wait=0.05, lane="slow")
        remaining_fast = await pool.acquire(
            "conv:c", scoped=True, max_wait=0.05, lane="priority",
        )
        pool.release(slow)
        pool.release(remaining_fast)

    async def test_fast_v1_drains_waiting_slow_slot_waiter(self) -> None:
        pool = TargetCacheSlotPool(1)
        held = await pool.acquire("conv:a", scoped=True, lane="priority")
        cancelled = asyncio.Event()

        async def slow_waiter():
            try:
                await pool.acquire("ephemeral:s", scoped=False, max_wait=5.0, lane="slow")
            except asyncio.CancelledError:
                cancelled.set()
                raise

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(pool._low), 1)
        t_fast = asyncio.create_task(
            pool.acquire("ephemeral:f", scoped=False, max_wait=5.0, lane="priority")
        )
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        self.assertEqual(len(pool._low), 0)
        pool.release(held)
        lease = await asyncio.wait_for(t_fast, timeout=1.0)
        pool.release(lease)
        with self.assertRaises(asyncio.CancelledError):
            await t_slow

    async def test_lease_context_sets_active_slot(self) -> None:
        pool = TargetCacheSlotPool(2)
        async with pool.lease("conv:x", scoped=True) as lease:
            self.assertEqual(format_slot_command("RESTORE 0 /t 2\n", slots=2),
                             f"SLOT {lease.slot} RESTORE 0 /t 2\n")


class SnapshotThinSlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_thin_prefixes_slot(self) -> None:
        written: list[str] = []

        class _Stdin:
            def write(self, data: bytes) -> None:
                written.append(data.decode("utf-8"))

            def flush(self) -> None:
                pass

        async def await_reply(prefix: str, timeout: float = 30.0) -> str:
            return "[snap] thin slot=4 kv=0,8"

        with patch.dict(os.environ, {"DFLASH_TARGET_CACHE_SLOTS": "2"}):
            tok = set_active_live_slot(0)
            try:
                ok = await snapshot_thin(
                    daemon_stdin=_Stdin(),
                    await_reply=await_reply,
                    slot=4,
                    kv_start=0,
                    kv_end=8,
                )
            finally:
                from target_cache_admission import reset_active_live_slot

                reset_active_live_slot(tok)
        self.assertTrue(ok)
        self.assertEqual(written, ["SLOT 0 SNAPSHOT_THIN 4 0 8\n"])


if __name__ == "__main__":
    unittest.main()
