"""Unit tests for P0b handler reliability helpers in server_tools."""
import asyncio
import os
import unittest
from unittest.mock import patch

from handler_reliability import (
    DaemonBusyError,
    PriorityDaemonLock,
    SlowLaneBumpRegistry,
    chat_stream_lock_wait_seconds,
    daemon_lock_wait_seconds,
    ephemeral_lock_wait_seconds,
    is_ephemeral_cache_scope,
    install_quiet_access_log_filter,
    quiet_access_logs_enabled,
    request_hard_ceiling_seconds,
    request_wall_timeout_seconds,
    scoped_lock_priority_enabled,
    scoped_lock_wait_cap_seconds,
    should_log_ephemeral_busy,
    slow_lane_lock_wait_seconds,
    tool_inline_snap_pin_enabled,
    tool_snapshot_max_kv_tokens,
)


class HandlerReliabilityConfigTests(unittest.TestCase):
    def test_daemon_lock_wait_defaults_to_wall_timeout(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                daemon_lock_wait_seconds(),
                request_wall_timeout_seconds(),
            )

    def test_daemon_lock_wait_env_override(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "45"}):
            self.assertEqual(daemon_lock_wait_seconds(), 45.0)

    def test_daemon_lock_wait_zero_means_unbounded(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "0"}):
            self.assertEqual(daemon_lock_wait_seconds(), float("inf"))

    def test_daemon_lock_wait_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "nope"}):
            self.assertEqual(
                daemon_lock_wait_seconds(),
                request_wall_timeout_seconds(),
            )

    def test_request_wall_timeout_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(request_wall_timeout_seconds(), 600.0)

    def test_request_wall_timeout_env_override(self):
        with patch.dict(os.environ, {"DFLASH_REQUEST_WALL_TIMEOUT_SEC": "180"}):
            self.assertEqual(request_wall_timeout_seconds(), 180.0)

    def test_request_hard_ceiling_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(request_hard_ceiling_seconds())

    def test_request_hard_ceiling_zero_disables(self):
        with patch.dict(os.environ, {"DFLASH_REQUEST_HARD_CEILING_SEC": "0"}):
            self.assertIsNone(request_hard_ceiling_seconds())

    def test_request_hard_ceiling_env_override(self):
        with patch.dict(os.environ, {"DFLASH_REQUEST_HARD_CEILING_SEC": "3600"}):
            self.assertEqual(request_hard_ceiling_seconds(), 3600.0)

    def test_request_hard_ceiling_invalid_disables(self):
        with patch.dict(os.environ, {"DFLASH_REQUEST_HARD_CEILING_SEC": "nope"}):
            self.assertIsNone(request_hard_ceiling_seconds())

    def test_tool_snapshot_max_kv_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tool_snapshot_max_kv_tokens(), 16384)

    def test_tool_snapshot_max_kv_zero_means_unlimited(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_SNAPSHOT_MAX_KV": "0"}):
            self.assertEqual(tool_snapshot_max_kv_tokens(), 0)

    def test_tool_inline_snap_pin_defaults_on(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(tool_inline_snap_pin_enabled())

    def test_deferred_conv_snap_max_tail_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            from handler_reliability import deferred_conv_snap_max_tail

            self.assertEqual(deferred_conv_snap_max_tail(), 8192)

    def test_deferred_conv_snap_max_tail_env_override(self):
        with patch.dict(os.environ, {"DFLASH_DEFERRED_CONV_SNAP_MAX_TAIL": "512"}):
            from handler_reliability import deferred_conv_snap_max_tail

            self.assertEqual(deferred_conv_snap_max_tail(), 512)

    def test_tool_inline_snap_pin_off(self):
        with patch.dict(os.environ, {"DFLASH_TOOL_INLINE_SNAP_PIN": "0"}):
            self.assertFalse(tool_inline_snap_pin_enabled())

    def test_quiet_access_logs_defaults_on(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(quiet_access_logs_enabled())

    def test_quiet_access_logs_off(self):
        with patch.dict(os.environ, {"DFLASH_QUIET_ACCESS_LOGS": "0"}):
            self.assertFalse(quiet_access_logs_enabled())

    def test_quiet_access_log_filter_hides_health_and_models(self):
        import logging

        with patch.dict(os.environ, {"DFLASH_QUIET_ACCESS_LOGS": "1"}):
            install_quiet_access_log_filter()
            logger = logging.getLogger("uvicorn.access")
            filt = next(f for f in logger.filters if f.__class__.__name__ == "_QuietAccessLogFilter")
            health = logging.LogRecord(
                "uvicorn.access", logging.INFO, "", 0,
                '127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK',
                (), None,
            )
            models = logging.LogRecord(
                "uvicorn.access", logging.INFO, "", 0,
                '172.22.0.2:1 - "GET /v1/models HTTP/1.1" 200 OK',
                (), None,
            )
            chat = logging.LogRecord(
                "uvicorn.access", logging.INFO, "", 0,
                '172.22.0.3:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
                (), None,
            )
            self.assertFalse(filt.filter(health))
            self.assertFalse(filt.filter(models))
            self.assertTrue(filt.filter(chat))

    def test_quiet_access_log_filter_off_shows_all(self):
        import logging

        with patch.dict(os.environ, {"DFLASH_QUIET_ACCESS_LOGS": "0"}):
            install_quiet_access_log_filter()
            logger = logging.getLogger("uvicorn.access")
            filt = next(f for f in logger.filters if f.__class__.__name__ == "_QuietAccessLogFilter")
            health = logging.LogRecord(
                "uvicorn.access", logging.INFO, "", 0,
                '127.0.0.1:1 - "GET /health HTTP/1.1" 200 OK',
                (), None,
            )
            self.assertTrue(filt.filter(health))

    def test_daemon_busy_error_carries_label(self):
        err = DaemonBusyError("chat-stream")
        self.assertEqual(err.label, "chat-stream")

    def test_is_ephemeral_cache_scope(self):
        self.assertTrue(is_ephemeral_cache_scope("ephemeral:abc123"))
        self.assertFalse(is_ephemeral_cache_scope("e078410b-cd79-4685-8b7b-8d760dc370e8"))

    def test_ephemeral_lock_wait_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ephemeral_lock_wait_seconds(), 5.0)

    def test_scoped_lock_wait_cap_when_unbounded(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "0"}):
            self.assertEqual(
                chat_stream_lock_wait_seconds(scoped=True),
                scoped_lock_wait_cap_seconds(),
            )

    def test_scoped_lock_wait_cap_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                scoped_lock_wait_cap_seconds(),
                request_wall_timeout_seconds(),
            )

    def test_scoped_lock_wait_cap_zero_means_wall(self):
        with patch.dict(
            os.environ,
            {
                "DFLASH_SCOPED_LOCK_WAIT_SEC": "0",
                "DFLASH_REQUEST_WALL_TIMEOUT_SEC": "600",
            },
        ):
            self.assertEqual(scoped_lock_wait_cap_seconds(), 600.0)

    def test_sse_keepalive_default(self):
        from handler_reliability import sse_keepalive_seconds

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sse_keepalive_seconds(), 15.0)

    def test_sse_keepalive_zero_disables(self):
        from handler_reliability import sse_keepalive_seconds

        with patch.dict(os.environ, {"DFLASH_SSE_KEEPALIVE_SEC": "0"}):
            self.assertEqual(sse_keepalive_seconds(), 0.0)

    def test_ephemeral_lock_wait_short_when_unbounded(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "0"}):
            self.assertEqual(chat_stream_lock_wait_seconds(scoped=False), 5.0)

    def test_slow_lane_lock_wait_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(slow_lane_lock_wait_seconds(), 30.0)
            self.assertEqual(
                chat_stream_lock_wait_seconds(scoped=False, lane="slow"),
                30.0,
            )

    def test_scoped_lock_wait_respects_explicit_cap(self):
        with patch.dict(os.environ, {
            "DFLASH_DAEMON_LOCK_WAIT_SEC": "120",
            "DFLASH_SCOPED_LOCK_WAIT_SEC": "45",
        }):
            self.assertEqual(chat_stream_lock_wait_seconds(scoped=True), 45.0)

    def test_scoped_lock_priority_defaults_on(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(scoped_lock_priority_enabled())

    def test_ephemeral_busy_log_debounce(self):
        import handler_reliability as hr

        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_LOG_DEBOUNCE_SEC": "60"}):
            hr._last_ephemeral_log = 0.0
            self.assertTrue(should_log_ephemeral_busy())
            self.assertFalse(should_log_ephemeral_busy())
            hr._last_ephemeral_log = 0.0
            self.assertTrue(should_log_ephemeral_busy())


class PriorityDaemonLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_v1_runs_before_queued_v1e(self):
        """L0: /v1 runs first; /v1e stays queued (no cancel → no Hermes 503)."""
        lock = PriorityDaemonLock()
        order: list[str] = []

        async def slow_waiter():
            await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
            order.append("slow")
            lock.release()

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            order.append("priority")
            lock.release()

        lock._held = True
        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.02)
        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.02)
        self.assertEqual(lock.scoped_waiting, 1)
        self.assertEqual(len(lock._low), 1)
        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)
        self.assertEqual(order[0], "priority")
        self.assertIn("slow", order)

    async def test_unscoped_v1_waits_with_scoped_on_high(self):
        """Unscoped /v1 also joins high and can wait (no instant fail vs scoped)."""
        lock = PriorityDaemonLock()
        lock._held = True

        async def scoped_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            lock.release()

        t_user = asyncio.create_task(scoped_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(lock.scoped_waiting, 1)

        loop = asyncio.get_running_loop()
        started = loop.time()
        with self.assertRaises(asyncio.TimeoutError):
            await lock.acquire(scoped=False, max_wait=0.25, lane="priority")
        elapsed = loop.time() - started
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 1.0)

        lock.release()
        await t_user

    async def test_v1e_stays_queued_when_v1_enqueues(self):
        lock = PriorityDaemonLock()
        lock._held = True

        async def slow_waiter():
            await lock.acquire(scoped=False, max_wait=10.0, lane="slow")
            lock.release()

        async def fast_waiter():
            await lock.acquire(scoped=False, max_wait=5.0, lane="priority")
            lock.release()

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 1)

        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.05)
        self.assertEqual(len(lock._low), 1, "slow waiter must remain queued")

        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)

    async def test_scoped_runs_before_later_unscoped_after_release(self):
        """After release, earlier high waiters are granted FIFO."""
        lock = PriorityDaemonLock()
        order: list[str] = []
        lock._held = True
        scoped_holds = asyncio.Event()
        release_scoped = asyncio.Event()

        async def scoped_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            order.append("scoped-acquired")
            scoped_holds.set()
            await release_scoped.wait()
            order.append("scoped-releasing")
            lock.release()

        async def unscoped_after_scoped():
            await scoped_holds.wait()
            await lock.acquire(scoped=False, max_wait=5.0, lane="priority")
            order.append("unscoped-acquired")
            lock.release()

        t_user = asyncio.create_task(scoped_waiter())
        await asyncio.sleep(0.01)
        t_other = asyncio.create_task(unscoped_after_scoped())
        await asyncio.sleep(0.01)

        lock.release()
        await asyncio.wait_for(scoped_holds.wait(), timeout=2.0)
        self.assertEqual(order, ["scoped-acquired"])
        release_scoped.set()
        await asyncio.wait_for(asyncio.gather(t_user, t_other), timeout=2.0)
        self.assertEqual(
            order,
            ["scoped-acquired", "scoped-releasing", "unscoped-acquired"],
        )

    async def test_fast_does_not_drain_queued_v1e_waiters(self):
        """Queued /v1e stay queued when /v1 arrives (release order still prefers /v1)."""
        lock = PriorityDaemonLock()
        lock._held = True

        async def slow_waiter(name: str):
            await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
            lock.release()

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            lock.release()

        t_e1 = asyncio.create_task(slow_waiter("e1"))
        t_e2 = asyncio.create_task(slow_waiter("e2"))
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 2)

        t_s = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 2, "slow waiters must not be drained")

        lock.release()
        await asyncio.wait_for(asyncio.gather(t_e1, t_e2, t_s), timeout=2.0)

    async def test_fast_not_blocked_by_queued_v1e(self):
        """/v1 must acquire before any previously-queued /v1e can run."""
        lock = PriorityDaemonLock()
        order: list[str] = []
        lock._held = True

        async def slow_waiter():
            await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
            order.append("slow")
            lock.release()

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            order.append("priority")
            lock.release()

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.01)

        self.assertEqual(len(lock._low), 1)

        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)
        self.assertEqual(order[0], "priority")
        self.assertIn("slow", order)

    def test_busy_retry_after_matches_ephemeral_wait(self):
        """_busy_response uses int(wait_sec) for Retry-After on ephemeral 503s."""
        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_LOCK_WAIT_SEC": "5"}):
            wait_sec = ephemeral_lock_wait_seconds()
            self.assertEqual(str(max(1, int(wait_sec))), "5")


class SlowLaneBumpTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_v1_leaves_waiting_v1e_queued(self):
        lock = PriorityDaemonLock()
        lock._held = True

        async def slow_waiter():
            await lock.acquire(scoped=False, max_wait=10.0, lane="slow")
            lock.release()

        async def fast_waiter():
            await lock.acquire(scoped=False, max_wait=5.0, lane="priority")
            lock.release()

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 1)

        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.05)
        self.assertEqual(len(lock._low), 1)
        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)

    async def test_bump_registry_notifies_inflight(self):
        reg = SlowLaneBumpRegistry()
        eid, ev = reg.register()
        self.assertEqual(reg.inflight, 1)
        self.assertFalse(ev.is_set())
        self.assertEqual(reg.bump_all(), 1)
        self.assertTrue(ev.is_set())
        reg.unregister(eid)
        self.assertEqual(reg.inflight, 0)


if __name__ == "__main__":
    unittest.main()
