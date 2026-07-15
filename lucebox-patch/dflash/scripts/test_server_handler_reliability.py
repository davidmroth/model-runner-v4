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
    request_wall_timeout_seconds,
    scoped_lock_priority_enabled,
    scoped_lock_wait_cap_seconds,
    should_log_ephemeral_busy,
    slow_lane_lock_wait_seconds,
    slow_lane_max_tokens,
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

    def test_ephemeral_max_tokens_defaults_and_clamp(self):
        from handler_reliability import ephemeral_max_tokens

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DFLASH_EPHEMERAL_MAX_TOKENS", None)
            self.assertEqual(ephemeral_max_tokens(), 2048)
        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_MAX_TOKENS": "512"}):
            self.assertEqual(ephemeral_max_tokens(), 512)
        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_MAX_TOKENS": "999999"}):
            self.assertEqual(ephemeral_max_tokens(), 65536)

    def test_scoped_lock_wait_cap_when_unbounded(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "0"}):
            self.assertEqual(
                chat_stream_lock_wait_seconds(scoped=True),
                scoped_lock_wait_cap_seconds(),
            )

    def test_scoped_lock_wait_cap_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scoped_lock_wait_cap_seconds(), 180.0)

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

    def test_slow_lane_max_tokens_defaults_to_ephemeral(self):
        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_MAX_TOKENS": "512"}, clear=True):
            self.assertEqual(slow_lane_max_tokens(), 512)
        with patch.dict(os.environ, {"DFLASH_SLOW_LANE_MAX_TOKENS": "256"}):
            self.assertEqual(slow_lane_max_tokens(), 256)

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
    async def test_fast_v1_cancels_waiting_v1e(self):
        """Any /v1 cancels queued /v1e, not merely deferring it."""
        lock = PriorityDaemonLock()
        order: list[str] = []

        async def slow_waiter():
            try:
                await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
                order.append("slow")
                lock.release()
            except (asyncio.CancelledError, DaemonBusyError):
                order.append("slow-cancelled")

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            order.append("priority")
            lock.release()

        lock._held = True
        t_slow = asyncio.create_task(slow_waiter())
        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.05)
        self.assertEqual(lock.scoped_waiting, 1)
        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)
        self.assertIn("priority", order)
        self.assertIn("slow-cancelled", order)
        self.assertNotIn("slow", order)

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

    async def test_v1e_cancelled_when_any_v1_enqueues_while_waiting(self):
        lock = PriorityDaemonLock()
        lock._held = True
        cancelled = asyncio.Event()

        async def slow_waiter():
            try:
                await lock.acquire(scoped=False, max_wait=10.0, lane="slow")
                lock.release()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def fast_waiter():
            await lock.acquire(scoped=False, max_wait=5.0, lane="priority")
            lock.release()

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 1)

        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        self.assertEqual(len(lock._low), 0)

        lock.release()
        with self.assertRaises(asyncio.CancelledError):
            await t_slow
        await t_fast

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

    async def test_fast_drains_queued_v1e_waiters(self):
        """/v1e already in _low must be cancelled when any /v1 enqueues."""
        lock = PriorityDaemonLock()
        lock._held = True
        cancelled: list[str] = []

        async def slow_waiter(name: str):
            try:
                await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
                lock.release()
            except (asyncio.CancelledError, DaemonBusyError):
                cancelled.append(name)

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            lock.release()

        t_e1 = asyncio.create_task(slow_waiter("e1"))
        t_e2 = asyncio.create_task(slow_waiter("e2"))
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 2)

        t_s = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 0, "slow waiters not drained")

        lock.release()
        await asyncio.wait_for(asyncio.gather(t_e1, t_e2, t_s), timeout=2.0)
        self.assertIn("e1", cancelled)
        self.assertIn("e2", cancelled)

    async def test_fast_not_blocked_by_queued_v1e(self):
        """/v1 must acquire before any previously-queued /v1e can run."""
        lock = PriorityDaemonLock()
        order: list[str] = []
        lock._held = True

        async def slow_waiter():
            try:
                await lock.acquire(scoped=False, max_wait=5.0, lane="slow")
                order.append("slow")
                lock.release()
            except (asyncio.CancelledError, DaemonBusyError):
                order.append("slow-cancelled")

        async def fast_waiter():
            await lock.acquire(scoped=True, max_wait=5.0, lane="priority")
            order.append("priority")
            lock.release()

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        t_fast = asyncio.create_task(fast_waiter())
        await asyncio.sleep(0.01)

        self.assertEqual(len(lock._low), 0)

        lock.release()
        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)
        self.assertIn("priority", order)
        self.assertNotIn("slow", order)

    def test_busy_retry_after_matches_ephemeral_wait(self):
        """_busy_response uses int(wait_sec) for Retry-After on ephemeral 503s."""
        with patch.dict(os.environ, {"DFLASH_EPHEMERAL_LOCK_WAIT_SEC": "5"}):
            wait_sec = ephemeral_lock_wait_seconds()
            self.assertEqual(str(max(1, int(wait_sec))), "5")


class SlowLaneBumpTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_v1_drains_waiting_v1e(self):
        lock = PriorityDaemonLock()
        lock._held = True
        cancelled = asyncio.Event()

        async def slow_waiter():
            try:
                await lock.acquire(scoped=False, max_wait=10.0, lane="slow")
                lock.release()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        t_slow = asyncio.create_task(slow_waiter())
        await asyncio.sleep(0.01)
        self.assertEqual(len(lock._low), 1)

        # Unscoped /v1 (also high) must drain /v1e waiters.
        t_fast = asyncio.create_task(
            lock.acquire(scoped=False, max_wait=5.0, lane="priority")
        )
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        self.assertEqual(len(lock._low), 0)
        lock.release()
        with self.assertRaises(asyncio.CancelledError):
            await t_slow
        await t_fast
        lock.release()

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
