"""Unit tests for P0b handler reliability helpers in server_tools."""
import asyncio
import os
import unittest
from unittest.mock import patch

from handler_reliability import (
    DaemonBusyError,
    PriorityDaemonLock,
    chat_stream_lock_wait_seconds,
    daemon_lock_wait_seconds,
    ephemeral_lock_wait_seconds,
    is_ephemeral_cache_scope,
    install_quiet_access_log_filter,
    quiet_access_logs_enabled,
    request_wall_timeout_seconds,
    scoped_lock_priority_enabled,
    scoped_lock_wait_cap_seconds,
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

    def test_ephemeral_lock_wait_short_when_unbounded(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "0"}):
            self.assertEqual(chat_stream_lock_wait_seconds(scoped=False), 5.0)

    def test_scoped_lock_wait_respects_explicit_cap(self):
        with patch.dict(os.environ, {
            "DFLASH_DAEMON_LOCK_WAIT_SEC": "120",
            "DFLASH_SCOPED_LOCK_WAIT_SEC": "45",
        }):
            self.assertEqual(chat_stream_lock_wait_seconds(scoped=True), 45.0)

    def test_scoped_lock_priority_defaults_on(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(scoped_lock_priority_enabled())


class PriorityDaemonLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_jumps_ahead_of_ephemeral_waiter(self):
        lock = PriorityDaemonLock()
        order: list[str] = []

        async def ephemeral_waiter():
            await lock.acquire(scoped=False, max_wait=5.0)
            order.append("bench")
            lock.release()

        async def scoped_waiter():
            await lock.acquire(scoped=True, max_wait=5.0)
            order.append("user")
            lock.release()

        lock._held = True
        t_bench = asyncio.create_task(ephemeral_waiter())
        t_user = asyncio.create_task(scoped_waiter())
        await asyncio.sleep(0.05)
        self.assertEqual(lock.scoped_waiting, 1)
        lock.release()
        await asyncio.wait_for(asyncio.gather(t_bench, t_user), timeout=2.0)
        self.assertEqual(order, ["user", "bench"])

    async def test_ephemeral_yields_when_scoped_queued(self):
        lock = PriorityDaemonLock()
        lock._held = True

        async def scoped_waiter():
            await lock.acquire(scoped=True, max_wait=5.0)
            lock.release()

        t_user = asyncio.create_task(scoped_waiter())
        await asyncio.sleep(0.01)
        with self.assertRaises(DaemonBusyError):
            await lock.acquire(scoped=False, max_wait=1.0)
        lock.release()
        await t_user


if __name__ == "__main__":
    unittest.main()
