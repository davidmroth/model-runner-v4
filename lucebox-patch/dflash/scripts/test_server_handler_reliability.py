"""Unit tests for P0b handler reliability helpers in server_tools."""
import os
import unittest
from unittest.mock import patch

from handler_reliability import (
    DaemonBusyError,
    daemon_lock_wait_seconds,
    request_wall_timeout_seconds,
)


class HandlerReliabilityConfigTests(unittest.TestCase):
    def test_daemon_lock_wait_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(daemon_lock_wait_seconds(), 120.0)

    def test_daemon_lock_wait_env_override(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "45"}):
            self.assertEqual(daemon_lock_wait_seconds(), 45.0)

    def test_daemon_lock_wait_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"DFLASH_DAEMON_LOCK_WAIT_SEC": "nope"}):
            self.assertEqual(daemon_lock_wait_seconds(), 120.0)

    def test_request_wall_timeout_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(request_wall_timeout_seconds(), 600.0)

    def test_request_wall_timeout_env_override(self):
        with patch.dict(os.environ, {"DFLASH_REQUEST_WALL_TIMEOUT_SEC": "180"}):
            self.assertEqual(request_wall_timeout_seconds(), 180.0)

    def test_daemon_busy_error_carries_label(self):
        err = DaemonBusyError("chat-stream")
        self.assertEqual(err.label, "chat-stream")


if __name__ == "__main__":
    unittest.main()
