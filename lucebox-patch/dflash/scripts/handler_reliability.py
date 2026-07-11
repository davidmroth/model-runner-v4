"""Handler reliability helpers for server_tools (no heavy runtime deps)."""
from __future__ import annotations

import os


class DaemonBusyError(Exception):
    """Raised when the single-flight daemon lock cannot be acquired in time."""

    def __init__(self, label: str):
        self.label = label
        super().__init__(label)


def daemon_lock_wait_seconds() -> float:
    raw = os.environ.get("DFLASH_DAEMON_LOCK_WAIT_SEC", "120")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 120.0


def request_wall_timeout_seconds() -> float:
    raw = os.environ.get("DFLASH_REQUEST_WALL_TIMEOUT_SEC", "600")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0
