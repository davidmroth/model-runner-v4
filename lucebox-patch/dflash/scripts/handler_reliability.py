"""Handler reliability helpers for server_tools (no heavy runtime deps)."""
from __future__ import annotations

import os


class DaemonBusyError(Exception):
    """Raised when the single-flight daemon lock cannot be acquired in time."""

    def __init__(self, label: str):
        self.label = label
        super().__init__(label)


def daemon_lock_wait_seconds() -> float:
    """Max seconds to wait for the single-flight daemon lock.

    Defaults to ``DFLASH_REQUEST_WALL_TIMEOUT_SEC`` (same bound as inference).
    Set ``DFLASH_DAEMON_LOCK_WAIT_SEC=0`` for an unbounded queue (no 503 on busy).
    """
    raw = os.environ.get("DFLASH_DAEMON_LOCK_WAIT_SEC")
    if raw is None:
        return request_wall_timeout_seconds()
    try:
        val = float(raw)
    except ValueError:
        return request_wall_timeout_seconds()
    if val <= 0:
        return float("inf")
    return max(1.0, val)


def tool_snapshot_max_kv_tokens() -> int:
    """Skip SNAPSHOT_THIN above this KV depth (daemon may crash on huge thin snaps)."""
    raw = os.environ.get("DFLASH_TOOL_SNAPSHOT_MAX_KV", "16384")
    try:
        return max(0, int(raw))
    except ValueError:
        return 16384


def tool_inline_snap_pin_enabled() -> bool:
    """Pin tool KV via inline ``snap=`` on cold prefill (Phase 1c; avoids SNAPSHOT_THIN)."""
    raw = os.environ.get("DFLASH_TOOL_INLINE_SNAP_PIN", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def request_wall_timeout_seconds() -> float:
    raw = os.environ.get("DFLASH_REQUEST_WALL_TIMEOUT_SEC", "600")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0
