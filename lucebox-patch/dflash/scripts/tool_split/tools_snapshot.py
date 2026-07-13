"""Persist / reload the last successful tools fingerprint for startup warmup.

The token prefix is rebuilt at warmup time via the same ``split_prompt`` path
as live traffic; we only store the OpenAI-shaped ``tools`` JSON + fingerprint.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def tool_warmup_enabled() -> bool:
    """``DFLASH_TOOL_WARMUP`` — default on."""
    raw = os.environ.get("DFLASH_TOOL_WARMUP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def tool_pin_protect_enabled() -> bool:
    """``DFLASH_TOOL_PIN_PROTECT`` — ephemeral traffic cannot evict protected pins."""
    raw = os.environ.get("DFLASH_TOOL_PIN_PROTECT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def default_tools_snapshot_path() -> Path:
    raw = os.environ.get("DFLASH_TOOL_WARMUP_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    # Prefer models volume (rw in compose); fall back to /tmp.
    preferred = Path("/opt/lucebox-hub/server/models/.dflash/tool-warmup.json")
    if preferred.parent.parent.is_dir():
        return preferred
    return Path("/tmp/dflash-tool-warmup.json")


@dataclass(frozen=True)
class ToolsSnapshot:
    fingerprint: str
    tools: list[dict[str, Any]]
    tool_prefix_len: int | None = None
    saved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "tools": self.tools,
            "tool_prefix_len": self.tool_prefix_len,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolsSnapshot | None":
        fp = data.get("fingerprint")
        tools = data.get("tools")
        if not isinstance(fp, str) or not fp:
            return None
        if not isinstance(tools, list) or not tools:
            return None
        prefix_len = data.get("tool_prefix_len")
        if prefix_len is not None:
            try:
                prefix_len = int(prefix_len)
            except (TypeError, ValueError):
                prefix_len = None
        saved_at = data.get("saved_at")
        if saved_at is not None and not isinstance(saved_at, str):
            saved_at = None
        return cls(
            fingerprint=fp,
            tools=tools,
            tool_prefix_len=prefix_len,
            saved_at=saved_at,
        )


def load_tools_snapshot(path: Path | None = None) -> ToolsSnapshot | None:
    snap_path = path or default_tools_snapshot_path()
    try:
        raw = snap_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"[tool-split] warmup load failed path={snap_path}: {exc}", flush=True)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[tool-split] warmup corrupt JSON path={snap_path}: {exc}", flush=True)
        return None
    if not isinstance(data, dict):
        return None
    return ToolsSnapshot.from_dict(data)


def save_tools_snapshot(
    fingerprint: str,
    tools: Sequence[Any],
    *,
    tool_prefix_len: int | None = None,
    path: Path | None = None,
) -> Path | None:
    """Atomically write tools snapshot. Returns path on success, else None."""
    if not fingerprint or not tools:
        return None
    snap_path = path or default_tools_snapshot_path()
    payload = ToolsSnapshot(
        fingerprint=fingerprint,
        tools=[t if isinstance(t, dict) else dict(t) for t in tools],
        tool_prefix_len=tool_prefix_len,
        saved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ).to_dict()
    try:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".tool-warmup-",
            suffix=".json",
            dir=str(snap_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
            os.replace(tmp_name, snap_path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    except OSError as exc:
        print(f"[tool-split] warmup save failed path={snap_path}: {exc}", flush=True)
        return None
    print(
        f"[tool-split] tools snapshot saved fp={fingerprint[:12]}… "
        f"n_tools={len(payload['tools'])} path={snap_path}",
        flush=True,
    )
    return snap_path
