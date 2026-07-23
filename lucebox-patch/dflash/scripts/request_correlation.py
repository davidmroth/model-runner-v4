"""Request correlation logging for multi-slot cross-talk forensics.

Emits one-line handler events that join:
  conversation cache scope ↔ live target-cache slot ↔ daemon req_id
  ↔ restore/admit ↔ first decoded tokens ↔ CANCEL/SCHED

Enable/disable with ``DFLASH_CORR_LOG`` (default on).
"""
from __future__ import annotations

import os
import time
from typing import Any, Sequence


def corr_log_enabled() -> bool:
    raw = os.environ.get("DFLASH_CORR_LOG", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _short_scope(scope: str | None, *, max_len: int = 48) -> str:
    if not scope:
        return "-"
    s = str(scope)
    if len(s) <= max_len:
        return s
    # Prefer keeping conversation UUID prefix before the fingerprint suffix.
    if ":" in s:
        head, _, tail = s.partition(":")
        keep = max(8, max_len - len(tail) - 4)
        if len(head) > keep:
            head = head[:keep] + "…"
        return f"{head}:{tail}" if tail else head
    return s[: max_len - 1] + "…"


def format_corr(
    event: str,
    *,
    scope: str | None = None,
    slot: int | None = None,
    req_id: int | None = None,
    **fields: Any,
) -> str:
    """Build a single ``[corr]`` line for grep-friendly forensics."""
    parts = [f"  [corr] {event}"]
    if scope is not None:
        parts.append(f"scope={_short_scope(scope)!r}")
    if slot is not None:
        parts.append(f"slot={int(slot)}")
    if req_id is not None:
        parts.append(f"req={int(req_id)}")
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        elif isinstance(value, str):
            # Keep quotes off short tokens; quote longer / spaced strings.
            if " " in value or len(value) > 24:
                parts.append(f"{key}={value!r}")
            else:
                parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def log_corr(event: str, **kwargs: Any) -> None:
    if not corr_log_enabled():
        return
    print(format_corr(event, **kwargs), flush=True)


def summarize_first_tokens(
    token_ids: Sequence[int],
    *,
    tokenizer: Any | None = None,
    n: int = 12,
) -> dict[str, Any]:
    """Compact first-token fingerprint for a generate collect."""
    ids = [int(t) for t in token_ids[: max(0, int(n))]]
    out: dict[str, Any] = {
        "n_tok": len(token_ids),
        "first_ids": ",".join(str(i) for i in ids) if ids else "-",
    }
    if tokenizer is not None and ids:
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            text = ""
        # Single-line, truncated; enough to spot peer-topic bleed.
        text = " ".join(text.replace("\n", " ").split())
        if len(text) > 80:
            text = text[:79] + "…"
        out["first_text"] = text or "-"
    return out


def cmd_kind(cmd_line: str) -> str:
    """Classify daemon command for corr start lines."""
    body = cmd_line.strip().upper()
    # Peel REQ / SLOT prefixes.
    while True:
        if body.startswith("REQ ") or body.startswith("REQUEST "):
            skip = 8 if body.startswith("REQUEST ") else 4
            body = body[skip:].lstrip()
            # drop id token
            parts = body.split(None, 1)
            body = parts[1] if len(parts) > 1 else ""
            continue
        if body.startswith("SLOT "):
            parts = body.split(None, 2)
            body = parts[2] if len(parts) > 2 else ""
            continue
        break
    if body.startswith("RESTORE_CHAIN"):
        return "RESTORE_CHAIN"
    if body.startswith("RESTORE"):
        return "RESTORE"
    if body.startswith("GENERATE") or body.startswith("START"):
        return "GENERATE"
    head = body.split(None, 1)[0] if body else "EMPTY"
    return head[:24]


class OrphanFrameMeter:
    """Rate-limit demux orphan (unregistered req_id) drop logs."""

    def __init__(self, *, debounce_sec: float = 2.0) -> None:
        self._debounce = max(0.0, float(debounce_sec))
        self._last_log = 0.0
        self._dropped = 0
        self._last_req: int | None = None
        self._last_kind: str | None = None
        # Phase A: orphans that arrive shortly after CANCEL after_collect.
        self._after_collect_until = 0.0
        self.after_collect_orphans = 0

    def mark_after_collect(self, *, window_sec: float = 5.0) -> None:
        """Arm a short window where orphan drops count as after-collect."""
        self._after_collect_until = time.monotonic() + max(0.0, float(window_sec))

    def note(self, req_id: int, kind: str) -> None:
        self._dropped += 1
        self._last_req = int(req_id)
        self._last_kind = kind
        now = time.monotonic()
        after_collect = now < self._after_collect_until
        if after_collect:
            self.after_collect_orphans += 1
        if not corr_log_enabled():
            return
        if self._debounce > 0 and (now - self._last_log) < self._debounce:
            return
        self._last_log = now
        event = "demux_orphan_after_collect" if after_collect else "demux_orphan"
        log_corr(
            event,
            req_id=self._last_req,
            kind=self._last_kind,
            dropped=self._dropped,
            after_collect_total=self.after_collect_orphans if after_collect else None,
        )
        self._dropped = 0
