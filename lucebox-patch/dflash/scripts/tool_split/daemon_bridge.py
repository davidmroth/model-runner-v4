"""Daemon IPC for tool KV thin snapshots."""
from __future__ import annotations

import asyncio
from typing import Any

from handler_reliability import tool_inline_snap_pin_enabled, tool_snapshot_max_kv_tokens


def tool_snap_prep_from_pending(
    pending_tool_snap: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Return ``(pin_slot, kv_end)`` for inline tool snap on cold prefill, or ``None``.

    Inline ``snap=`` into the tool pin slot is only used when ``SNAPSHOT_THIN`` is
    skipped (depth above ``DFLASH_TOOL_SNAPSHOT_MAX_KV``). Below that threshold,
    ``SNAPSHOT_THIN`` alone registers the thin slot for ``RESTORE_CHAIN``; combining
    both on the same slot crashes the layer-split daemon.
    """
    if not tool_inline_snap_pin_enabled() or pending_tool_snap is None:
        return None
    slot, kv_end = pending_tool_snap
    if kv_end <= 0:
        return None
    max_kv = tool_snapshot_max_kv_tokens()
    if max_kv > 0 and kv_end <= max_kv:
        return None
    return (slot, kv_end)


def append_inline_snap(cmd: str, snap: tuple[int, int] | None) -> str:
    """Append ``snap=cut:slot`` to a daemon command line (no trailing newline)."""
    if snap is None:
        return cmd
    slot, cut = snap
    return f"{cmd} snap={cut}:{slot}"


async def finish_tool_inline_snap(
    *,
    orchestrator,
    bus: Any,
    fingerprint: str,
    tool_snap_prep: tuple[int, int] | None,
) -> bool:
    """Confirm tool pin when inline ``snap=`` registered a thin slot in the daemon."""
    if tool_snap_prep is None or not fingerprint:
        return False
    slot, kv_end = tool_snap_prep
    await bus.drain_inline_snap()
    if bus.inline_snap_slot() == slot:
        orchestrator.tool_slots.confirm(fingerprint, slot)
        print(
            f"[tool-split] tool KV pinned slot={slot} len={kv_end} "
            f"fp={fingerprint[:12]}… (inline thin)",
            flush=True,
        )
        return True
    print(
        f"[tool-split] inline tool snap missed slot={slot} "
        f"(ack={bus.inline_snap_slot()!r}) fp={fingerprint[:12]}…",
        flush=True,
    )
    return False


async def snapshot_thin(
    *,
    daemon_stdin,
    await_reply,
    slot: int,
    kv_start: int,
    kv_end: int,
) -> bool:
    """Capture KV range ``[kv_start, kv_end)`` into a thin snapshot slot."""
    line = f"SNAPSHOT_THIN {slot} {kv_start} {kv_end}\n"
    daemon_stdin.write(line.encode("utf-8"))
    daemon_stdin.flush()
    try:
        reply = await await_reply("[snap] thin slot=", timeout=30.0)
    except (asyncio.TimeoutError, EOFError) as exc:
        print(f"[tool-split] SNAPSHOT_THIN slot={slot} failed: {exc}", flush=True)
        return False
    ok = reply.startswith(f"[snap] thin slot={slot} kv={kv_start},{kv_end}")
    if not ok:
        print(f"[tool-split] SNAPSHOT_THIN unexpected reply: {reply!r}", flush=True)
    return ok


async def commit_pending_tool_snap(
    *,
    orchestrator,
    daemon_stdin,
    await_reply,
    fingerprint: str,
    slot: int,
    kv_end: int,
) -> None:
    if kv_end <= 0:
        orchestrator.tool_slots.release_reservation(fingerprint, slot)
        return
    if orchestrator.tool_slots.pinned_slot(fingerprint) is not None:
        return
    max_kv = tool_snapshot_max_kv_tokens()
    if max_kv > 0 and kv_end > max_kv:
        orchestrator.tool_slots.release_reservation(fingerprint, slot)
        print(
            f"[tool-split] SNAPSHOT_THIN skipped kv_end={kv_end} > max={max_kv} "
            f"(set DFLASH_TOOL_SNAPSHOT_MAX_KV=0 to force; may crash daemon)",
            flush=True,
        )
        return
    ok = await snapshot_thin(
        daemon_stdin=daemon_stdin,
        await_reply=await_reply,
        slot=slot,
        kv_start=0,
        kv_end=kv_end,
    )
    if ok:
        orchestrator.tool_slots.confirm(fingerprint, slot)
        print(
            f"[tool-split] tool KV pinned slot={slot} len={kv_end} "
            f"fp={fingerprint[:12]}…",
            flush=True,
        )
    else:
        orchestrator.tool_slots.release_reservation(fingerprint, slot)
        print(
            f"[tool-split] SNAPSHOT_THIN failed; released reservation "
            f"slot={slot} fp={fingerprint[:12]}…",
            flush=True,
        )
