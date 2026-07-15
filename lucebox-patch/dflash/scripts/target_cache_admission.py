"""Live target-cache slot admission + SLOT command prefix (Phase 3 M3a).

Separate from PrefixSnapshot / tool-pin slot ids: these are the N VRAM live
``TargetCache`` slots allocated with ``--target-cache-slots``.
"""
from __future__ import annotations

import asyncio
import os
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import AsyncIterator, Callable


_active_live_slot: ContextVar[int | None] = ContextVar(
    "dflash_active_live_slot", default=None
)


def target_cache_slots() -> int:
    """Number of live target-cache slots (default 1). Clamped to 1..16."""
    raw = os.environ.get("DFLASH_TARGET_CACHE_SLOTS", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 1
    return max(1, min(n, 16))


def stream_tagged_enabled() -> bool:
    raw = os.environ.get("DFLASH_STREAM_TAGGED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def multi_slot_drop_exclusive() -> bool:
    """When N>1, skip PriorityDaemonLock (needs tagged demux — M3b).

    Default off so compose can plumb N>1 for SLOT sticky tests without
    interleaving untagged token streams. Prefer ``overlap_mode_enabled()``
    once demux + START/SCHED are wired on the HTTP path.
    """
    raw = os.environ.get("DFLASH_MULTI_SLOT_DROP_EXCLUSIVE", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def overlap_mode_enabled() -> bool:
    """True when live multi-slot + tagged stream are both configured."""
    return target_cache_slots() > 1 and stream_tagged_enabled()


def schedule_quantum() -> int:
    """Decode quantum for RESTORE_CHAIN admit / START (default 8)."""
    raw = os.environ.get("DFLASH_SCHED_QUANTUM", "8").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, min(n, 4096))


def _peel_req_slot_prefixes(body: str) -> tuple[str, str]:
    """Split leading ``REQ`` / ``SLOT`` tokens from a daemon command body.

    Returns ``(prefix_with_trailing_space, remainder)``. Prefix is empty when
    neither decorator is present.
    """
    prefixes: list[str] = []
    rest = body.strip()
    while rest:
        upper = rest.upper()
        if upper.startswith("REQ ") or upper.startswith("REQUEST "):
            skip = 8 if upper.startswith("REQUEST ") else 4
            p = skip
            while p < len(rest) and rest[p].isspace():
                p += 1
            while p < len(rest) and not rest[p].isspace():
                p += 1
            prefixes.append(rest[:p].strip())
            rest = rest[p:].lstrip()
            continue
        if upper.startswith("SLOT "):
            p = 5
            while p < len(rest) and rest[p].isspace():
                p += 1
            while p < len(rest) and not rest[p].isspace():
                p += 1
            prefixes.append(rest[:p].strip())
            rest = rest[p:].lstrip()
            continue
        break
    prefix = (" ".join(prefixes) + " ") if prefixes else ""
    return prefix, rest


def is_restore_chain_command(line: str) -> bool:
    """True when the command is RESTORE_CHAIN after optional REQ/SLOT prefixes."""
    body = line[:-1] if line.endswith("\n") else line
    _, rest = _peel_req_slot_prefixes(body.strip())
    return rest.upper().startswith("RESTORE_CHAIN ")


def append_restore_chain_quantum(line: str, quantum: int | None = None) -> str:
    """Append ``<quantum>`` to a RESTORE_CHAIN line when missing.

    Leaves non-RESTORE_CHAIN lines and already-quantized lines unchanged.
    Honors leading ``REQ`` / ``SLOT`` prefixes produced by format helpers.
    """
    q = schedule_quantum() if quantum is None else max(1, int(quantum))
    nl = line.endswith("\n")
    body = line[:-1] if nl else line
    body = body.strip()
    prefix, rest = _peel_req_slot_prefixes(body)
    if not rest.upper().startswith("RESTORE_CHAIN "):
        return line
    # Strip trailing snap= so quantum sits before snap=.
    snap = ""
    if " snap=" in rest:
        rest, snap_part = rest.split(" snap=", 1)
        snap = f" snap={snap_part}"
        rest = rest.strip()
    parts = rest.split()
    # RESTORE_CHAIN thick thin path n_gen [quantum]
    if len(parts) >= 6:
        try:
            int(parts[5])
            return line  # already has quantum
        except ValueError:
            pass
    if len(parts) < 5:
        return line
    rest = f"{' '.join(parts[:5])} {q}{snap}"
    return f"{prefix}{rest}" + ("\n" if nl else "")


def parse_restore_chain_admit_remaining(line: str) -> int | None:
    """Parse ``remaining=N`` from an ``ok RESTORE_CHAIN_ADMIT …`` reply."""
    if "RESTORE_CHAIN_ADMIT" not in line:
        return None
    for part in line.split():
        if part.startswith("remaining="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def format_req_prefix_needed() -> bool:
    """Commands should carry ``REQ <id>`` when the daemon uses tagged emit."""
    return stream_tagged_enabled()


def active_live_slot() -> int | None:
    return _active_live_slot.get()


def set_active_live_slot(slot: int) -> Token:
    return _active_live_slot.set(int(slot))


def reset_active_live_slot(token: Token | None) -> None:
    """Reset the live-slot ContextVar; tolerate cross-context SSE teardown.

    Starlette/anyio may run StreamingResponse finally in a different context
    than the one that called ``set_active_live_slot``. ``Token.reset`` then
    raises ``ValueError`` and would otherwise skip lease release.
    """
    if token is None:
        _active_live_slot.set(None)
        return
    try:
        _active_live_slot.reset(token)
    except ValueError:
        _active_live_slot.set(None)


def format_slot_command(
    line: str,
    slot: int | None = None,
    *,
    slots: int | None = None,
) -> str:
    """Prefix ``SLOT k `` when live target-cache slots > 1.

    No-op for N<=1 (daemon does not require SLOT). Uses the active ContextVar
    lease when ``slot`` is omitted.
    """
    n = target_cache_slots() if slots is None else max(1, int(slots))
    if n <= 1:
        return line
    if slot is None:
        slot = _active_live_slot.get()
    if slot is None:
        raise ValueError("target_cache_slots>1 requires an explicit or active SLOT")
    nl = line.endswith("\n")
    body = line[:-1] if nl else line
    body = body.strip()
    if not body:
        out = f"SLOT {int(slot)}"
    elif body.upper().startswith("SLOT "):
        out = body
    else:
        out = f"SLOT {int(slot)} {body}"
    return out + ("\n" if nl else "")


def sticky_affinity_key(key: str | None, *, scoped: bool) -> str:
    """Normalize lease affinity. Ephemeral traffic does not keep sticky slots."""
    if key and not key.startswith("ephemeral:"):
        return key
    if scoped:
        return key or "scoped:anonymous"
    return key or f"ephemeral:{id(object())}"


def reserved_fast_slots(n_slots: int) -> int:
    """Slots held exclusively for fast-lane (``/v1``) traffic.

    Auto: reserve 1 when ``n_slots >= 2`` so ``/v1e`` cannot take the last
    live slot. Override with ``DFLASH_RESERVED_FAST_SLOTS`` (clamped to
    ``0 .. n_slots-1``).
    """
    n = max(1, int(n_slots))
    raw = os.environ.get("DFLASH_RESERVED_FAST_SLOTS", "").strip()
    if not raw:
        return 1 if n >= 2 else 0
    try:
        return max(0, min(int(raw), n - 1))
    except ValueError:
        return 1 if n >= 2 else 0


@dataclass(frozen=True)
class SlotLease:
    slot: int
    key: str
    scoped: bool
    lane: str = "fast"


class TargetCacheSlotPool:
    """Sticky free-list of live target-cache slots with three-tier waits.

    Queues (wake order: high → mid → low):

    - **high** — scoped ``/v1``
    - **mid** — unscoped ``/v1``
    - **low** — slow lane ``/v1e``

    Any ``/v1`` enqueue drains waiting ``/v1e``. Scoped also drains mid.
    When a fast waiter cannot grant and a slow holder exists, ``bump_slow``
    is invoked so in-flight ``/v1e`` can CANCEL and free a slot.
    """

    def __init__(self, n_slots: int) -> None:
        self._n = max(1, min(int(n_slots), 16))
        self._free: deque[int] = deque(range(self._n))
        self._sticky: dict[str, int] = {}
        self._held: dict[int, str] = {}
        self._held_lane: dict[int, str] = {}
        # Waiter: (fut, key, scoped, lane)
        self._high: deque[tuple[asyncio.Future[SlotLease], str, bool, str]] = deque()
        self._mid: deque[tuple[asyncio.Future[SlotLease], str, bool, str]] = deque()
        self._low: deque[tuple[asyncio.Future[SlotLease], str, bool, str]] = deque()
        self._bump_slow: Callable[[], None] | None = None

    def set_bump_slow_callback(self, cb: Callable[[], None] | None) -> None:
        self._bump_slow = cb

    @property
    def n_slots(self) -> int:
        return self._n

    def held_slots(self) -> frozenset[int]:
        return frozenset(self._held)

    def sticky_slot(self, key: str) -> int | None:
        return self._sticky.get(key)

    def _reserved_fast(self) -> int:
        return reserved_fast_slots(self._n)

    @staticmethod
    def _tier(*, scoped: bool, lane: str) -> str:
        if lane == "slow":
            return "low"
        if scoped:
            return "high"
        return "mid"

    def _queue_for(
        self, tier: str
    ) -> deque[tuple[asyncio.Future[SlotLease], str, bool, str]]:
        if tier == "high":
            return self._high
        if tier == "mid":
            return self._mid
        return self._low

    def _has_slow_holder(self) -> bool:
        return any(lane == "slow" for lane in self._held_lane.values())

    def _maybe_bump_slow(self, *, waiter_lane: str) -> None:
        if waiter_lane == "slow":
            return
        if self._has_slow_holder() and self._bump_slow is not None:
            self._bump_slow()

    def _try_grant(
        self,
        key: str,
        *,
        scoped: bool,
        lane: str = "fast",
    ) -> SlotLease | None:
        sticky = self._sticky.get(key)
        if sticky is not None and sticky not in self._held:
            try:
                self._free.remove(sticky)
            except ValueError:
                pass
            self._held[sticky] = key
            self._held_lane[sticky] = lane
            return SlotLease(slot=sticky, key=key, scoped=scoped, lane=lane)
        if not self._free:
            return None
        if lane == "slow" and len(self._free) <= self._reserved_fast():
            return None
        slot = self._free.popleft()
        keep_sticky = scoped and not key.startswith("ephemeral:")
        if keep_sticky:
            self._sticky[key] = slot
        self._held[slot] = key
        self._held_lane[slot] = lane
        return SlotLease(slot=slot, key=key, scoped=scoped, lane=lane)

    def _wake_one_queue(
        self,
        queue: deque[tuple[asyncio.Future[SlotLease], str, bool, str]],
    ) -> bool:
        """Grant as many waiters as free slots allow from ``queue``.

        Returns False if a grant failed while free slots remain (reserved).
        """
        while self._free and queue:
            fut, key, scoped, lane = queue.popleft()
            if fut.done():
                continue
            lease = self._try_grant(key, scoped=scoped, lane=lane)
            if lease is None:
                queue.appendleft((fut, key, scoped, lane))
                return False
            fut.set_result(lease)
        return True

    def _wake_waiters(self) -> None:
        if not self._wake_one_queue(self._high):
            return
        if self._high:
            return
        if not self._wake_one_queue(self._mid):
            return
        if self._high or self._mid:
            return
        self._wake_one_queue(self._low)

    def _drain_queue(
        self,
        queue: deque[tuple[asyncio.Future[SlotLease], str, bool, str]],
    ) -> None:
        while queue:
            efut, _ek, _es, _el = queue.popleft()
            if not efut.done():
                efut.cancel()

    def _drop_waiter(
        self,
        fut: asyncio.Future[SlotLease],
        *,
        tier: str,
    ) -> None:
        queue = self._queue_for(tier)
        for i, (f, _k, _s, _lane) in enumerate(queue):
            if f is fut:
                del queue[i]
                break
        if not fut.done():
            fut.cancel()

    async def acquire(
        self,
        key: str,
        *,
        scoped: bool = True,
        max_wait: float = float("inf"),
        lane: str = "fast",
    ) -> SlotLease:
        lane = "slow" if lane == "slow" else "fast"
        if lane == "slow":
            scoped = False
        key = sticky_affinity_key(key, scoped=scoped)
        lease = self._try_grant(key, scoped=scoped, lane=lane)
        if lease is not None:
            return lease

        tier = self._tier(scoped=scoped, lane=lane)
        # Fail-fast: slow lane never queues behind any /v1 waiter.
        if tier == "low" and (self._high or self._mid):
            raise asyncio.TimeoutError()

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[SlotLease] = loop.create_future()
        self._queue_for(tier).append((fut, key, scoped, lane))
        if tier == "high":
            self._drain_queue(self._mid)
            self._drain_queue(self._low)
            self._maybe_bump_slow(waiter_lane=lane)
        elif tier == "mid":
            self._drain_queue(self._low)
            self._maybe_bump_slow(waiter_lane=lane)
        if not self._free:
            self._maybe_bump_slow(waiter_lane=lane)
        try:
            if max_wait == float("inf"):
                return await fut
            return await asyncio.wait_for(fut, timeout=max_wait)
        except asyncio.TimeoutError:
            self._drop_waiter(fut, tier=tier)
            raise
        except asyncio.CancelledError:
            self._drop_waiter(fut, tier=tier)
            raise

    def release(self, lease: SlotLease) -> None:
        owner = self._held.get(lease.slot)
        if owner is None:
            return
        if owner != lease.key:
            return
        del self._held[lease.slot]
        self._held_lane.pop(lease.slot, None)
        if lease.key.startswith("ephemeral:"):
            self._sticky.pop(lease.key, None)
        if lease.slot not in self._free:
            self._free.append(lease.slot)
        self._wake_waiters()

    @asynccontextmanager
    async def lease(
        self,
        key: str,
        *,
        scoped: bool = True,
        max_wait: float = float("inf"),
        lane: str = "fast",
    ) -> AsyncIterator[SlotLease]:
        acquired = await self.acquire(
            key, scoped=scoped, max_wait=max_wait, lane=lane,
        )
        tok = set_active_live_slot(acquired.slot)
        try:
            yield acquired
        finally:
            reset_active_live_slot(tok)
            self.release(acquired)
