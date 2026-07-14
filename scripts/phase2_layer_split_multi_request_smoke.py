#!/usr/bin/env python3
"""Phase 2 layer-split multi-slot smoke on ai.local (M2a + M2b).

Runs outside the live lucebox container against a side-built test_dflash.
Expects VRAM free (stop model-runner-v4-lucebox first). Prefer AR-only
(no draft) and no DFLASH_KVFLASH (multi-slot refuses kvflash reattach).

Gates:
  N=1  LIST + generate + SNAPSHOT_THIN + RESTORE_CHAIN (tool-pin certify)
  N=2  tagged START+SCHED_DRAIN; slot_required; busy refuse;
       tool-pin RESTORE on slot 0 while slot 1 idle (and inverse)
"""
from __future__ import annotations

import argparse
import os
import re
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


# Minimal chat-ish token prefix used as the "tool" pin region.
TOOL_IDS = [151644, 872, 198, 8948, 151645, 198, 151644, 77091, 198]
TOOL_LEN = len(TOOL_IDS)
# Unique thin snapshot slots so A/B pins don't overwrite each other.
THIN_A = 4
THIN_B = 5
# Thick probe slots for live-cache fingerprints (isolation assert).
PROBE_A = 7
PROBE_B = 8


def write_raw_i32(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(ids)}i", *ids))


def write_counted_i32(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack("<i", len(ids)) + struct.pack(f"<{len(ids)}i", *ids))


def parse_cur_pos(line: str) -> int:
    m = re.search(r"cur_pos=(-?\d+)", line)
    if not m:
        raise AssertionError(f"no cur_pos in: {line}")
    return int(m.group(1))


class Daemon:
    def __init__(self, proc: subprocess.Popen, stream_r: int):
        self.proc = proc
        self.stream_r = stream_r
        self.stderr_lines: list[str] = []
        self._stream_lock = threading.Lock()
        self._parsed_frames: list[tuple[str, int, int | None]] = []
        self._stream_buf = b""
        self._stderr_thr = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thr.start()
        self._stream_thr = threading.Thread(target=self._drain_stream, daemon=True)
        self._stream_thr.start()
        self._await_ready()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))
            sys.stderr.write(f"[daemon.err] {line}")

    def _drain_stream(self) -> None:
        # Keep the token pipe drained so generate cannot block on a full pipe.
        while True:
            ready, _, _ = select.select([self.stream_r], [], [], 0.5)
            if not ready:
                if self.proc.poll() is not None:
                    return
                continue
            try:
                chunk = os.read(self.stream_r, 4096)
            except OSError:
                return
            if not chunk:
                return
            with self._stream_lock:
                self._stream_buf += chunk
                while len(self._stream_buf) >= 4:
                    (v,) = struct.unpack_from("<i", self._stream_buf, 0)
                    if v == -2:
                        if len(self._stream_buf) < 12:
                            break
                        _, req_id, tok = struct.unpack_from("<iii", self._stream_buf, 0)
                        self._stream_buf = self._stream_buf[12:]
                        kind = "done" if tok == -1 else "cont" if tok == -4 else "tag"
                        self._parsed_frames.append((kind, tok, req_id))
                    else:
                        self._stream_buf = self._stream_buf[4:]
                        kind = "done" if v == -1 else "cont" if v == -4 else "tok"
                        self._parsed_frames.append((kind, v, None))

    def _await_ready(self, timeout: float = 600.0) -> None:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"daemon exited early code={self.proc.returncode}\n"
                    + "\n".join(self.stderr_lines[-40:])
                )
            ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                continue
            sys.stderr.write(f"[daemon.out] {line}")
            low = line.lower()
            if "[daemon] ready" in low or "target_cache_slots=" in low:
                return
        raise TimeoutError("daemon ready banner timeout\n" + "\n".join(self.stderr_lines[-40:]))

    def cmd(self, line: str, expect_prefix: str | None = None, timeout: float = 120.0) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"daemon died after '{line}'")
            ready, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not ready:
                continue
            out = self.proc.stdout.readline()
            if not out:
                continue
            sys.stderr.write(f"[daemon.out] {out}")
            if expect_prefix is None or out.startswith(expect_prefix) or out.startswith("err "):
                return out.rstrip("\n")
        raise TimeoutError(f"timeout waiting for response to: {line}")

    def read_stream(self, max_frames: int = 64, idle: float = 0.5) -> list[tuple[str, int, int | None]]:
        """Return demuxed frames drained by the background stream reader."""
        deadline = time.time() + idle
        last_count = 0
        while time.time() < deadline:
            with self._stream_lock:
                count = len(self._parsed_frames)
            if count > last_count:
                last_count = count
                deadline = time.time() + idle
            time.sleep(0.05)
        with self._stream_lock:
            frames = [f for f in self._parsed_frames if f[0] != "raw"][-max_frames:]
            # Clear consumed frames so next call sees new data only.
            self._parsed_frames = []
            return frames

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        try:
            os.close(self.stream_r)
        except OSError:
            pass


def launch(
    bin_path: Path,
    target: Path,
    draft: Path | None,
    slots: int,
    tagged: bool,
    max_ctx: int,
    work: Path,
    target_gpus: str = "0,1",
) -> Daemon:
    stream_r, stream_w = os.pipe()
    env = os.environ.copy()
    env.pop("DFLASH_LEGACY_DAEMON", None)
    env.pop("DFLASH_KVFLASH", None)
    env["DFLASH27B_KV_TQ3"] = env.get("DFLASH27B_KV_TQ3", "1")
    # Side-binary on host needs build-tree .so dirs (compose sets this in-container).
    build = bin_path.resolve().parent
    lib_dirs = [
        build,
        build / "bin",
        build / "deps/llama.cpp/ggml/src",
        build / "deps/llama.cpp/ggml/src/ggml-cuda",
        build / "deps/llama.cpp/src",
        build / "deps/llama.cpp/common",
        build / "deps/llama.cpp/tools/mtmd",
    ]
    extra = ":".join(str(p) for p in lib_dirs if p.is_dir())
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{extra}:{prev}" if prev else extra
    # Prefer no-draft argv layout for VRAM headroom with N=2 partial caches.
    cmd = [str(bin_path), str(target)]
    if draft is not None:
        cmd.append(str(draft))
    cmd += [
        "--daemon",
        "--fast-rollback",
        f"--max-ctx={max_ctx}",
        f"--stream-fd={stream_w}",
        f"--target-cache-slots={slots}",
        f"--target-gpus={target_gpus}",
    ]
    if tagged:
        cmd.append("--stream-tagged")
    cmd = ["stdbuf", "-oL", "-eL", *cmd]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        pass_fds=(stream_w,),
        cwd=str(work),
        env=env,
    )
    os.close(stream_w)
    return Daemon(proc, stream_r)


def gate_n1(d: Daemon, work: Path) -> None:
    print("== N=1 LIST_TARGET_CACHE_SLOTS ==")
    out = d.cmd("LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon] target_cache_slots=")
    assert "target_cache_slots=1" in out, out

    prompt = work / "p1.bin"
    write_counted_i32(prompt, TOOL_IDS)
    out_path = work / "o1.bin"
    print("== N=1 generate ==")
    out = d.cmd(f"generate {prompt} 4 {out_path}", expect_prefix="ok ", timeout=300)
    assert out.startswith("ok "), out
    assert out_path.exists() and out_path.stat().st_size > 4

    print(f"== N=1 SNAPSHOT_THIN tool pin slot={THIN_A} kv=0,{TOOL_LEN} ==")
    # N=1: SLOT prefix optional. Pin tool region [0, TOOL_LEN).
    thin = d.cmd(f"SNAPSHOT_THIN {THIN_A} 0 {TOOL_LEN}", expect_prefix="[snap] thin", timeout=120)
    assert f"thin slot={THIN_A}" in thin, thin
    d.read_stream(idle=0.3)

    # Full prompt = tool prefix + short chat suffix; restore should suffix-prefill only.
    suffix_ids = [220, 220, 1110, 264]
    full = TOOL_IDS + suffix_ids
    restore_prompt = work / "restore1.raw"
    write_raw_i32(restore_prompt, full)
    print("== N=1 RESTORE_CHAIN (tool pin) ==")
    out = d.cmd(
        f"RESTORE_CHAIN -1 {THIN_A} {restore_prompt} 4",
        expect_prefix="ok ",
        timeout=300,
    )
    assert out.startswith("ok "), out
    assert "RESTORE_CHAIN thick=-1" in out, out
    # prefix_len may be filled asynchronously after the ok ack (-1 placeholder).
    assert "suffix_n=" in out or "prefix_len=" in out, out
    d.read_stream(idle=0.5)
    print("N=1 OK")


def _finger_live(d: Daemon, live_slot: int, probe_slot: int) -> int:
    """Activate live_slot, SNAPSHOT into probe_slot, return cur_pos fingerprint."""
    snap = d.cmd(
        f"SLOT {live_slot} SNAPSHOT {probe_slot}",
        expect_prefix="[snap] inline",
        timeout=120,
    )
    assert f"inline slot={probe_slot}" in snap, snap
    assert f"live={live_slot}" in snap, snap
    return parse_cur_pos(snap)


def _seed_and_pin(
    d: Daemon,
    work: Path,
    live_slot: int,
    thin_slot: int,
    prompt_ids: list[int],
    tool_len: int,
) -> Path:
    """Prefill live_slot via generate, pin thin tool region, return work dir tag."""
    p = work / f"seed_s{live_slot}.bin"
    o = work / f"seed_o{live_slot}.bin"
    write_counted_i32(p, prompt_ids)
    # SLOT before generate so the seed lands in the intended live TargetCache.
    # generate itself does not parse SLOT — activate first, then generate.
    act = d.cmd(f"SLOT {live_slot} LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon]")
    assert f"active={live_slot}" in act, act
    out = d.cmd(f"generate {p} 4 {o}", expect_prefix="ok ", timeout=300)
    assert out.startswith("ok "), out
    thin = d.cmd(
        f"SLOT {live_slot} SNAPSHOT_THIN {thin_slot} 0 {tool_len}",
        expect_prefix="[snap] thin",
        timeout=120,
    )
    assert f"thin slot={thin_slot}" in thin, thin
    d.read_stream(idle=0.3)
    return p


def _isolation_restore(
    d: Daemon,
    work: Path,
    *,
    restore_slot: int,
    idle_slot: int,
    thin_slot: int,
    idle_probe: int,
) -> None:
    print(f"== N=2 isolation: RESTORE on live={restore_slot}, idle={idle_slot} ==")
    idle_pos_before = _finger_live(d, idle_slot, idle_probe)
    print(f"  idle slot {idle_slot} fingerprint cur_pos={idle_pos_before}")

    # Bare RESTORE_CHAIN must refuse when N>1 (no SLOT).
    bare = work / "bare.raw"
    write_raw_i32(bare, TOOL_IDS + [220, 220])
    bad = d.cmd(f"RESTORE_CHAIN -1 {thin_slot} {bare} 2", expect_prefix="err ", timeout=30)
    assert "err slot_required" in bad, bad
    print("  slot_required OK")

    suffix = TOOL_IDS + [220, 1110, 264, 1234]
    rp = work / f"restore_s{restore_slot}.raw"
    write_raw_i32(rp, suffix)
    out = d.cmd(
        f"SLOT {restore_slot} RESTORE_CHAIN -1 {thin_slot} {rp} 4",
        expect_prefix="ok ",
        timeout=300,
    )
    assert out.startswith("ok "), out
    assert f"live={restore_slot}" in out, out
    assert "RESTORE_CHAIN thick=-1" in out, out
    d.read_stream(idle=0.5)

    idle_pos_after = _finger_live(d, idle_slot, idle_probe + 1)
    print(f"  idle slot {idle_slot} after restore cur_pos={idle_pos_after}")
    assert idle_pos_after == idle_pos_before, (
        f"idle live slot {idle_slot} mutated: {idle_pos_before} -> {idle_pos_after}"
    )
    print(f"  isolation OK (live {restore_slot} restored, {idle_slot} untouched)")


def gate_n2(d: Daemon, work: Path) -> None:
    print("== N=2 LIST ==")
    out = d.cmd("LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon] target_cache_slots=")
    assert "target_cache_slots=2" in out, out

    p0 = work / "p0.raw"
    p1 = work / "p1.raw"
    # START expects raw int32 LE tokens (not count-prefixed).
    write_raw_i32(p0, TOOL_IDS)
    write_raw_i32(p1, TOOL_IDS + [220, 220])

    print("== N=2 START req0 ==")
    # REQ/SLOT are prefixes on the same line — not standalone commands.
    out = d.cmd(f"REQ 1 START {p0} 12 4", expect_prefix="ok START", timeout=300)
    assert out.startswith("ok START"), out
    frames0 = d.read_stream(idle=1.0)
    print(f"  frames after START0: {frames0[:8]}...")

    print("== N=2 START req1 ==")
    out = d.cmd(f"REQ 2 START {p1} 12 4", expect_prefix="ok START", timeout=300)
    assert out.startswith("ok START"), out
    frames1 = d.read_stream(idle=1.0)

    print("== N=2 SCHED_DRAIN ==")
    out = d.cmd("SCHED_DRAIN", expect_prefix="ok SCHED_DRAIN", timeout=300)
    assert out.startswith("ok SCHED_DRAIN"), out
    frames = frames0 + frames1 + d.read_stream(max_frames=128, idle=2.0)
    tagged = [f for f in frames if f[0] == "tag"]
    req_ids = {f[2] for f in tagged if f[2] is not None}
    print(f"  tagged_tokens={len(tagged)} req_ids={sorted(req_ids)}")
    assert len(tagged) >= 2, f"expected tagged tokens, got {frames}"
    assert len(req_ids) >= 1, f"expected demuxable req ids, got {frames}"

    print("== N=2 busy RESTORE_CHAIN ==")
    listing = d.cmd("LIST_REQUESTS", expect_prefix="[scheduler]")
    if "requests=" in listing and listing.strip() != "[scheduler] requests=":
        bad = d.cmd(f"SLOT 0 RESTORE_CHAIN 0 - {p0} 2", expect_prefix="err ", timeout=30)
        assert "err slot_busy" in bad, bad
        print("  slot_busy OK")
    else:
        start = d.cmd(f"REQ 3 SLOT 0 START {p0} 64 8", expect_prefix="ok START", timeout=300)
        assert start.startswith("ok START"), start
        d.read_stream(idle=0.5)
        bad = d.cmd(f"SLOT 0 RESTORE_CHAIN 0 - {p0} 2", expect_prefix="err ", timeout=30)
        assert "err slot_busy" in bad, bad
        print("  slot_busy OK (re-admit)")
        out = d.cmd("CANCEL 3", expect_prefix="[scheduler]", timeout=30)
        assert "cancelled" in out or out.startswith("[scheduler]"), out
        d.read_stream(idle=0.5)

    # --- M2b: tool pin + RESTORE isolation (after scheduler traffic settles) ---
    # Cancel any leftover busy requests so generate/RESTORE are free.
    listing = d.cmd("LIST_REQUESTS", expect_prefix="[scheduler]")
    for m in re.finditer(r"(\d+)@slot", listing):
        rid = m.group(1)
        d.cmd(f"CANCEL {rid}", expect_prefix="[scheduler]", timeout=30)
        d.read_stream(idle=0.2)

    # Distinct seeds so idle fingerprint differs from restore target.
    seed0 = TOOL_IDS
    seed1 = TOOL_IDS + [220, 220, 220, 1110]
    assert len(seed0) == TOOL_LEN
    assert len(seed1) > TOOL_LEN

    print("== N=2 seed slot 0 + pin thin A ==")
    _seed_and_pin(d, work, live_slot=0, thin_slot=THIN_A, prompt_ids=seed0, tool_len=TOOL_LEN)
    print("== N=2 seed slot 1 (idle victim, longer) ==")
    # Seed slot 1 but do NOT pin from it for first isolation — leave as parked live KV.
    p = work / "seed_s1.bin"
    o = work / "seed_o1.bin"
    write_counted_i32(p, seed1)
    act = d.cmd("SLOT 1 LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon]")
    assert "active=1" in act, act
    out = d.cmd(f"generate {p} 4 {o}", expect_prefix="ok ", timeout=300)
    assert out.startswith("ok "), out

    _isolation_restore(
        d, work, restore_slot=0, idle_slot=1, thin_slot=THIN_A, idle_probe=PROBE_A
    )

    print("== N=2 inverse: pin thin B on slot 1, restore while slot 0 idle ==")
    _seed_and_pin(d, work, live_slot=1, thin_slot=THIN_B, prompt_ids=seed1, tool_len=TOOL_LEN)
    # Refresh slot 0 parked state with a known distinct seed for fingerprint.
    p0b = work / "seed_s0b.bin"
    o0b = work / "seed_o0b.bin"
    write_counted_i32(p0b, seed0 + [999])
    act = d.cmd("SLOT 0 LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon]")
    assert "active=0" in act, act
    out = d.cmd(f"generate {p0b} 4 {o0b}", expect_prefix="ok ", timeout=300)
    assert out.startswith("ok "), out

    _isolation_restore(
        d, work, restore_slot=1, idle_slot=0, thin_slot=THIN_B, idle_probe=PROBE_B
    )
    print("N=2 OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="", help="Optional draft path; omit for AR-only (fits 24GB)")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--target-gpus", default="0,1")
    ap.add_argument("--gate", choices=("n1", "n2", "all"), default="all")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    target = Path(args.target)
    draft = Path(args.draft) if args.draft else None
    assert bin_path.is_file(), bin_path
    assert target.is_file(), target
    if draft is not None:
        assert draft.is_file(), draft

    with tempfile.TemporaryDirectory(prefix="phase2-ls-smoke-") as td:
        work = Path(td)
        if args.gate in ("n1", "all"):
            print("\n===== GATE N=1 (layer-split / M2b tool-pin) =====")
            d = launch(
                bin_path, target, draft, slots=1, tagged=False,
                max_ctx=args.max_ctx, work=work, target_gpus=args.target_gpus,
            )
            try:
                gate_n1(d, work)
            finally:
                d.close()

        if args.gate in ("n2", "all"):
            print("\n===== GATE N=2 (layer-split / M2b isolation) =====")
            d = launch(
                bin_path, target, draft, slots=2, tagged=True,
                max_ctx=args.max_ctx, work=work, target_gpus=args.target_gpus,
            )
            try:
                gate_n2(d, work)
            finally:
                d.close()

    print("\nALL PHASE-2 M2b GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
