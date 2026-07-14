#!/usr/bin/env python3
"""Phase 1 gate: side-binary single-GPU multi-request smoke on ai.local.

Runs outside the live lucebox container against test_dflash.phase1.
Expects VRAM free (stop model-runner-v4-lucebox first).

Gates:
  N=1  LIST_TARGET_CACHE_SLOTS + generate + SNAPSHOT + RESTORE
  N=2  two START + SCHED_DRAIN; demux tagged frames; no crash
  busy RESTORE_CHAIN returns err slot_busy while a request owns the slot
"""
from __future__ import annotations

import argparse
import os
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def write_raw_i32(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(ids)}i", *ids))


def write_counted_i32(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack("<i", len(ids)) + struct.pack(f"<{len(ids)}i", *ids))


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

    def _await_ready(self, timeout: float = 180.0) -> None:
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
) -> Daemon:
    stream_r, stream_w = os.pipe()
    env = os.environ.copy()
    env.pop("DFLASH_LEGACY_DAEMON", None)
    env["DFLASH27B_KV_TQ3"] = env.get("DFLASH27B_KV_TQ3", "1")
    # Prefer no-draft argv layout so 27B Q4 fits a single 24GB card for the spike.
    # test_dflash treats argv[2] starting with '-' as no-draft.
    cmd = [str(bin_path), str(target)]
    if draft is not None:
        cmd.append(str(draft))
    cmd += [
        "--daemon",
        "--fast-rollback",
        f"--max-ctx={max_ctx}",
        f"--stream-fd={stream_w}",
        f"--target-cache-slots={slots}",
        "--target-gpu=0",
    ]
    if tagged:
        cmd.append("--stream-tagged")
    # Force line-buffered stdio before any printf (setvbuf after [cfg] is too late).
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
    prompt_raw = work / "p1.raw"
    # Minimal chat-ish token soup; just needs a valid forward pass.
    ids = [151644, 872, 198, 8948, 151645, 198, 151644, 77091, 198]
    write_counted_i32(prompt, ids)
    write_raw_i32(prompt_raw, ids)
    out_path = work / "o1.bin"
    print("== N=1 generate ==")
    out = d.cmd(f"generate {prompt} 8 {out_path}", expect_prefix="ok ", timeout=180)
    assert out.startswith("ok "), out
    assert out_path.exists() and out_path.stat().st_size > 4

    print("== N=1 SNAPSHOT + RESTORE ==")
    d.cmd("SNAPSHOT 0", expect_prefix="[snap]")
    # Prefer RESTORE (single-GPU); RESTORE_CHAIN is layer-split/tool-split path.
    # RESTORE reads uncounted int32 tokens.
    out = d.cmd(f"RESTORE 0 {prompt_raw} 4", timeout=180)
    assert out.startswith("ok "), out
    d.read_stream(idle=1.0)
    print("N=1 OK")


def gate_n2(d: Daemon, work: Path) -> None:
    print("== N=2 LIST ==")
    out = d.cmd("LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon] target_cache_slots=")
    assert "target_cache_slots=2" in out, out

    p0 = work / "p0.raw"
    p1 = work / "p1.raw"
    # START expects raw int32 LE tokens (not count-prefixed).
    ids = [151644, 872, 198, 8948, 151645, 198, 151644, 77091, 198]
    write_raw_i32(p0, ids)
    write_raw_i32(p1, ids + [220, 220])

    print("== N=2 START req0 ==")
    # REQ/SLOT are prefixes on the same line — not standalone commands.
    out = d.cmd(f"REQ 1 START {p0} 12 4", expect_prefix="ok START", timeout=180)
    assert out.startswith("ok START"), out
    frames0 = d.read_stream(idle=1.0)
    print(f"  frames after START0: {frames0[:8]}...")

    print("== N=2 START req1 ==")
    out = d.cmd(f"REQ 2 START {p1} 12 4", expect_prefix="ok START", timeout=180)
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
        bad = d.cmd(f"RESTORE_CHAIN 0 - {p0} 2", timeout=30)
        assert "err slot_busy" in bad, bad
        print("  slot_busy OK")
    else:
        start = d.cmd(f"REQ 3 START {p0} 64 8", expect_prefix="ok START", timeout=180)
        assert start.startswith("ok START"), start
        d.read_stream(idle=0.5)
        bad = d.cmd(f"RESTORE_CHAIN 0 - {p0} 2", timeout=30)
        assert "err slot_busy" in bad, bad
        print("  slot_busy OK (re-admit)")
        # CANCEL prints a scheduler line; accept either cancelled or silence via LIST.
        out = d.cmd("CANCEL 3", expect_prefix="[scheduler]", timeout=30)
        assert "cancelled" in out or out.startswith("[scheduler]"), out
        d.read_stream(idle=0.5)
    print("N=2 OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="", help="Optional draft path; omit for AR-only (fits 24GB)")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--gate", choices=("n1", "n2", "all"), default="all")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    target = Path(args.target)
    draft = Path(args.draft) if args.draft else None
    assert bin_path.is_file(), bin_path
    assert target.is_file(), target
    if draft is not None:
        assert draft.is_file(), draft

    with tempfile.TemporaryDirectory(prefix="phase1-smoke-") as td:
        work = Path(td)
        if args.gate in ("n1", "all"):
            print("\n===== GATE N=1 =====")
            d = launch(bin_path, target, draft, slots=1, tagged=False, max_ctx=args.max_ctx, work=work)
            try:
                gate_n1(d, work)
            finally:
                d.close()

        if args.gate in ("n2", "all"):
            print("\n===== GATE N=2 =====")
            d = launch(bin_path, target, draft, slots=2, tagged=True, max_ctx=args.max_ctx, work=work)
            try:
                gate_n2(d, work)
            finally:
                d.close()

    print("\nALL PHASE-1 GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
