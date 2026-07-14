#!/usr/bin/env python3
"""Phase 3 M3b gate: dual START + SCHED_DRAIN demuxed by TaggedFrameBuffer.

Side-binary layer-split smoke (stop model-runner-v4-lucebox first).
Uses the production demux parser from ``tagged_stream_demux.py``.

Gates:
  N=2  REQ START ×2 + SCHED_DRAIN; both req_ids receive tagged tokens
  compose stays N=1 — do not enable DFLASH_TARGET_CACHE_SLOTS=2 in prod yet
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

# Prefer in-tree patch scripts (same module serve uses).
_SCRIPTS = Path(__file__).resolve().parents[1] / "lucebox-patch/dflash/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tagged_stream_demux import TaggedFrameBuffer  # noqa: E402


def write_raw_i32(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(ids)}i", *ids))


class Daemon:
    def __init__(self, proc: subprocess.Popen, stream_r: int):
        self.proc = proc
        self.stream_r = stream_r
        self.stderr_lines: list[str] = []
        self._stream_lock = threading.Lock()
        self._parser = TaggedFrameBuffer()
        self._frames: list[tuple[str, int, int | None]] = []
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
                for fr in self._parser.push(chunk):
                    self._frames.append((fr.kind, fr.value, fr.req_id))

    def _await_ready(self, timeout: float = 180.0) -> None:
        assert self.proc.stdout is not None
        # Also drain stdout in background so RESTORE_CHAIN / ok lines never block.
        # (phase2 hang lesson — keep the pipe empty while we wait for banners.)
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
        raise TimeoutError(
            "daemon ready banner timeout\n" + "\n".join(self.stderr_lines[-40:])
        )

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

    def read_stream(self, max_frames: int = 128, idle: float = 0.5):
        deadline = time.time() + idle
        last_count = 0
        while time.time() < deadline:
            with self._stream_lock:
                count = len(self._frames)
            if count > last_count:
                last_count = count
                deadline = time.time() + idle
            time.sleep(0.05)
        with self._stream_lock:
            frames = self._frames[-max_frames:]
            self._frames = []
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
    *,
    slots: int,
    max_ctx: int,
    target_gpus: str,
    work: Path,
) -> Daemon:
    stream_r, stream_w = os.pipe()
    try:
        os.set_inheritable(stream_w, True)
    except Exception:
        pass
    # Larger pipe so tagged frames (3× size) do not stall START.
    try:
        import fcntl

        F_SETPIPE_SZ = getattr(fcntl, "F_SETPIPE_SZ", 1031)
        fcntl.fcntl(stream_r, F_SETPIPE_SZ, 1 << 20)
        fcntl.fcntl(stream_w, F_SETPIPE_SZ, 1 << 20)
    except Exception:
        pass

    env = os.environ.copy()
    env.pop("DFLASH_LEGACY_DAEMON", None)
    env.pop("DFLASH_KVFLASH", None)
    env["DFLASH27B_KV_TQ3"] = env.get("DFLASH27B_KV_TQ3", "1")
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
        "--stream-tagged",
    ]
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


def gate_overlap(d: Daemon, work: Path) -> None:
    print("== LIST_TARGET_CACHE_SLOTS ==")
    out = d.cmd("LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon] target_cache_slots=")
    assert "target_cache_slots=2" in out, out

    ids = [151644, 872, 198, 8948, 151645, 198, 151644, 77091, 198]
    p0 = work / "p0.raw"
    p1 = work / "p1.raw"
    write_raw_i32(p0, ids)
    write_raw_i32(p1, ids + [220, 220])

    print("== REQ 1 SLOT 0 START ==")
    out = d.cmd(f"REQ 1 SLOT 0 START {p0} 12 4", expect_prefix="ok START", timeout=300)
    assert out.startswith("ok START"), out
    frames0 = d.read_stream(idle=1.0)

    print("== REQ 2 SLOT 1 START ==")
    out = d.cmd(f"REQ 2 SLOT 1 START {p1} 12 4", expect_prefix="ok START", timeout=300)
    assert out.startswith("ok START"), out
    frames1 = d.read_stream(idle=1.0)

    print("== SCHED_DRAIN ==")
    out = d.cmd("SCHED_DRAIN", expect_prefix="ok SCHED_DRAIN", timeout=300)
    assert out.startswith("ok SCHED_DRAIN"), out
    frames = frames0 + frames1 + d.read_stream(max_frames=256, idle=2.0)

    tagged = [f for f in frames if f[0] == "tag"]
    req_ids = {f[2] for f in tagged if f[2] is not None}
    print(f"  tagged_tokens={len(tagged)} req_ids={sorted(req_ids)}")
    assert len(tagged) >= 4, f"expected tagged tokens from both START quanta, got {frames}"
    assert req_ids >= {1, 2}, f"expected both req ids demuxed, got {req_ids} frames={frames}"
    print("OVERLAP DEMUX OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="", help="Optional; omit for AR-only VRAM headroom")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--target-gpus", default="0,1")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    target = Path(args.target)
    draft = Path(args.draft) if args.draft else None
    assert bin_path.is_file(), bin_path
    assert target.is_file(), target

    with tempfile.TemporaryDirectory(prefix="phase3-m3b-") as td:
        work = Path(td)
        print("\n===== PHASE-3 M3b OVERLAP DEMUX =====")
        d = launch(
            bin_path,
            target,
            draft,
            slots=2,
            max_ctx=args.max_ctx,
            target_gpus=args.target_gpus,
            work=work,
        )
        try:
            gate_overlap(d, work)
        finally:
            d.close()

    print("\nALL PHASE-3 M3b GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
