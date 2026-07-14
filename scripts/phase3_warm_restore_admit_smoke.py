#!/usr/bin/env python3
"""Phase 3 gate: warm RESTORE_CHAIN + quantum admit (dual slots).

Side-binary layer-split smoke (stop model-runner-v4-lucebox first).

Protocol:
  SLOT k SNAPSHOT_THIN …          pin tool KV from live cache
  REQ id SLOT k RESTORE_CHAIN … total quantum
  → ok RESTORE_CHAIN … then ok RESTORE_CHAIN_ADMIT remaining=…
  SCHED_DRAIN completes both live requests via CONTINUE quanta

Compose stays N=1 until this + HTTP warm path are green.
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

_SCRIPTS = Path(__file__).resolve().parents[1] / "lucebox-patch/dflash/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from tagged_stream_demux import TaggedFrameBuffer  # noqa: E402

THIN_A = 4
THIN_B = 5
TOOL_IDS = [151644, 872, 198, 8948, 151645, 198]
TOOL_LEN = len(TOOL_IDS)


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
        self._parser = TaggedFrameBuffer()
        self._frames: list[tuple[str, int, int | None]] = []
        self._stdout_thr = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stderr_thr = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stream_thr = threading.Thread(target=self._drain_stream, daemon=True)
        self._stdout_q: list[str] = []
        self._stdout_cv = threading.Condition()
        self._stderr_thr.start()
        self._stdout_thr.start()
        self._stream_thr.start()
        self._await_ready()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))
            sys.stderr.write(f"[daemon.err] {line}")

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            sys.stderr.write(f"[daemon.out] {line}")
            with self._stdout_cv:
                self._stdout_q.append(line)
                self._stdout_cv.notify_all()

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
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"daemon exited early code={self.proc.returncode}\n"
                    + "\n".join(self.stderr_lines[-40:])
                )
            with self._stdout_cv:
                self._stdout_cv.wait(timeout=1.0)
                while self._stdout_q:
                    line = self._stdout_q.pop(0)
                    low = line.lower()
                    if "[daemon] ready" in low or "target_cache_slots=" in low:
                        return
        raise TimeoutError(
            "daemon ready banner timeout\n" + "\n".join(self.stderr_lines[-40:])
        )

    def cmd(self, line: str, expect_prefix: str | None = None, timeout: float = 120.0) -> str:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.await_stdout(expect_prefix=expect_prefix, timeout=timeout)

    def await_stdout(self, expect_prefix: str | None = None, timeout: float = 120.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("daemon died while waiting for stdout")
            with self._stdout_cv:
                self._stdout_cv.wait(timeout=1.0)
                while self._stdout_q:
                    out = self._stdout_q.pop(0)
                    if expect_prefix is None or out.startswith(expect_prefix) or out.startswith("err "):
                        return out.rstrip("\n")
        raise TimeoutError(f"timeout waiting for stdout prefix={expect_prefix!r}")

    def read_stream(self, max_frames: int = 256, idle: float = 0.5):
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


def _pin_tools(d: Daemon, work: Path, live_slot: int, thin_slot: int) -> None:
    prompt = work / f"pin{live_slot}.bin"
    write_counted_i32(prompt, TOOL_IDS)
    out = work / f"pin{live_slot}.out"
    print(f"== SLOT {live_slot} generate (seed for SNAPSHOT_THIN) ==")
    d.cmd(f"SLOT {live_slot} generate {prompt} 1 {out}", expect_prefix="ok ", timeout=300)
    d.read_stream(idle=0.3)
    print(f"== SLOT {live_slot} SNAPSHOT_THIN {thin_slot} ==")
    thin = d.cmd(
        f"SLOT {live_slot} SNAPSHOT_THIN {thin_slot} 0 {TOOL_LEN}",
        expect_prefix="[snap] thin",
        timeout=120,
    )
    assert f"thin slot={thin_slot}" in thin, thin


def gate_warm_admit(d: Daemon, work: Path) -> None:
    print("== LIST_TARGET_CACHE_SLOTS ==")
    out = d.cmd("LIST_TARGET_CACHE_SLOTS", expect_prefix="[daemon] target_cache_slots=")
    assert "target_cache_slots=2" in out, out

    _pin_tools(d, work, live_slot=0, thin_slot=THIN_A)
    _pin_tools(d, work, live_slot=1, thin_slot=THIN_B)

    suffix = [220, 220, 1110, 264]
    p0 = work / "r0.raw"
    p1 = work / "r1.raw"
    write_raw_i32(p0, TOOL_IDS + suffix)
    write_raw_i32(p1, TOOL_IDS + suffix + [220])

    print("== REQ 1 SLOT 0 RESTORE_CHAIN … quantum ==")
    out = d.cmd(
        f"REQ 1 SLOT 0 RESTORE_CHAIN -1 {THIN_A} {p0} 12 4",
        expect_prefix="ok ",
        timeout=300,
    )
    assert out.startswith("ok "), out
    admit0 = out if "RESTORE_CHAIN_ADMIT" in out else d.await_stdout(
        expect_prefix="ok RESTORE_CHAIN_ADMIT", timeout=30,
    )
    assert "RESTORE_CHAIN_ADMIT" in admit0 and "req=1" in admit0 and "remaining=" in admit0, admit0
    frames0 = d.read_stream(idle=1.0)

    print("== REQ 2 SLOT 1 RESTORE_CHAIN … quantum ==")
    out = d.cmd(
        f"REQ 2 SLOT 1 RESTORE_CHAIN -1 {THIN_B} {p1} 12 4",
        expect_prefix="ok ",
        timeout=300,
    )
    assert out.startswith("ok "), out
    admit1 = out if "RESTORE_CHAIN_ADMIT" in out else d.await_stdout(
        expect_prefix="ok RESTORE_CHAIN_ADMIT", timeout=30,
    )
    assert "RESTORE_CHAIN_ADMIT" in admit1 and "req=2" in admit1 and "remaining=" in admit1, admit1
    frames1 = d.read_stream(idle=1.0)

    print("== SCHED_DRAIN ==")
    drain = d.cmd("SCHED_DRAIN", expect_prefix="ok SCHED_DRAIN", timeout=300)
    assert drain.startswith("ok SCHED_DRAIN"), drain
    frames = frames0 + frames1 + d.read_stream(max_frames=256, idle=2.0)
    tagged = [f for f in frames if f[0] == "tag"]
    req_ids = {f[2] for f in tagged if f[2] is not None}
    print(f"  tagged_tokens={len(tagged)} req_ids={sorted(req_ids)}")
    assert len(tagged) >= 4, f"expected tagged tokens, got {frames}"
    assert req_ids >= {1, 2}, f"expected both req ids, got {req_ids} frames={frames}"
    print("WARM RESTORE_CHAIN ADMIT OK")


def gate_large_max_tokens_no_phantom_drain(d: Daemon, work: Path) -> None:
    """Agent-shaped max_tokens must not yield SCHED_DRAIN steps≈remaining.

    Repro for the N=2 truncation bug: first quantum ends (often EOS) while
    remaining keeps a huge leftover; SCHED burnt remaining 1-by-1 in ~100ms.
    """
    import re

    print("== large max_tokens admit (no phantom SCHED) ==")
    _pin_tools(d, work, live_slot=0, thin_slot=THIN_A)
    # Short assistant-bound suffix so the first quantum often hits EOS.
    suffix = [220, 220, 1110, 264, 198]
    p = work / "eos_budget.raw"
    write_raw_i32(p, TOOL_IDS + suffix)
    huge = 64000
    quantum = 8
    out = d.cmd(
        f"REQ 9 SLOT 0 RESTORE_CHAIN -1 {THIN_A} {p} {huge} {quantum}",
        expect_prefix="ok ",
        timeout=300,
    )
    assert out.startswith("ok "), out
    admit = out if "RESTORE_CHAIN_ADMIT" in out else d.await_stdout(
        expect_prefix="ok RESTORE_CHAIN_ADMIT", timeout=30,
    )
    assert "RESTORE_CHAIN_ADMIT" in admit and "remaining=" in admit, admit
    m_rem = re.search(r"remaining=(\d+)", admit)
    assert m_rem, admit
    remaining = int(m_rem.group(1))
    print(f"  admit: {admit.strip()} → remaining={remaining}")
    d.read_stream(idle=0.5)

    t0 = time.time()
    drain = d.cmd("SCHED_DRAIN", expect_prefix="ok SCHED_DRAIN", timeout=300)
    elapsed = time.time() - t0
    assert drain.startswith("ok SCHED_DRAIN"), drain
    m_steps = re.search(r"steps=(\d+)", drain)
    assert m_steps, drain
    steps = int(m_steps.group(1))
    print(f"  SCHED_DRAIN steps={steps} elapsed={elapsed:.3f}s")
    if remaining == 0:
        assert steps <= 16, (
            f"remaining=0 but SCHED_DRAIN still spun steps={steps} ({drain})"
        )
    else:
        # Real decode of leftover tokens can't finish in a flash at ~32k steps.
        assert steps < 512 or elapsed >= 2.0, (
            f"phantom drain suspected: remaining={remaining} steps={steps} "
            f"elapsed={elapsed:.3f}s ({drain})"
        )
        assert steps < remaining, (
            f"SCHED burned more steps than remaining budget "
            f"({steps} >= {remaining})"
        )
    print("LARGE max_tokens NO-PHANTOM-DRAIN OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", default="")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--target-gpus", default="0,1")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    target = Path(args.target)
    draft = Path(args.draft) if args.draft else None
    assert bin_path.is_file(), bin_path
    assert target.is_file(), target

    with tempfile.TemporaryDirectory(prefix="phase3-warm-") as td:
        work = Path(td)
        print("\n===== PHASE-3 WARM RESTORE_CHAIN ADMIT =====")
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
            gate_warm_admit(d, work)
            gate_large_max_tokens_no_phantom_drain(d, work)
        finally:
            d.close()

    print("\nALL PHASE-3 WARM ADMIT GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
