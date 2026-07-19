#!/usr/bin/env python3
"""Phase 0 scorecard: RESTORE_CHAIN suffix / prefill percentiles.

Reads lucebox daemon logs (stdin or files) and reports p50/p90 for
``suffix_n``, ``prefill_s``, implied tok/s, plus ``thick=-1`` rate.

Example (on ai.local host or via SSH):

  docker logs model-runner-v4-lucebox --since 48h 2>&1 \\
    | python3 scripts/prefill_suffix_scorecard.py

  python3 scripts/prefill_suffix_scorecard.py --json /tmp/score.json \\
    --check-gates < logs.txt

See docs/prefill-suffix-first-plan.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

RESTORE_RE = re.compile(
    r"\[restore-chain\].*?\bthick=(-?\d+)\b.*?\bsuffix_n=(-?\d+)\b.*?\bprefill_s=([0-9.]+)\b"
)

# Near-term Phase A exit criteria (docs/prefill-suffix-first-plan.md §6).
PHASE_A_GATES = {
    "suffix_n_p50_max": 200,
    "suffix_n_p90_max": 1000,
    "prefill_s_p50_max": 2.0,
    "prefill_s_p90_max": 8.0,
    "thick_minus1_pct_max": 2.0,
    "min_samples": 200,
}

FOLLOW_LT_8K = 8000


@dataclass(frozen=True)
class Sample:
    thick: int
    suffix_n: int
    prefill_s: float

    @property
    def tok_s(self) -> float | None:
        if self.prefill_s < 0.05 or self.suffix_n < 1:
            return None
        return self.suffix_n / self.prefill_s


def percentile(xs: Sequence[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    # Nearest-rank on [0, n-1] (matches prior ad-hoc baseline).
    idx = int(round((p / 100.0) * (len(ys) - 1)))
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


def parse_lines(lines: Iterable[str]) -> list[Sample]:
    out: list[Sample] = []
    for line in lines:
        m = RESTORE_RE.search(line)
        if not m:
            continue
        thick = int(m.group(1))
        suffix_n = int(m.group(2))
        prefill_s = float(m.group(3))
        if suffix_n < 0 or prefill_s < 0:
            continue
        out.append(Sample(thick=thick, suffix_n=suffix_n, prefill_s=prefill_s))
    return out


def _stat_block(xs: Sequence[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "max": float(max(xs)),
        "mean": float(sum(xs) / len(xs)),
    }


def build_report(samples: Sequence[Sample]) -> dict:
    thick_m1 = sum(1 for s in samples if s.thick < 0)
    n = len(samples)
    suffix = [float(s.suffix_n) for s in samples]
    pref = [s.prefill_s for s in samples]
    rates = [s.tok_s for s in samples if s.tok_s is not None]
    follow = [float(s.suffix_n) for s in samples if s.suffix_n < FOLLOW_LT_8K]
    coldish = [float(s.suffix_n) for s in samples if s.suffix_n >= FOLLOW_LT_8K]
    return {
        "samples": n,
        "thick_minus1": thick_m1,
        "thick_minus1_pct": (100.0 * thick_m1 / n) if n else 0.0,
        "suffix_n": _stat_block(suffix),
        "prefill_s": _stat_block(pref),
        "tok_s": _stat_block([float(x) for x in rates]),
        "suffix_n_lt_8k": _stat_block(follow),
        "suffix_n_ge_8k": _stat_block(coldish),
        "gates_phase_a": PHASE_A_GATES,
    }


def check_gates(report: dict) -> list[str]:
    fails: list[str] = []
    n = int(report["samples"])
    if n < PHASE_A_GATES["min_samples"]:
        fails.append(
            f"samples={n} < min_samples={PHASE_A_GATES['min_samples']}"
        )
    sn = report["suffix_n"]
    ps = report["prefill_s"]
    if sn.get("p50") is not None and sn["p50"] > PHASE_A_GATES["suffix_n_p50_max"]:
        fails.append(
            f"suffix_n p50={sn['p50']:.0f} > {PHASE_A_GATES['suffix_n_p50_max']}"
        )
    if sn.get("p90") is not None and sn["p90"] > PHASE_A_GATES["suffix_n_p90_max"]:
        fails.append(
            f"suffix_n p90={sn['p90']:.0f} > {PHASE_A_GATES['suffix_n_p90_max']}"
        )
    if ps.get("p50") is not None and ps["p50"] > PHASE_A_GATES["prefill_s_p50_max"]:
        fails.append(
            f"prefill_s p50={ps['p50']:.3f} > {PHASE_A_GATES['prefill_s_p50_max']}"
        )
    if ps.get("p90") is not None and ps["p90"] > PHASE_A_GATES["prefill_s_p90_max"]:
        fails.append(
            f"prefill_s p90={ps['p90']:.3f} > {PHASE_A_GATES['prefill_s_p90_max']}"
        )
    if report["thick_minus1_pct"] > PHASE_A_GATES["thick_minus1_pct_max"]:
        fails.append(
            f"thick=-1 {report['thick_minus1_pct']:.1f}% > "
            f"{PHASE_A_GATES['thick_minus1_pct_max']}%"
        )
    return fails


def _fmt_stat(name: str, block: dict) -> str:
    if block.get("n", 0) == 0:
        return f"  {name}: n=0"
    return (
        f"  {name}: n={block['n']}  p50={block['p50']:.4g}  "
        f"p90={block['p90']:.4g}  p95={block['p95']:.4g}  "
        f"max={block['max']:.4g}  mean={block['mean']:.4g}"
    )


def print_report(report: dict, gate_fails: list[str] | None) -> None:
    print("prefill suffix scorecard")
    print(f"  samples:          {report['samples']}")
    print(
        f"  thick=-1:         {report['thick_minus1']} "
        f"({report['thick_minus1_pct']:.1f}%)"
    )
    print(_fmt_stat("suffix_n", report["suffix_n"]))
    print(_fmt_stat("prefill_s", report["prefill_s"]))
    print(_fmt_stat("tok_s", report["tok_s"]))
    print(_fmt_stat("suffix_n (<8k)", report["suffix_n_lt_8k"]))
    print(_fmt_stat("suffix_n (≥8k)", report["suffix_n_ge_8k"]))
    if gate_fails is None:
        return
    print("phase A gates:")
    if not gate_fails:
        print("  PASS")
    else:
        print("  FAIL")
        for f in gate_fails:
            print(f"    - {f}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        help="Log files (default: stdin)",
    )
    ap.add_argument(
        "--json",
        metavar="PATH",
        help="Write full report JSON to PATH (- for stdout after text)",
    )
    ap.add_argument(
        "--check-gates",
        action="store_true",
        help="Compare against Phase A near-term exit criteria; exit 1 on fail",
    )
    args = ap.parse_args(argv)

    if args.paths:
        chunks: list[str] = []
        for p in args.paths:
            chunks.extend(
                Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
            )
        samples = parse_lines(chunks)
    else:
        samples = parse_lines(sys.stdin)

    report = build_report(samples)
    gate_fails = check_gates(report) if args.check_gates else None
    print_report(report, gate_fails)

    if args.json:
        payload = dict(report)
        if gate_fails is not None:
            payload["gate_fails"] = gate_fails
            payload["gates_pass"] = not gate_fails
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.json == "-":
            sys.stdout.write(text)
        else:
            Path(args.json).write_text(text, encoding="utf-8")

    if args.check_gates and gate_fails:
        return 1
    if report["samples"] == 0:
        print("error: no [restore-chain] samples found", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
