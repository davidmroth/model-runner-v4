#!/usr/bin/env python3
"""Triage helper: classify [pc] lookup miss / thick=-1 patterns from logs."""
from __future__ import annotations

import re
import sys
from collections import Counter

SCOPE_RE = re.compile(r"scope='([^']+)'")
RESTORE_RE = re.compile(
    r"\[restore-chain\].*?\bthick=(-?\d+)\b.*?\bsuffix_n=(-?\d+)\b.*?\bprefill_s=([0-9.]+)\b"
)


def main() -> int:
    lines = sys.stdin.read().splitlines()
    events: list[tuple[str, str, str, str]] = []
    for line in lines:
        if "[pc] lookup miss" in line:
            m = SCOPE_RE.search(line)
            reason_m = re.search(r"reason=(\S+)", line)
            reason = reason_m.group(1) if reason_m else "?"
            events.append(("miss", m.group(1) if m else "?", reason, line))
        elif "[pc] lookup hit" in line:
            m = SCOPE_RE.search(line)
            events.append(("hit", m.group(1) if m else "?", "", line))
        elif "inline-snap committed" in line:
            m = SCOPE_RE.search(line)
            events.append(("commit", m.group(1) if m else "?", "", line))
        elif "deferred conv snap failed" in line:
            events.append(("def_fail", "?", "", line))
        elif "deferred conv snap skipped" in line:
            events.append(("def_skip", "?", "", line))
        elif "prepare_inline_snap blocked" in line:
            events.append(("prep_blocked", "?", "", line))
        elif "want=" in line and "have=" in line and "evicting" in line:
            events.append(("scope_evict", "?", "", line[:200]))
        elif "key_cut=" in line and "evicting" in line:
            events.append(("cut_evict", "?", "", line[:200]))

    seen_commit: set[str] = set()
    seen_hit: set[str] = set()
    miss_first = miss_after_commit = miss_cron = 0
    miss_reasons: Counter[str] = Counter()
    after_commit_scopes: Counter[str] = Counter()
    for kind, scope, reason, _line in events:
        if kind == "commit":
            seen_commit.add(scope)
        elif kind == "hit":
            seen_hit.add(scope)
        elif kind == "miss":
            miss_reasons[reason or "?"] += 1
            if scope.startswith("cron_"):
                miss_cron += 1
            if scope in seen_commit:
                miss_after_commit += 1
                after_commit_scopes[scope] += 1
            else:
                miss_first += 1

    print("=== lookup miss classification ===")
    print(f"lookup_miss total: {sum(1 for k, _, _, _ in events if k == 'miss')}")
    print(f"  first-seen scope (cold): {miss_first}")
    print(f"  AFTER prior commit (evict/thrash?): {miss_after_commit}")
    print(f"  cron scopes among misses: {miss_cron}")
    if miss_reasons:
        print(f"  reasons: {dict(miss_reasons)}")
    print(f"commits: {sum(1 for k,_,_,_ in events if k=='commit')} "
          f"unique={len({s for k,s,_,_ in events if k=='commit'})}")
    print(f"hits: {sum(1 for k,_,_,_ in events if k=='hit')} "
          f"unique={len({s for k,s,_,_ in events if k=='hit'})}")
    print(f"scope_evict lines: {sum(1 for k,_,_,_ in events if k=='scope_evict')}")
    print(f"cut_evict lines: {sum(1 for k,_,_,_ in events if k=='cut_evict')}")
    print(f"deferred snap failed: {sum(1 for k,_,_,_ in events if k=='def_fail')}")
    print(f"deferred snap skipped: {sum(1 for k,_,_,_ in events if k=='def_skip')}")
    print(f"prepare blocked: {sum(1 for k,_,_,_ in events if k=='prep_blocked')}")
    if after_commit_scopes:
        print("top miss-after-commit scopes:")
        for s, c in after_commit_scopes.most_common(8):
            print(f"  {c}x {s[:90]}")

    buckets: Counter[str] = Counter()
    thick_ok = thick_m1 = 0
    for line in lines:
        m = RESTORE_RE.search(line)
        if not m:
            continue
        t, s = int(m.group(1)), int(m.group(2))
        if t < 0:
            thick_m1 += 1
            continue
        thick_ok += 1
        if s < 100:
            buckets["0-99"] += 1
        elif s < 300:
            buckets["100-299"] += 1
        elif s < 600:
            buckets["300-599"] += 1
        elif s < 1200:
            buckets["600-1199"] += 1
        elif s < 3000:
            buckets["1200-2999"] += 1
        else:
            buckets["3000+"] += 1
    print("=== thick>=0 suffix buckets ===")
    print(f"n_ok={thick_ok} n_m1={thick_m1} buckets={dict(buckets)}")

    # Startup cap lines
    for line in lines[:200]:
        if "prefix-cache" in line or "prefix_cache" in line or "full-cache enabled" in line:
            print("startup:", line[:200])
        if "prefix-cache-slots" in line or "Prefix cache" in line:
            print("startup:", line[:200])
    for line in lines:
        if "[pc] full-cache enabled" in line or "[pc] chat markers" in line:
            print("pc:", line[:220])
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
