#!/usr/bin/env python3
"""Add stderr diagnostics to prefix_snapshot_save failure path in
qwen35_target_shard_ipc_daemon.cpp so we can see exactly why the thick snap
fails during the deferred conv snap job."""
import pathlib, sys

path = pathlib.Path(
    "/media/data/projects/lucebox-hub-src/server/src/qwen35/"
    "qwen35_target_shard_ipc_daemon.cpp"
)
src = path.read_text()

# Match on a simpler, unambiguous anchor: the two lines just after the for-loop
OLD = (
    '                if (ok) {\n'
    '                    snapshot_logits[(size_t)slot] = prefill_last_logits;\n'
    '                } else {\n'
    '                    free_prefix_slot(slot);\n'
    '                }'
)

NEW = (
    '                if (ok) {\n'
    '                    snapshot_logits[(size_t)slot] = prefill_last_logits;\n'
    '                } else {\n'
    '                    const int cur_pos_diag = shards.empty() ? -1\n'
    '                        : shards[0].cache.cur_pos;\n'
    '                    std::fprintf(stderr,\n'
    '                        "[snap] prefix_snapshot_save slot=%d FAILED "\n'
    '                        "cur_pos=%d shards=%zu snap_backends=%zu err=%s\\n",\n'
    '                        slot, cur_pos_diag, shards.size(),\n'
    '                        snapshot_backends.size(),\n'
    '                        dflash27b_last_error());\n'
    '                    std::fflush(stderr);\n'
    '                    free_prefix_slot(slot);\n'
    '                }'
)

count = src.count(OLD)
if count != 1:
    print(f"ERROR: expected 1 occurrence, found {count}")
    sys.exit(1)

backup = path.with_suffix(".cpp.bak2")
backup.write_text(src)
path.write_text(src.replace(OLD, NEW, 1))
print(f"OK: patched {path}")
print(f"    backup -> {backup}")
