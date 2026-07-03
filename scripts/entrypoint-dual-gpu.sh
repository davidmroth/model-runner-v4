#!/usr/bin/env bash
# Wrap lucebox-hub entrypoint.sh and inject multi-GPU flags before dflash_server starts.
#
# Modes (DFLASH_LAYER_SPLIT=1 default for dual-GPU):
#   layer-split — 27B target sharded across GPUs (--target-devices + --target-layer-split)
#   dual-device — full target on GPU0, DFlash draft on GPU1 (legacy)

set -euo pipefail

# Tool-split path uses Python server_tools + test_dflash daemon (RESTORE_CHAIN / SNAPSHOT_THIN).
if [ "${DFLASH_TOOL_SPLIT_ENABLED:-0}" = "1" ]; then
  exec /scripts/entrypoint-tool-split-serve.sh "$@"
fi

STOCK="/opt/lucebox-hub/server/scripts/entrypoint.sh"
PATCHED="/tmp/lucebox-entrypoint-dual-gpu.sh"
MARKER="# model-runner-v4 dual-gpu patch v2"

if [ ! -f "$PATCHED" ] || ! grep -qF "$MARKER" "$PATCHED"; then
  awk -v marker="$MARKER" '
    /^exec "\$\{CMD\[@\]\}"/ && !done {
      print marker
      print "if [ \"${DFLASH_LAYER_SPLIT:-0}\" = \"1\" ]; then"
      print "  CMD+=(--target-devices \"${DFLASH_TARGET_DEVICES:-cuda:0,cuda:1}\")"
      print "  CMD+=(--target-layer-split \"${DFLASH_TARGET_LAYER_SPLIT:-32,32}\")"
      print "  CMD+=(--peer-access)"
      print "  if [ -n \"${DFLASH_DRAFT_GPU:-}\" ] && [ -n \"${DRAFT_ARG}\" ]; then"
      print "    CMD+=(--draft-device \"cuda:${DFLASH_DRAFT_GPU}\")"
      print "  fi"
      print "else"
      print "  [ -n \"${DFLASH_TARGET_GPU:-}\" ] && CMD+=(--target-device \"cuda:${DFLASH_TARGET_GPU}\")"
      print "  [ -n \"${DFLASH_DRAFT_GPU:-}\" ] && [ -n \"${DRAFT_ARG}\" ] && CMD+=(--draft-device \"cuda:${DFLASH_DRAFT_GPU}\")"
      print "  [ \"${DFLASH_PEER_ACCESS:-0}\" = \"1\" ] && CMD+=(--peer-access)"
      print "fi"
      done=1
    }
    { print }
  ' "$STOCK" > "$PATCHED"
  chmod +x "$PATCHED"
fi

exec "$PATCHED" "$@"
