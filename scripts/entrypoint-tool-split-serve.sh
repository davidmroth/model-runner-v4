#!/usr/bin/env bash
# Run Python server_tools.py with tool-split + PFlash (test_dflash daemon backend).
# Activated when DFLASH_TOOL_SPLIT_ENABLED=1 (see entrypoint-dual-gpu.sh).
set -euo pipefail

PATCH_SCRIPTS="${DFLASH_PATCH_SCRIPTS:-/opt/lucebox-hub/patch/dflash/scripts}"
TARGET="${DFLASH_TARGET:-/opt/lucebox-hub/server/models/qwen3.6-27b-gguf/unsloth-q4km/Qwen3.6-27B-Q4_K_M.gguf}"
DRAFT="${DFLASH_DRAFT:-}"
BIN="${DFLASH_DAEMON_BIN:-/opt/lucebox-hub/server/build/test_dflash}"
MAX_CTX="${DFLASH_MAX_CTX:-16384}"
HOST="${DFLASH_HOST:-0.0.0.0}"
PORT="${DFLASH_PORT:-8080}"

echo "[tool-split-serve] patch scripts: ${PATCH_SCRIPTS}"
echo "[tool-split-serve] target: ${TARGET}"
echo "[tool-split-serve] draft: ${DRAFT}"
echo "[tool-split-serve] daemon bin: ${BIN}"
echo "[tool-split-serve] layer_split=${DFLASH_LAYER_SPLIT:-0} tool_split=${DFLASH_TOOL_SPLIT_ENABLED:-0}"

if [ ! -f "${PATCH_SCRIPTS}/server_tools.py" ]; then
  echo "[tool-split-serve] ERROR: ${PATCH_SCRIPTS}/server_tools.py missing (mount lucebox patch?)" >&2
  exit 1
fi

# FastAPI stack for server_tools (not bundled in stock serve path).
uv pip install -q "fastapi>=0.115,<1" "uvicorn>=0.32,<1" 2>/dev/null || true

ARGS=(
  "${PATCH_SCRIPTS}/server_tools.py"
  --host "${HOST}"
  --port "${PORT}"
  --target "${TARGET}"
  --bin "${BIN}"
  --max-ctx "${MAX_CTX}"
  --budget "${DFLASH_DDTREE_BUDGET:-22}"
  --prefix-cache-slots "${DFLASH_PREFIX_CACHE_SLOTS:-1}"
  --prefill-cache-slots "${DFLASH_PREFILL_CACHE_SLOTS:-2}"
  --tool-split auto
  --tool-split-profile "${DFLASH_TOOL_SPLIT_PROFILE:-auto}"
  --tool-split-pinned-slots "${DFLASH_TOOL_SPLIT_PINNED_SLOTS:-2}"
)

if [ -n "${DRAFT}" ] && [ -f "${DRAFT}" ]; then
  ARGS+=(--draft "${DRAFT}")
fi

# Keep an F32 mirror of the draft's target-feature window on the draft GPU.
# Without it, every decode step re-copies the whole 4096-row BF16 feature
# window across GPUs through the CPU (~700ms/step at 8K+ context).
if [ "${DFLASH_DRAFT_FEATURE_MIRROR:-1}" = "1" ]; then
  ARGS+=(--draft-feature-mirror)
fi

if [ "${DFLASH_PREFILL_MODE:-off}" != "off" ]; then
  ARGS+=(
    --prefill-compression "${DFLASH_PREFILL_MODE}"
    --prefill-threshold "${DFLASH_PREFILL_THRESHOLD:-3000}"
    --prefill-keep-ratio "${DFLASH_PREFILL_KEEP:-0.10}"
  )
  if [ -n "${DFLASH_PREFILL_DRAFTER:-}" ]; then
    ARGS+=(--prefill-drafter "${DFLASH_PREFILL_DRAFTER}")
  fi
fi

if [ "${DFLASH_LAYER_SPLIT:-0}" = "1" ]; then
  ARGS+=(--target-gpus "${DFLASH_TARGET_DEVICES:-cuda:0,cuda:1}")
  if [ -n "${DFLASH_TARGET_LAYER_SPLIT:-}" ]; then
    ARGS+=(--target-layer-split "${DFLASH_TARGET_LAYER_SPLIT}")
  else
    ARGS+=(--target-layer-split)
  fi
  [ "${DFLASH_PEER_ACCESS:-0}" = "1" ] && ARGS+=(--peer-access)
else
  [ -n "${DFLASH_TARGET_GPU:-}" ] && ARGS+=(--target-gpu "${DFLASH_TARGET_GPU}")
  [ -n "${DFLASH_DRAFT_GPU:-}" ] && ARGS+=(--draft-gpu "${DFLASH_DRAFT_GPU}")
  [ "${DFLASH_PEER_ACCESS:-0}" = "1" ] && ARGS+=(--peer-access)
fi

if [ "${DFLASH_TOOL_SPLIT_COMPRESS_CONV:-1}" = "0" ]; then
  ARGS+=(--no-tool-split-compress-conv)
fi

if [ -n "${DFLASH_TOOL_SPLIT_PLUGIN_DIR:-}" ]; then
  ARGS+=(--tool-split-plugin-dir "${DFLASH_TOOL_SPLIT_PLUGIN_DIR}")
fi

export PYTHONPATH="${PATCH_SCRIPTS}${PYTHONPATH:+:${PYTHONPATH}}"

# server_tools.py speaks the legacy inline daemon protocol (SNAPSHOT_THIN with
# kv ranges, RESTORE_CHAIN, "[snap] inline" acks). Keep test_dflash on the
# legacy loop instead of run_qwen35_daemon.
export DFLASH_LEGACY_DAEMON="${DFLASH_LEGACY_DAEMON:-1}"

# The daemon binary links shared ggml libs from its build tree; its RUNPATH
# points at the build container's /src path, which doesn't exist here. Put the
# mounted build's ggml dirs first so the patched libs win over any stock copy.
BUILD_DIR="$(cd "$(dirname "${BIN}")" && pwd)"
GGML_DIR="${BUILD_DIR}/deps/llama.cpp/ggml/src"
if [ -d "${GGML_DIR}" ]; then
  export LD_LIBRARY_PATH="${GGML_DIR}:${GGML_DIR}/ggml-cuda${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${PATCH_SCRIPTS}"

exec uv run --no-sync --directory /opt/lucebox-hub python "${ARGS[@]}"
