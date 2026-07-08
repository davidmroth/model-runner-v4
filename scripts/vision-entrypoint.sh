#!/bin/sh
# BeeLlama DFlash + mmproj vision sidecar (GPU 1 by default).
set -eu

HOST="${VISION_HOST:-0.0.0.0}"
PORT="${VISION_PORT:-8081}"
MODEL_ROOT="${VISION_MODEL_ROOT:-/opt/lucebox-hub/server/models/qwen3.6-27b-gguf}"
MODEL="${VISION_GGUF:-${MODEL_ROOT}/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf}"
MMPROJ="${VISION_MMPROJ:-${MODEL_ROOT}/mmproj-F16.gguf}"
DRAFT="${VISION_DFLASH_DRAFT:-${MODEL_ROOT}/dflash-q4km/Qwen3.6-27B-DFlash-Q4_K_M.gguf}"
CTX="${VISION_CTX_SIZE:-16384}"
BATCH="${VISION_BATCH_SIZE:-2048}"
UBATCH="${VISION_UBATCH_SIZE:-512}"
KV="${VISION_KV_TYPE:-q4_0}"

if [ ! -f "${MODEL}" ]; then
  echo "[vision] missing target GGUF: ${MODEL}" >&2
  exit 1
fi
if [ ! -f "${MMPROJ}" ]; then
  echo "[vision] missing mmproj: ${MMPROJ}" >&2
  exit 1
fi
if [ ! -f "${DRAFT}" ]; then
  echo "[vision] missing DFlash draft: ${DRAFT}" >&2
  exit 1
fi

echo "[vision] target=${MODEL}"
echo "[vision] mmproj=${MMPROJ}"
echo "[vision] draft=${DRAFT}"
echo "[vision] cuda_visible=${NVIDIA_VISIBLE_DEVICES:-all}"

# shellcheck disable=SC2086
exec /app/llama-server \
  --host "${HOST}" \
  --port "${PORT}" \
  -m "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --no-mmproj-offload \
  --image-min-tokens "${IMAGE_MIN_TOKENS:-1024}" \
  --image-max-tokens "${IMAGE_MAX_TOKENS:-1024}" \
  -c "${CTX}" \
  -b "${BATCH}" \
  -ub "${UBATCH}" \
  -ngl 99 \
  -np 1 \
  --cache-type-k "${KV}" \
  --cache-type-v "${KV}" \
  --jinja \
  --reasoning off \
  --reasoning-format deepseek \
  --spec-draft-model "${DRAFT}" \
  --spec-type dflash \
  --spec-dflash-cross-ctx "${DFLASH_CROSS_CTX:-512}" \
  --kv-unified \
  --spec-draft-ngl 99 \
  --flash-attn on \
  --fit off \
  --cache-prompt
