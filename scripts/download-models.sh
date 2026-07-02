#!/usr/bin/env bash
# Download shared Qwen3.6 weights into MODEL_DIR (default: ../models-cache).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
BASE="${MODEL_DIR}/qwen3.6-27b-gguf"

pip install -q 'huggingface-hub>=0.26,<1'

mkdir -p "${BASE}/unsloth-q4km" "${BASE}/dflash-lucebox"

# Lucebox needs the standard GGUF (not the MTP variant used by BeeLlama v3).
TARGET="${BASE}/unsloth-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT="${BASE}/dflash-lucebox/dflash-draft-3.6-q4_k_m.gguf"

if [[ ! -f "${TARGET}" ]]; then
  echo "Downloading Lucebox target (unsloth/Qwen3.6-27B-GGUF, non-MTP)..."
  hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf \
    --local-dir "${BASE}/unsloth-q4km"
fi

if [[ ! -f "${DRAFT}" ]]; then
  echo "Downloading Lucebox DFlash draft..."
  hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf \
    --local-dir "${BASE}/dflash-lucebox"
fi

PFLASH_DIR="${BASE}/pflash"
PFLASH_DRAFTER="${PFLASH_DIR}/Qwen3-0.6B-BF16.gguf"
mkdir -p "${PFLASH_DIR}"

if [[ ! -f "${PFLASH_DRAFTER}" ]]; then
  echo "Downloading PFlash prefill drafter (Qwen3-0.6B BF16 GGUF)..."
  hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf \
    --local-dir "${PFLASH_DIR}"
fi

echo "Shared cache ready under ${MODEL_DIR}"
ls -lh "${TARGET}" "${DRAFT}" "${PFLASH_DRAFTER}"
