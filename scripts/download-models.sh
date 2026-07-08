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

if [[ ! -f "${PFLASH_DRAFTER}" ]]; then
  echo "Downloading PFlash prefill drafter (Qwen3-0.6B BF16 GGUF)..."
  hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf \
    --local-dir "${PFLASH_DIR}"
fi

MMPROJ="${BASE}/mmproj-F16.gguf"
MTP_TARGET="${BASE}/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf"
MTP_DRAFT="${BASE}/dflash-q4km/Qwen3.6-27B-DFlash-Q4_K_M.gguf"
mkdir -p "${BASE}/unsloth-mtp-q4km" "${BASE}/dflash-q4km"

if [[ ! -f "${MMPROJ}" ]]; then
  echo "Downloading vision mmproj (unsloth/Qwen3.6-27B-GGUF)..."
  hf download unsloth/Qwen3.6-27B-GGUF mmproj-F16.gguf --local-dir "${BASE}"
fi

if [[ ! -f "${MTP_TARGET}" ]]; then
  echo "Downloading BeeLlama vision target (MTP Q4_K_M)..."
  hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-Q4_K_M.gguf \
    --local-dir "${BASE}/unsloth-mtp-q4km"
fi

if [[ ! -f "${MTP_DRAFT}" ]]; then
  echo "Downloading BeeLlama DFlash draft (Anbeeld)..."
  hf download Anbeeld/Qwen3.6-27B-DFlash-GGUF Qwen3.6-27B-DFlash-Q4_K_M.gguf \
    --local-dir "${BASE}/dflash-q4km"
fi

echo "Shared cache ready under ${MODEL_DIR}"
ls -lh "${TARGET}" "${DRAFT}" "${PFLASH_DRAFTER}" "${MMPROJ}" "${MTP_TARGET}" "${MTP_DRAFT}" 2>/dev/null || \
  ls -lh "${TARGET}" "${DRAFT}" "${PFLASH_DRAFTER}"
