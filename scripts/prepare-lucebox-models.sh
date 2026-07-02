#!/usr/bin/env bash
# Build Lucebox Hub flat layout as symlinks into the shared models-cache.
set -euo pipefail

CACHE="${MODEL_DIR:-/models}"
HUB="${LUCEBOX_MODEL_DIR:-${CACHE}/lucebox-hub}"

TARGET_REL="qwen3.6-27b-gguf/unsloth-mtp-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_REL="qwen3.6-27b-gguf/dflash-lucebox/dflash-draft-3.6-q4_k_m.gguf"

if [[ ! -f "${CACHE}/${TARGET_REL}" ]]; then
  echo "Missing target model: ${CACHE}/${TARGET_REL}" >&2
  echo "Run: docker compose --profile download run --rm download" >&2
  exit 1
fi
if [[ ! -f "${CACHE}/${DRAFT_REL}" ]]; then
  echo "Missing Lucebox draft: ${CACHE}/${DRAFT_REL}" >&2
  exit 1
fi

mkdir -p "${HUB}/draft"
ln -sfn "../${TARGET_REL}" "${HUB}/Qwen3.6-27B-Q4_K_M.gguf"
ln -sfn "../${DRAFT_REL}" "${HUB}/draft/dflash-draft-3.6-q4_k_m.gguf"

echo "Lucebox model dir ready: ${HUB}"
ls -la "${HUB}" "${HUB}/draft"
