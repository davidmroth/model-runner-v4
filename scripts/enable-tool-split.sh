#!/usr/bin/env bash
# Enable split tool KV + conversation PFlash in model-runner-v4 .env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

set_kv() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    sed -i.bak "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    echo "${key}=${val}" >> "$ENV_FILE"
  fi
}

[[ -f "$ENV_FILE" ]] || cp "${ROOT}/.env.example" "$ENV_FILE"

set_kv DFLASH_TOOL_SPLIT_ENABLED 1
set_kv DFLASH_TOOL_SPLIT_PROFILE auto
set_kv DFLASH_TOOL_SPLIT_PINNED_SLOTS 2
set_kv DFLASH_TOOL_SPLIT_COMPRESS_CONV 1

echo "Updated ${ENV_FILE}: tool-split enabled (profile=auto)."
echo "Requires DFLASH_LAYER_SPLIT=0 for KV snapshots."
echo "Recommended slot budget (8 max): prefix=1 prefill=2 tool_pins=2"
echo "Recreate lucebox:"
echo "  cd ${ROOT} && docker compose --profile serve up -d lucebox"
