#!/usr/bin/env bash
# Disable tool-split — restores reliable agent/cron inference when prefix cache mis-restores.
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

set_kv DFLASH_TOOL_SPLIT_ENABLED 0
set_kv DFLASH_TOOL_SPLIT_COMPRESS_CONV 0
set_kv DFLASH_PREFILL_MODE off

echo "Updated ${ENV_FILE}: tool-split disabled, PFlash off."
echo "Recreate lucebox:"
echo "  cd ${ROOT} && docker compose --profile serve up -d --force-recreate lucebox"
