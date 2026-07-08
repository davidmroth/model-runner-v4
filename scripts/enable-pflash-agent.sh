#!/usr/bin/env bash
# Enable PFlash for short interactive prompts (3k–8k tokens).
# WARNING: breaks multi-step Hermes agent/cron jobs — see docs/feedback-pflash-agent-regression.md.
# Use only for one-shot chat or benchmarks, not production agent workloads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
PFLASH_DRAFTER="/opt/lucebox-hub/server/models/qwen3.6-27b-gguf/pflash/Qwen3-0.6B-BF16.gguf"

touch "${ENV_FILE}"
set_kv() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "${ENV_FILE}" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "${ENV_FILE}"
  else
    echo "${key}=${val}" >> "${ENV_FILE}"
  fi
}

set_kv DFLASH_PREFILL_MODE auto
set_kv DFLASH_PREFILL_DRAFTER "${PFLASH_DRAFTER}"
set_kv DFLASH_PREFILL_THRESHOLD 3000
set_kv DFLASH_PREFILL_KEEP 0.10
set_kv DFLASH_LAZY 0

echo "PFlash (agent threshold 3000) enabled in ${ENV_FILE}. Recreate lucebox:"
echo "  docker compose --profile serve up -d --force-recreate lucebox"
