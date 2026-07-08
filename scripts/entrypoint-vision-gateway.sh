#!/usr/bin/env bash
# Start Lucebox text backend on :18080, vision gateway on :8080.
set -euo pipefail

TEXT_PORT="${DFLASH_TEXT_PORT:-18080}"
GATEWAY_PORT="${DFLASH_GATEWAY_PORT:-8080}"
VISION_ENABLED="${DFLASH_VISION_ENABLED:-1}"

if [ "${VISION_ENABLED}" != "1" ]; then
  exec /scripts/entrypoint-dual-gpu.sh "$@"
fi

echo "[vision-gateway] starting text backend on :${TEXT_PORT}"

# Run stock/tool-split entrypoint with an internal port.
export DFLASH_PORT="${TEXT_PORT}"
export DFLASH_HOST="127.0.0.1"

/scripts/entrypoint-dual-gpu.sh "$@" &
TEXT_PID=$!

cleanup() {
  kill "${TEXT_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[vision-gateway] waiting for text backend health..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${TEXT_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${TEXT_PID}" 2>/dev/null; then
    echo "[vision-gateway] text backend exited early" >&2
    wait "${TEXT_PID}" || true
    exit 1
  fi
  sleep 5
done

if ! curl -sf "http://127.0.0.1:${TEXT_PORT}/health" >/dev/null 2>&1; then
  echo "[vision-gateway] text backend failed health check" >&2
  exit 1
fi

if [ "${VISION_ENABLED}" = "1" ]; then
  echo "[vision-gateway] waiting for vision sidecar..."
  for _ in $(seq 1 120); do
    if curl -sf "${DFLASH_VISION_BACKEND:-http://vision:8081}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi

export DFLASH_TEXT_BACKEND="http://127.0.0.1:${TEXT_PORT}"
export DFLASH_GATEWAY_HOST="0.0.0.0"
export DFLASH_GATEWAY_PORT="${GATEWAY_PORT}"
export DFLASH_PATCH_SCRIPTS="${DFLASH_PATCH_SCRIPTS:-/opt/lucebox-hub/patch/dflash/scripts}"
export PYTHONPATH="${DFLASH_PATCH_SCRIPTS}${PYTHONPATH:+:${PYTHONPATH}}"

uv pip install -q "fastapi>=0.115,<1" "uvicorn>=0.32,<1" "httpx>=0.28,<1" 2>/dev/null || true

echo "[vision-gateway] listening on :${GATEWAY_PORT} (text=${DFLASH_TEXT_BACKEND}, vision=${DFLASH_VISION_BACKEND:-http://vision:8081})"
exec uv run --no-sync --directory /opt/lucebox-hub python /scripts/vision-gateway.py
