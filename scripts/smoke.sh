#!/usr/bin/env bash
set -euo pipefail
PORT="${APP_PORT:-8001}"
MODEL="${PRIMARY_MODEL:-qwen3.6-27b-autoround}"
BASE="http://127.0.0.1:${PORT}"

echo "== health =="
curl -sf "${BASE}/health" | head -c 200
echo

echo "== models =="
curl -sf "${BASE}/v1/models" | python3 -m json.tool | head -20

echo "== chat =="
curl -sf "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":8,\"temperature\":0}"

echo
