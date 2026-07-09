#!/usr/bin/env bash
# Unified benchmark harness for layer-split + vision + prefix-cache validation.
set -euo pipefail

AI_HOST="${AI_HOST:-192.168.87.153}"
BOT_USER="${BOT_USER:-bot}"
ROOT_REMOTE="${AI_REMOTE_ROOT:-/media/data/projects/model-runner-v4}"
INFERENCE_BASE="${INFERENCE_BASE:-http://model-runner-v4-lucebox:8080}"
NETWORK="${BENCH_NETWORK:-ai-inference}"

echo "== benchmark suite (base=${INFERENCE_BASE}) =="

run_py() {
  local script="$1"
  shift
  ssh "${BOT_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
docker run --rm --network ${NETWORK} \\
  -e INFERENCE_BASE=${INFERENCE_BASE} \\
  -v ${ROOT_REMOTE}/scripts:/scripts:ro \\
  python:3.12-slim python /scripts/${script} "\$@"
EOF
}

echo "-- /props --"
ssh "${BOT_USER}@${AI_HOST}" \
  "docker exec model-runner-v4-lucebox curl -sf http://127.0.0.1:8080/props | python3 -c \"
import json,sys
p=json.load(sys.stdin)
caps=p.get('capabilities') or {}
print('vision_supported=', caps.get('vision_supported', p.get('vision_supported')))
print('target_sharding=', caps.get('target_sharding', p.get('target_sharding')))
print('speculative.enabled=', (p.get('speculative') or {}).get('enabled'))
\""

echo "-- vision smoke --"
run_py vision_smoke_test.py

if [ "${SKIP_DECODE_BENCH:-0}" != "1" ]; then
  echo "-- decode bench --"
  run_py decode_bench.py
fi

if [ "${SKIP_HERMES_BENCH:-0}" != "1" ]; then
  echo "-- hermes-style bench --"
  run_py hermes_style_bench.py
fi

if [ "${SKIP_TOOL_CACHE:-0}" != "1" ]; then
  echo "-- tool cache test --"
  run_py tool_cache_test.py
fi

echo "== benchmark suite PASS =="
