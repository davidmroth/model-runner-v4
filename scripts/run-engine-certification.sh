#!/usr/bin/env bash
# Engine certification: stability + speed gates for tool-split deploy.
# Run on ai.local (or any host with model-runner-v4-lucebox + ai-platform-proxy).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROXY="${PROXY_URL:-http://127.0.0.1:8000}"
MODEL="${CERT_MODEL:-qwen3.6-27b-autoround}"
LUCEBOX="${LUCEBOX_CONTAINER:-model-runner-v4-lucebox}"
OUT_DIR="${CERT_OUT_DIR:-/tmp/engine-cert-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "=== Engine certification ==="
echo "proxy=$PROXY model=$MODEL out=$OUT_DIR"

# --- Unit tests (patch scripts in lucebox container or local mount) ---
echo ""
echo "--- Unit: prefix_cache + parse_tool_calls ---"
PATCH_DIR="${LUCEBOX_PATCH_DIR:-/media/data/projects/model-runner-v4/lucebox-patch/dflash/scripts}"
if docker ps --format '{{.Names}}' | grep -qx "$LUCEBOX"; then
  docker exec "$LUCEBOX" bash -c \
    "PYTHONPATH=/opt/lucebox-hub/patch/dflash/scripts uv run --no-sync --directory /opt/lucebox-hub python -m unittest discover -s /opt/lucebox-hub/patch/dflash/scripts -p 'test_*.py' -q" \
    | tee "$OUT_DIR/unit-tests.log"
elif [ -d "$PATCH_DIR" ]; then
  docker run --rm -v "$PATCH_DIR:/scripts:ro" python:3.12-slim \
    bash -c 'pip install -q "fastapi>=0.115,<1" "transformers>=4.46,<5" && cd /scripts && python3 -m unittest test_prefix_cache_slot_depth.py test_parse_tool_calls.py -q' \
    | tee "$OUT_DIR/unit-tests.log"
else
  echo "WARN: lucebox container and patch dir missing — skip unit tests"
fi

# --- Pollution regression ---
echo ""
echo "--- E2E: cache pollution regression ---"
docker run --rm --network ai-inference \
  -v "$ROOT/scripts/test_cache_pollution.py:/w/test_cache_pollution.py:ro" \
  python:3.12-slim \
  python3 /w/test_cache_pollution.py "$PROXY" "$MODEL" \
  | tee "$OUT_DIR/cache-pollution.log"

# --- Thorough tool-split bench (speed + RESTORE_CHAIN markers) ---
echo ""
echo "--- E2E: tool-split thorough benchmark ---"
docker run --rm --network ai-inference \
  -v "$ROOT/scripts/benchmark-tool-split-thorough.py:/w/bench.py:ro" \
  python:3.12-slim \
  python3 /w/bench.py "$PROXY" "$MODEL" "$LUCEBOX" "$OUT_DIR/tool-split-bench.json" \
  | tee "$OUT_DIR/tool-split-bench.log"

# --- Log markers from lucebox ---
echo ""
echo "--- Lucebox log markers (last 30m) ---"
docker logs "$LUCEBOX" --since 30m 2>&1 \
  | grep -E 'empty prompt|inline snap failed|RESTORE_CHAIN|tool KV pinned|lookup hit|cache scope|tool-split' \
  | tail -50 \
  | tee "$OUT_DIR/lucebox-markers.log" || true

echo ""
echo "=== Certification artifacts: $OUT_DIR ==="
if [ -f "$OUT_DIR/tool-split-bench.json" ]; then
  python3 -c "
import json, sys
p='$OUT_DIR/tool-split-bench.json'
r=json.load(open(p))
s=r.get('summary',{})
print('bench:', s.get('passed'), '/', s.get('total'), 'checks')
print('agent_after_tool_s:', s.get('agent_after_tool_s'))
print('agent_after_tool_prefill_ms:', s.get('agent_after_tool_prefill_ms'))
"
fi
