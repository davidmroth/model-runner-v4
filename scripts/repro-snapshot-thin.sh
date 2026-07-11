#!/usr/bin/env bash
# Phase 1a: reproduce SNAPSHOT_THIN failure on layer-split (ai.local staging).
# Forces legacy thin snap (disables inline pin) and optional unlimited depth.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LUCEBOX="${LUCEBOX_CONTAINER:-model-runner-v4-lucebox}"
PROXY="${PROXY_URL:-http://ai-platform-proxy:8000}"
MODEL="${CERT_MODEL:-qwen3.6-27b-autoround}"
OUT="${REPRO_OUT:-/tmp/snapshot-thin-repro-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "=== SNAPSHOT_THIN repro (Phase 1a) ==="
echo "container=$LUCEBOX proxy=$PROXY out=$OUT"
echo ""
echo "Prerequisites on ai.local .env:"
echo "  DFLASH_TOOL_INLINE_SNAP_PIN=0"
echo "  DFLASH_TOOL_SNAPSHOT_MAX_KV=0"
echo "Then: docker compose up -d --force-recreate model-runner-v4-lucebox"
echo ""

docker logs "$LUCEBOX" --tail 30 2>&1 | tee "$OUT/pre.log" || true

echo ""
echo "--- Single cold tool turn (benchmark phase A) ---"
docker run --rm --network ai-inference \
  -v "$ROOT/scripts/benchmark-tool-split-thorough.py:/w/bench.py:ro" \
  python:3.12-slim \
  python3 -c "
import subprocess, sys
sys.path.insert(0, '/w')
# minimal: one cold agent turn only
import urllib.request, json, time
BASE = '${PROXY}'
MODEL = '${MODEL}'
TOOLS = [{'type':'function','function':{'name':'read_file','description':'Read file','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}]
payload = {'model': MODEL, 'messages': [{'role':'user','content':'Read /etc/hostname one line.'}], 'tools': TOOLS, 'max_tokens': 32, 'temperature': 0}
req = urllib.request.Request(f'{BASE}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.load(resp)
print('completion_tokens', body.get('usage', {}).get('completion_tokens'))
" 2>&1 | tee "$OUT/request.log" || echo "REQUEST_FAILED" | tee -a "$OUT/request.log"

sleep 2
docker logs "$LUCEBOX" --since 5m 2>&1 | tee "$OUT/post.log"
docker logs "$LUCEBOX" --since 5m 2>&1 | grep -E 'SNAPSHOT_THIN|tool KV pinned|segfault|error|daemon|inline tool pin' | tee "$OUT/markers.log" || true

if docker inspect "$LUCEBOX" --format '{{.State.Status}}' 2>/dev/null | grep -q running; then
  echo "daemon_container=running" | tee "$OUT/status.txt"
else
  echo "daemon_container=NOT_RUNNING" | tee "$OUT/status.txt"
fi

echo ""
echo "Artifacts: $OUT"
echo "Next: compare inline snap= path vs SNAPSHOT_THIN in lucebox-hub (docs/lucebox-sharded-snapshots-spec.md)"
