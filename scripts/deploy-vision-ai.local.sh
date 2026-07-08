#!/usr/bin/env bash
# Sync vision integration files to ai.local and recreate v4 stack.
set -euo pipefail

AI_HOST="${AI_HOST:-192.168.87.153}"
AI_USER="${AI_USER:-david}"
BOT_USER="${BOT_USER:-bot}"
ROOT_LOCAL="${MODEL_RUNNER_V4_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT_REMOTE="${AI_REMOTE_ROOT:-/media/data/projects/model-runner-v4}"
LHUB_LOCAL="${LUCEBOX_HUB_DIR:-$(cd "${ROOT_LOCAL}/../lucebox-hub" && pwd)}"
PATCH_REMOTE="${ROOT_REMOTE}/lucebox-patch/dflash/scripts"

echo "== deploy vision → ${AI_USER}@${AI_HOST}:${ROOT_REMOTE} =="

ssh "${AI_USER}@${AI_HOST}" "mkdir -p '${PATCH_REMOTE}' '${ROOT_REMOTE}/scripts'"

scp "${ROOT_LOCAL}/scripts/"{entrypoint-vision-gateway.sh,entrypoint-dual-gpu.sh,entrypoint-tool-split-serve.sh,vision-gateway.py,vision-entrypoint.sh,download-models.sh,vision_smoke_test.py} \
    "${AI_USER}@${AI_HOST}:${ROOT_REMOTE}/scripts/"

scp "${ROOT_LOCAL}/docker-compose.yml" "${AI_USER}@${AI_HOST}:${ROOT_REMOTE}/"

scp "${LHUB_LOCAL}/server/scripts/"{vision_detect.py,server_tools.py} \
    "${AI_USER}@${AI_HOST}:${PATCH_REMOTE}/"

ssh "${AI_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
cd '${ROOT_REMOTE}'
chmod +x scripts/*.sh scripts/vision-entrypoint.sh 2>/dev/null || true
if ! grep -q '^DFLASH_VISION_ENABLED=' .env 2>/dev/null; then
  echo 'DFLASH_VISION_ENABLED=1' >> .env
fi
if ! grep -q '^VISION_GPU=' .env 2>/dev/null; then
  echo 'VISION_GPU=1' >> .env
fi
sed -i 's/^DFLASH_VISION_ENABLED=.*/DFLASH_VISION_ENABLED=1/' .env
grep -q '^LUCEBOX_GPU=' .env || echo 'LUCEBOX_GPU=0' >> .env
sed -i 's/^LUCEBOX_GPU=.*/LUCEBOX_GPU=0/' .env
grep -q '^VISION_GPU=' .env || echo 'VISION_GPU=1' >> .env
sed -i 's/^VISION_GPU=.*/VISION_GPU=1/' .env
docker compose --profile serve up -d --force-recreate lucebox vision
EOF

echo "== waiting for health (up to 10 min) =="
for _ in $(seq 1 60); do
  if ssh "${BOT_USER}@${AI_HOST}" "curl -sf http://model-runner-v4-lucebox:8080/health" >/dev/null 2>&1; then
    echo "gateway healthy"
    break
  fi
  sleep 10
done

echo "== vision smoke (direct gateway) =="
ssh "${BOT_USER}@${AI_HOST}" \
  "docker run --rm --network ai-inference -e INFERENCE_BASE=http://model-runner-v4-lucebox:8080 \
    -v ${ROOT_REMOTE}/scripts:/scripts:ro python:3.12-slim \
    python /scripts/vision_smoke_test.py"

echo "Done. Logs: ssh ${BOT_USER}@${AI_HOST} docker logs -f model-runner-v4-vision"
