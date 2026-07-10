#!/usr/bin/env bash
# Deploy native mmproj vision to ai.local.
#
# SOP: git is the single source of truth — this script only git-pulls on
# ai.local. No scp/tar of source trees. See docs/deployment-sop.md.
set -euo pipefail

AI_HOST="${AI_HOST:-192.168.87.153}"
AI_USER="${AI_USER:-david}"
BOT_USER="${BOT_USER:-bot}"
ROOT_REMOTE="${AI_REMOTE_ROOT:-/media/data/projects/model-runner-v4}"
SRC_REMOTE="${LUCEBOX_SRC_REMOTE:-/media/data/projects/lucebox-hub-src}"
MR_BRANCH="${MODEL_RUNNER_BRANCH:-feat/vision}"
LB_BRANCH="${LUCEBOX_BRANCH:-feat/native-mmproj}"
BUILD_REMOTE="${SRC_REMOTE}/server/build-mmproj"
CUDA_IMAGE="${CUDA_BUILD_IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu22.04}"
CUDA_PLATFORM="${CUDA_BUILD_PLATFORM:-linux/amd64}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== 1/5 git pull lucebox-hub (${LB_BRANCH}) on ${AI_USER}@${AI_HOST} =="
ssh "${AI_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
cd '${SRC_REMOTE}'
git fetch origin '+refs/heads/${LB_BRANCH}:refs/remotes/origin/${LB_BRANCH}'
git clean -fd
git checkout -f -B '${LB_BRANCH}' "origin/${LB_BRANCH}"
git reset --hard "origin/${LB_BRANCH}"
git submodule update --init --recursive server/deps/llama.cpp
echo "lucebox-hub HEAD=\$(git rev-parse --short HEAD)"
EOF

echo "== 2/5 C++ build (build-mmproj) in CUDA container =="
ssh "${AI_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
cd '${SRC_REMOTE}'
find . -name '._*' -delete 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true
docker run --rm --platform '${CUDA_PLATFORM}' --gpus all \\
  -v '${SRC_REMOTE}:/src' \\
  -w /src/server \\
  '${CUDA_IMAGE}' \\
  bash -c '
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq cmake ninja-build git libgomp1 >/dev/null
    git config --global --add safe.directory /src/server/deps/llama.cpp
    rm -rf build-mmproj
    cmake -B build-mmproj -G Ninja \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \\
      -DDFLASH27B_MMPROJ=ON \\
      -DDFLASH27B_SERVER=ON
    cmake --build build-mmproj --target dflash_server test_dflash -j\$(nproc)
    ls -lh build-mmproj/dflash_server build-mmproj/test_dflash
  '
EOF

echo "== 3/5 git pull model-runner-v4 (${MR_BRANCH}) + compose recreate =="
ssh "${AI_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
cd '${ROOT_REMOTE}'
git fetch origin '+refs/heads/${MR_BRANCH}:refs/remotes/origin/${MR_BRANCH}'
git clean -fd
git checkout -f -B '${MR_BRANCH}' "origin/${MR_BRANCH}"
git reset --hard "origin/${MR_BRANCH}"
echo "model-runner-v4 HEAD=\$(git rev-parse --short HEAD)"
chmod +x scripts/*.sh 2>/dev/null || true
touch .env
grep -q '^DFLASH_LAYER_SPLIT=' .env 2>/dev/null || echo 'DFLASH_LAYER_SPLIT=1' >> .env
sed -i 's/^DFLASH_LAYER_SPLIT=.*/DFLASH_LAYER_SPLIT=1/' .env
grep -q '^DFLASH_DRAFT_GPU=' .env 2>/dev/null || echo 'DFLASH_DRAFT_GPU=1' >> .env
sed -i 's/^DFLASH_DRAFT_GPU=.*/DFLASH_DRAFT_GPU=1/' .env
grep -q '^IMAGE_MIN_TOKENS=' .env 2>/dev/null || echo 'IMAGE_MIN_TOKENS=256' >> .env
sed -i 's/^IMAGE_MIN_TOKENS=.*/IMAGE_MIN_TOKENS=256/' .env
grep -q '^IMAGE_MAX_TOKENS=' .env 2>/dev/null || echo 'IMAGE_MAX_TOKENS=1024' >> .env
grep -q '^DFLASH_TOOL_SPLIT_ENABLED=' .env 2>/dev/null || echo 'DFLASH_TOOL_SPLIT_ENABLED=0' >> .env
sed -i 's/^DFLASH_TOOL_SPLIT_ENABLED=.*/DFLASH_TOOL_SPLIT_ENABLED=0/' .env
grep -q '^DFLASH_PREFIX_CACHE_SLOTS=' .env 2>/dev/null || echo 'DFLASH_PREFIX_CACHE_SLOTS=4' >> .env
grep -q '^DFLASH_MMPROJ=' .env 2>/dev/null || \\
  echo 'DFLASH_MMPROJ=/opt/lucebox-hub/server/models/qwen3.6-27b-gguf/mmproj-F16.gguf' >> .env
grep -q '^LUCEBOX_DFLASH_BUILD=' .env 2>/dev/null || echo "LUCEBOX_DFLASH_BUILD=${BUILD_REMOTE}" >> .env
sed -i "s|^LUCEBOX_DFLASH_BUILD=.*|LUCEBOX_DFLASH_BUILD=${BUILD_REMOTE}|" .env
grep -q '^DFLASH_SERVER_BIN=' .env 2>/dev/null || \\
  echo 'DFLASH_SERVER_BIN=/opt/lucebox-hub/dflash-build/dflash_server' >> .env
grep -q '^DFLASH_DAEMON_BIN=' .env 2>/dev/null || \\
  echo 'DFLASH_DAEMON_BIN=/opt/lucebox-hub/dflash-build/test_dflash' >> .env
docker compose --profile serve stop vision 2>/dev/null || true
docker compose --profile serve rm -f vision 2>/dev/null || true
docker compose --profile serve up -d --force-recreate lucebox
EOF

echo "== 4/5 waiting for health (up to 15 min) =="
for i in $(seq 1 90); do
  if ssh "${BOT_USER}@${AI_HOST}" "docker exec model-runner-v4-lucebox curl -sf http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    echo "lucebox healthy (${i}0s)"
    break
  fi
  if [ "$i" -eq 90 ]; then
    echo "ERROR: health check timed out" >&2
    ssh "${BOT_USER}@${AI_HOST}" "docker logs --tail 80 model-runner-v4-lucebox" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "== 5/5 verification: /props + vision smoke + text decode bench =="
ssh "${BOT_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
docker exec model-runner-v4-lucebox curl -sf http://127.0.0.1:8080/props | python3 -c "
import json,sys
p=json.load(sys.stdin)
caps=p.get('capabilities') or {}
runtime=p.get('runtime') or {}
vs=caps.get('vision_supported') or p.get('vision_supported')
sharding=runtime.get('target_sharding', caps.get('target_sharding', p.get('target_sharding')))
spec=p.get('speculative') or {}
draft=p.get('draft_path') or p.get('dflash',{}).get('draft_path') or (p.get('model') or {}).get('draft_path')
print('vision_supported=', vs)
print('target_sharding=', sharding)
print('speculative.enabled=', spec.get('enabled'))
print('draft_path=', draft)
assert vs, 'vision_supported must be true'
assert sharding, 'target_sharding must be true for dual-GPU layer-split'
"
docker run --rm --network ai-inference \\
  -e INFERENCE_BASE=http://model-runner-v4-lucebox:8080 \\
  -v ${ROOT_REMOTE}/scripts:/scripts:ro \\
  python:3.12-slim python /scripts/vision_smoke_test.py
if [ -f ${ROOT_REMOTE}/scripts/decode_bench.py ]; then
  docker run --rm --network ai-inference \\
    -v ${ROOT_REMOTE}/scripts/decode_bench.py:/tmp/b.py:ro \\
    python:3.12-slim python /tmp/b.py
fi
echo "== hermes vision repro (direct lucebox) =="
docker run --rm --network ai-inference \\
  -e INFERENCE_BASE=http://model-runner-v4-lucebox:8080 \\
  -v ${ROOT_REMOTE}/scripts:/scripts:ro \\
  python:3.12-slim python /scripts/vision_hermes_repro_test.py
echo "== hermes vision repro (ai-platform proxy) =="
docker run --rm --network ai-inference \\
  -e INFERENCE_BASE=http://ai-platform-proxy:8000 \\
  -v ${ROOT_REMOTE}/scripts:/scripts:ro \\
  python:3.12-slim python /scripts/vision_hermes_repro_test.py || \\
  echo "WARN: proxy repro failed (queue saturation or proxy down)"
EOF

echo "Done. Layer-split + native vision + DFlash on :8080."
echo "Logs: ssh ${BOT_USER}@${AI_HOST} docker logs -f model-runner-v4-lucebox"

echo "== 6/6 ai-platform proxy: vision-safe first-token timeout =="
ssh "${AI_USER}@${AI_HOST}" bash -s <<'PROXYEOF'
set -euo pipefail
AP_ROOT="${AI_PLATFORM_ROOT:-/media/data/projects/ai-platform}"
if [ -d "$AP_ROOT" ]; then
  touch "$AP_ROOT/.env"
  grep -q '^BACKEND_FIRST_TOKEN_TIMEOUT_SEC=' "$AP_ROOT/.env" 2>/dev/null || \
    echo 'BACKEND_FIRST_TOKEN_TIMEOUT_SEC=600' >> "$AP_ROOT/.env"
  sed -i 's/^BACKEND_FIRST_TOKEN_TIMEOUT_SEC=.*/BACKEND_FIRST_TOKEN_TIMEOUT_SEC=600/' "$AP_ROOT/.env"
  (cd "$AP_ROOT" && docker compose up -d proxy) || true
  echo "ai-platform BACKEND_FIRST_TOKEN_TIMEOUT_SEC=$(grep BACKEND_FIRST_TOKEN "$AP_ROOT/.env")"
else
  echo "skip: $AP_ROOT not found"
fi
PROXYEOF
