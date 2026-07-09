#!/usr/bin/env bash
# Build lucebox-hub dflash_server with native mmproj (DFLASH27B_MMPROJ=ON) on ai.local.
set -euo pipefail

AI_HOST="${AI_HOST:-192.168.87.153}"
AI_USER="${AI_USER:-david}"
SRC_REMOTE="${LUCEBOX_SRC_REMOTE:-/media/data/projects/lucebox-hub-src}"
BRANCH="${LUCEBOX_MMPROJ_BRANCH:-feat/native-mmproj}"
BUILD_DIR="${LUCEBOX_BUILD_REMOTE:-${SRC_REMOTE}/server/build-mmproj}"

echo "== build native mmproj dflash_server on ${AI_USER}@${AI_HOST} =="
echo "   src=${SRC_REMOTE} branch=${BRANCH} build=${BUILD_DIR}"

ssh "${AI_USER}@${AI_HOST}" bash -s <<EOF
set -euo pipefail
cd '${SRC_REMOTE}'
git fetch origin
git checkout '${BRANCH}'
git pull --ff-only origin '${BRANCH}' || true
docker run --rm --gpus all \\
  -v '${SRC_REMOTE}:/src' \\
  -w /src/server \\
  nvidia/cuda:12.8.0-devel-ubuntu22.04 \\
  bash -c 'apt-get update -qq && apt-get install -y -qq cmake ninja-build git >/dev/null &&
    cmake -B build-mmproj -G Ninja -DCMAKE_BUILD_TYPE=Release \\
      -DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_MMPROJ=ON &&
    cmake --build build-mmproj --target dflash_server test_dflash -j\$(nproc) &&
    ls -lh build-mmproj/dflash_server build-mmproj/test_dflash'
EOF

echo "Done. Set on ai.local model-runner-v4 .env:"
echo "  LUCEBOX_DFLASH_BUILD=${BUILD_DIR}"
echo "  DFLASH_SERVER_BIN=/opt/lucebox-hub/dflash-build/dflash_server"
echo "  DFLASH_DAEMON_BIN=/opt/lucebox-hub/dflash-build/test_dflash"
echo "  DFLASH_MMPROJ=/opt/lucebox-hub/server/models/qwen3.6-27b-gguf/mmproj-F16.gguf"
echo "  DFLASH_VISION_ENABLED=0"
