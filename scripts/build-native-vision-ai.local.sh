#!/usr/bin/env bash
# Remote C++ build: dflash_server with native mmproj (DFLASH27B_MMPROJ=ON).
# Runs inside nvidia/cuda devel container on ai.local (Flow B).
set -euo pipefail

AI_HOST="${AI_HOST:-192.168.87.153}"
AI_USER="${AI_USER:-david}"
SRC_REMOTE="${LUCEBOX_SRC_REMOTE:-/media/data/projects/lucebox-hub-src}"
BRANCH="${LUCEBOX_BRANCH:-feat/native-mmproj}"
CUDA_IMAGE="${CUDA_BUILD_IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu22.04}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 8)}"

echo "== build native vision dflash_server → ${AI_USER}@${AI_HOST}:${SRC_REMOTE} (${BRANCH}) =="

ssh "${AI_USER}@${AI_HOST}" "SKIP_GIT_SYNC=${SKIP_GIT_SYNC:-0}" bash -s <<EOF
set -euo pipefail
cd '${SRC_REMOTE}'
if [ "\${SKIP_GIT_SYNC:-0}" != "1" ] && [ -d .git ]; then
  git fetch origin '${BRANCH}' 2>/dev/null || git fetch origin || true
  git checkout '${BRANCH}' 2>/dev/null || git checkout -B '${BRANCH}' || true
  git pull --ff-only origin '${BRANCH}' 2>/dev/null || true
  git submodule update --init --recursive server/deps/llama.cpp
fi
if [ ! -f server/deps/llama.cpp/ggml/CMakeLists.txt ]; then
  echo "ERROR: llama.cpp submodule missing under server/deps/llama.cpp" >&2
  exit 1
fi
# Strip macOS AppleDouble metadata (breaks nvcc if tar'd from a Mac).
find . -name '._*' -delete 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true

docker run --rm --gpus all \\
  -v '${SRC_REMOTE}:/src' \\
  -w /src/server \\
  '${CUDA_IMAGE}' \\
  bash -c '
    set -euo pipefail
    apt-get update -qq && apt-get install -y -qq cmake ninja-build git >/dev/null
    rm -rf build
    find /src -name "._*" -delete 2>/dev/null || true
    find /src -name ".DS_Store" -delete 2>/dev/null || true
    cmake -S . -B build -G Ninja \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DDFLASH27B_MMPROJ=ON \\
      -DDFLASH27B_SERVER=ON \\
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
    cmake --build build -j${JOBS} --target dflash_server
    ls -lh build/dflash_server
  '
EOF

echo "Build complete: ${SRC_REMOTE}/server/build/dflash_server"
