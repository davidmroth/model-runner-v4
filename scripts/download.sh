#!/usr/bin/env bash
# Download Qwen3.6-27B GGUF weights for llama.cpp (club-3090 unsloth-mtp-q4km + mmproj).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose --profile download run --rm download
