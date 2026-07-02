#!/usr/bin/env bash
# Health-check Lucebox inside the container (host :8080 is not published).
set -euo pipefail

CONTAINER="${1:-model-runner-v4-lucebox}"

docker exec "${CONTAINER}" curl -sf http://127.0.0.1:8080/health
echo ""
docker exec "${CONTAINER}" curl -sf http://127.0.0.1:8080/v1/models | head -c 400
echo ""
