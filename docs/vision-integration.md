# Native vision (mmproj) in Lucebox v4

Vision is implemented **inside Lucebox Hub** via mtmd/mmproj — not a BeeLlama sidecar.

Upstream code lives on `lucebox-hub` branch `feat/native-mmproj`:

- `VisionEncoder` — mtmd wrapper (`server/src/vision/`)
- `http_server.cpp` — parses `image_url` / base64, builds `MultimodalPrompt`
- `Qwen35Backend::do_prefill_multimodal` — injects image embeddings into DFlash graph prefill
- `/props` → `capabilities.vision_supported` when `--mmproj` is loaded

## Build requirement

Stock GHCR image is compiled with `DFLASH27B_MMPROJ=OFF`. Production needs a **local rebuild**:

```bash
./scripts/build-native-vision-ai.local.sh
# or full pipeline (sync + build + deploy + smoke):
./scripts/deploy-native-vision-ai.local.sh
```

This produces `server/build/dflash_server` bind-mounted into the runner.

## model-runner-v4 wiring

```ini
# .env
LUCEBOX_DFLASH_BUILD=../lucebox-hub-src/server/build
DFLASH_SERVER_BIN=/opt/lucebox-hub/dflash-build/dflash_server
DFLASH_DAEMON_BIN=/opt/lucebox-hub/dflash-build/test_dflash
DFLASH_MMPROJ=/opt/lucebox-hub/server/models/qwen3.6-27b-gguf/mmproj-F16.gguf
DFLASH_MMPROJ_NO_OFFLOAD=1          # keep mmproj on GPU (v3 parity)
IMAGE_MIN_TOKENS=1024
IMAGE_MAX_TOKENS=1024
DFLASH_VISION_ENABLED=0             # sidecar path — do not use with native mmproj
```

`download-models.sh` fetches `mmproj-F16.gguf` from `unsloth/Qwen3.6-27B-GGUF`.

## Validate

```bash
python scripts/vision_smoke_test.py
INFERENCE_BASE=http://model-runner-v4-lucebox:8080 python scripts/vision_smoke_test.py
```

`/props` should show `vision_supported: true` from native server (not gateway injection).

## WebUI / Hermes stack

End-to-end path: **WebUI** → Hermes gateway (webchat plugin) → **ai-platform proxy** `:8000` → **lucebox** `:8080`.

Hermes settings (in `config.yaml`):

```yaml
agent:
  image_input_mode: native   # attach pixels on the user turn (not vision_analyze)
model:
  provider: custom
  base_url: http://<ai.local>:8000/v1
```

**Proxy first-token timeout:** multimodal prefill can take **30–120+ seconds** before the first SSE chunk (no tokens during prefill). Set on ai-platform:

```bash
# ai-platform/.env
BACKEND_FIRST_TOKEN_TIMEOUT_SEC=600   # was 180 — Hermes gateway_timeout is 1800s
BACKEND_REQUEST_TIMEOUT_SEC=600
```

Without this, Hermes sees `Remote end closed connection` / empty responses after ~3 retries even when lucebox would succeed.

**Repro tests** (run on ai.local):

```bash
# Tiny PNG + 38-tool Hermes-scale payload via proxy
INFERENCE_BASE=http://127.0.0.1:8000 python scripts/vision_hermes_repro_test.py

# Direct lucebox smoke
INFERENCE_BASE=http://model-runner-v4-lucebox:8080 python scripts/vision_smoke_test.py
```

**Common failures:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| HTTP 400 `vision input requires --mmproj` | Native build not deployed | `./scripts/deploy-native-vision-ai.local.sh` |
| `cudaMalloc` / `error=prefill` | GPU0 OOM on large tool+image prefill | lucebox `feat/native-mmproj` text chunking; eventually layer-split+vision |
| Empty Hermes response, lucebox `ok=true` | Proxy 180s first-token timeout | `BACKEND_FIRST_TOKEN_TIMEOUT_SEC=600` |
| `vision_analyze` timeout ~181s | `image_input_mode` not `native` | Set `agent.image_input_mode: native` |

## Known tradeoffs (upstream)

| Behavior | Detail |
|----------|--------|
| **Spec decode + mmproj** | Both load together; text-only requests use DFlash, image requests use AR decode |
| **Layer-split + vision** | Supported when mmproj is loaded; `LayerSplitBackend` must delegate `supports_multimodal()` to the adapter |
| **Prefix cache + vision** | Multimodal restore not implemented (`kv_offset != 0` rejected) |
| **PFlash + vision** | Compression path is text-only |

## Rollback

```bash
# Use stock build (no mmproj)
unset DFLASH_MMPROJ
LUCEBOX_DFLASH_BUILD=../lucebox-hub-src/server/build
docker compose --profile serve up -d --force-recreate lucebox
```
