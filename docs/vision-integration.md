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
./scripts/build-native-mmproj-ai.local.sh
```

This produces `server/build-mmproj/dflash_server` bind-mounted into the runner.

## model-runner-v4 wiring

```ini
# .env
LUCEBOX_DFLASH_BUILD=../lucebox-hub-src/server/build-mmproj
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

## Known tradeoffs (upstream)

| Behavior | Detail |
|----------|--------|
| **Spec decode + mmproj** | Both load together; text-only requests use DFlash, image requests use AR decode |
| **Layer-split + vision** | Not supported yet — use single-GPU or non-split placement for vision testing |
| **Prefix cache + vision** | Multimodal restore not implemented (`kv_offset != 0` rejected) |
| **PFlash + vision** | Compression path is text-only |

## Rollback

```bash
# Use stock build (no mmproj)
unset DFLASH_MMPROJ
LUCEBOX_DFLASH_BUILD=../lucebox-hub-src/server/build
docker compose --profile serve up -d --force-recreate lucebox
```
