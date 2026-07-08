# Vision integration (v4)

Hybrid vision for model-runner-v4: **Lucebox DFlash stays on GPU0** for text;
**BeeLlama mmproj sidecar on GPU1** handles multimodal requests.

## Architecture

```
Client (:8000 ai-platform)
    → model-runner-v4-lucebox :8080 (vision-gateway)
        ├─ text  → :18080 stock Lucebox dflash_server
        └─ image → vision:8081 BeeLlama llama-server + mmproj
```

## Enable

```bash
DFLASH_VISION_ENABLED=1
LUCEBOX_GPU=0
VISION_GPU=1
docker compose --profile serve up -d
```

## Validate

```bash
python scripts/vision_smoke_test.py
# via proxy:
INFERENCE_BASE=http://ai.local:8000 python scripts/vision_smoke_test.py
```

## Rollback

```bash
DFLASH_VISION_ENABLED=0
# use entrypoint-dual-gpu.sh only (set in compose or recreate without vision service)
docker compose --profile serve up -d --force-recreate lucebox
docker compose --profile serve stop vision
```

## Future work

Native mmproj inside the Lucebox dflash daemon (single process, shared KV) is
not in this PR — the sidecar path proves the routing and Hermes contract first.
