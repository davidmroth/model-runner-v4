# model-runner-v4

[Lucebox Hub](https://github.com/Luce-Org/lucebox-hub) inference engine — speculative DFlash decode for Qwen3.6-27B on consumer GPUs.

Runs **internal port 8080 only**. Client traffic goes through **ai-platform** proxy on `:8000`.

## Prerequisites

- Docker + NVIDIA runtime
- Shared network: `docker network create ai-inference` (or `ai-platform/scripts/init-network.sh`)
- ai-platform running

## Setup

```bash
cp .env.example .env
# Shared weights: ~/projects/models-cache (see ../ai-platform README)
# Lucebox uses DFLASH_TARGET / DFLASH_DRAFT in .env for explicit paths.

docker compose --profile download run --rm download
docker compose --profile serve up -d
./scripts/healthcheck.sh
```

## Switch from v3

```bash
../ai-platform/scripts/use-v4.sh
```

## Notes

- **Text only** — no vision/mmproj (use v3 for vision workloads).
- Dual 3090: target on GPU0, draft on GPU1 via `DFLASH_TARGET_GPU=0` / `DFLASH_DRAFT_GPU=1`.
  Lucebox's stock entrypoint ignores those env vars; this repo's `entrypoint-dual-gpu.sh`
  wrapper injects `--target-device cuda:0` / `--draft-device cuda:1` for `dflash_server`.
- **Decode benchmark** (HumanEval, n=256, direct engine): ~87 tok/s on dual RTX 3090 vs
  ~130 tok/s single-GPU reference in upstream `server/RESULTS.md` (Qwen3.5). Gap is mostly
  Qwen3.6 + dual-GPU draft placement; still ~2.3× over AR (~38 tok/s).
- **PFlash** (speculative prefill for 16k+ prompts): optional — `./scripts/enable-pflash.sh`
  then recreate. Default off: loading the 0.6B prefill drafter on GPU1 next to the DFlash draft
  costs ~15% decode unless `DFLASH_LAZY=0` and threshold is high. Never set `DFLASH_LAZY=1` on dual-GPU.
- **Power**: upstream sweet spot is `nvidia-smi -pl 220` (needs root on the host).
- Image: `ghcr.io/luce-org/lucebox-hub:cuda12`
- Watchdog and client API are in **ai-platform**, not this repo.

## Layout

```
model-runner-v4/
├── docker-compose.yml   # download + lucebox (engine only)
└── scripts/
```
