# Deployment SOP — Git Is the Single Source of Truth

**Rule:** All code on `ai.local` comes from **git remote**, never from hand-edited
files, `scp`, or `tar` sync on the host. The dev machine commits and pushes;
`ai.local` only `git fetch` / `git checkout` / `git pull`.

Host-specific state (`.env`, model weights, Docker volumes) lives on the host
and is **not** in git. Everything else is reproducible from branches.

## Repos and paths on ai.local

| Repo | Remote | Path on ai.local | Deploy branch (vision stack) |
|------|--------|------------------|--------------------------------|
| `lucebox-hub` | `github.com/davidmroth/lucebox-hub` | `/media/data/projects/lucebox-hub-src` | `feat/native-mmproj` |
| `model-runner-v4` | `github.com/davidmroth/model-runner-v4` | `/media/data/projects/model-runner-v4` | `feat/vision` |

`docker-compose.yml` bind-mounts `lucebox-hub-src/server/scripts/entrypoint.sh` so
runtime flags (`--mmproj`, etc.) match the pulled branch — the stock GHCR image
entrypoint alone is not sufficient for native vision.

## Do / Don't

| Do | Don't |
|----|-------|
| Commit + push from dev machine | Edit source under `lucebox-hub-src` or `model-runner-v4` on ai.local |
| `git fetch origin <branch>` + `git reset --hard origin/<branch>` on ai.local | `scp` / `tar` source trees to ai.local |
| Edit `.env` on ai.local for runtime knobs | Patch Python/C++ on the host without committing |
| Rebuild C++ in CUDA devel container after pull | Build on macOS and copy binaries |
| Run verification sidecars after deploy | Assume healthy because container started |

On ai.local, **`git reset --hard`** is intentional — it discards any drift from
hand-edits and matches the pushed branch exactly.

## Standard deploy sequence (native vision + DFlash)

From the dev machine:

```bash
# 1. Commit and push both repos (dev machine)
cd ~/development/projects/lucebox-hub
git push -u origin feat/native-mmproj

cd ~/development/projects/model-runner-v4
git push -u origin feat/vision

# 2. Pull, build, compose, test (orchestrated)
./scripts/deploy-native-vision-ai.local.sh
```

What the deploy script does **on ai.local only via git**:

1. `lucebox-hub-src`: `git fetch` → `checkout feat/native-mmproj` → `pull --ff-only`
2. CUDA container build → `server/build-mmproj/dflash_server`
3. `model-runner-v4`: `git fetch` → `checkout feat/vision` → `pull --ff-only`
4. `docker compose --profile serve up -d --force-recreate lucebox`
5. `/props` + `vision_smoke_test.py` + `decode_bench.py`

## Host-only files (not in git)

These are intentional local state on ai.local:

- `/media/data/projects/model-runner-v4/.env` — runtime env vars
- `/media/data/projects/models-cache/` — GGUF weights
- `server/build-mmproj/` — compiled artifacts (rebuilt after each C++ pull)

Back up `.env` before major changes. Never commit secrets.

## C++ rebuild trigger

Rebuild when **any** of these change on the pulled branch:

- `server/src/**` (C++)
- `server/CMakeLists.txt`
- `server/deps/llama.cpp` submodule pointer

Python-only changes in `server/scripts/` on the patched path still follow
[deployment-flow.md](./deployment-flow.md) Flow A if using `lucebox-patch`;
native vision uses `dflash_server` HTTP path and does not need the patch
for image parsing.

## Verification checklist

After every deploy:

1. **Git SHA** — logs or `docker exec` should match the pushed commit.
2. **`/props`** — `vision_supported: true`, draft path present, `speculative` as expected.
3. **`vision_smoke_test.py`** — 1×1 PNG color question returns a word.
4. **`decode_bench.py`** — text decode TPS regression (draft still active for text-only).

## Rollback

```bash
# On ai.local (david@)
cd /media/data/projects/lucebox-hub-src
git checkout <previous-sha>

cd /media/data/projects/model-runner-v4
git checkout <previous-sha>

# Rebuild if C++ changed, then recreate
cd /media/data/projects/model-runner-v4
docker compose --profile serve up -d --force-recreate lucebox
```

## Related docs

- [deployment-flow.md](./deployment-flow.md) — topology, mounts, Flow A–D
- [vision-integration.md](./vision-integration.md) — mmproj build flags and env knobs
