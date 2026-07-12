# Deployment SOP — Git Is the Single Source of Truth

**Effective:** July 2026  
**Applies to:** All humans and AI assistants working on the inference stack.

## Rule (non-negotiable)

**Never deploy source code to `ai.local` via `scp`, `rsync`, hand-edits, or
`tar`.** Commit on the dev machine, push to the remote, then `git pull` on
`ai.local`. The host runs what is in git — nothing else.

This rule exists because untracked `scp` copies cause `git pull` failures
(“untracked working tree files would be overwritten by merge”), hide drift from
the team, and make rollback impossible.

### Allowed on `ai.local` without git

| Path | What | Example |
|------|------|---------|
| `model-runner-v4/.env` | Runtime knobs | `DFLASH_DRAFT_FEATURE_MIRROR=1` |
| `ai-platform/.env` | Proxy knobs | `SMART_BENCHMARK_REFERENCE_ENABLED=0` |
| Docker volumes | Weights, caches | `models-cache/` |
| Build artifacts | Rebuilt after C++ pull | `lucebox-hub-src/server/build-mmproj/` |

**Do not** edit `lucebox-patch/`, `server/scripts/`, compose files, or
entrypoints on the host. Change them in git, push, pull.

---

## Repos on `ai.local`

| Repo | Path on host | Typical branch | Container |
|------|--------------|----------------|-----------|
| `model-runner-v4` | `/media/data/projects/model-runner-v4` | `feat/vision` | `model-runner-v4-lucebox` |
| `lucebox-hub` | `/media/data/projects/lucebox-hub-src` | `feat/native-mmproj` | (bind-mount into lucebox) |
| `ai-platform` | `/media/data/projects/ai-platform` | `main` or team branch | `ai-platform-proxy` |

`model-runner-v4` bind-mounts `lucebox-patch/dflash/scripts/` into the
lucebox container at `/opt/lucebox-hub/patch/dflash/scripts` (read-only).
**Patch changes deploy via git pull + container recreate** — not `scp`.

---

## Standard workflow (every code change)

### 1. Dev machine — commit and push

```bash
cd ~/development/projects/model-runner-v4   # or lucebox-hub, ai-platform
# edit, test locally if possible
git add <files>
git commit -m "describe the change"
git push origin <branch>
```

### 2. `ai.local` — pull and recreate

```bash
ssh bot@ai.local

# model-runner-v4 (Python patch, compose, entrypoints, scripts)
cd /media/data/projects/model-runner-v4
git fetch origin
git pull origin feat/vision          # or: git pull --ff-only
docker compose up -d --force-recreate lucebox

# ai-platform (proxy changes)
cd /media/data/projects/ai-platform
git pull
docker compose up -d --force-recreate ai-platform-proxy   # service name may vary; use docker ps

# lucebox-hub (C++ only — rebuild if src/ changed)
cd /media/data/projects/lucebox-hub-src
git pull origin feat/native-mmproj
# see deployment-flow.md Flow B for CUDA rebuild, then recreate lucebox
```

### 3. Verify

```bash
curl -sf http://127.0.0.1:8000/health
docker logs --tail 30 model-runner-v4-lucebox
# optional: scripts/run-engine-certification.sh (from pulled repo)
```

---

## Do / Don't

| Do | Don't |
|----|-------|
| Commit + push from dev machine | `scp` files to `lucebox-patch/` or `server/scripts/` |
| `git pull` on `ai.local` | Hand-edit Python/C++ source on the host |
| `docker compose up -d --force-recreate` after pull | `docker restart` only (misses compose/env changes) |
| Edit `.env` on host for runtime knobs | Commit secrets or host-specific `.env` to git |
| Rebuild C++ in CUDA devel container after hub pull | Build on macOS and copy binaries |
| Use `git status` before pull | Assume pull will overwrite untracked files safely |

---

## Fixing `git pull` blocked by untracked files

If you previously used `scp` (or an agent did), pull may fail with:

```
error: The following untracked working tree files would be overwritten by merge
```

**If the untracked files are stale copies of what git is about to deliver:**

```bash
cd /media/data/projects/model-runner-v4
# remove only the paths git names in the error message
rm -rf lucebox-patch
rm -f docs/inference-engine-north-star.md
rm -f scripts/run-engine-certification.sh scripts/test_cache_pollution.py
git pull
```

**If you might have host-only edits worth keeping:**

```bash
cp -a lucebox-patch /tmp/lucebox-patch.bak.$(date +%Y%m%d)
diff -ru /tmp/lucebox-patch.bak.* lucebox-patch   # inspect before deleting
# then remove and pull
```

After a successful pull, recreate the container so bind-mounts pick up
tracked files.

---

## Config-only changes (no git)

Runtime tuning stays in host `.env` files:

```bash
ssh bot@ai.local
cd /media/data/projects/model-runner-v4
# edit .env (e.g. DFLASH_DRAFT_FEATURE_MIRROR=1)
docker compose up -d --force-recreate lucebox
```

Confirm in logs: `docker logs model-runner-v4-lucebox 2>&1 | grep draft_feature_mirror`

---

## C++ rebuild trigger

Rebuild `test_dflash` / `dflash_server` on `ai.local` when the pulled
`lucebox-hub` branch changes:

- `server/src/**`
- `server/CMakeLists.txt`
- `server/deps/llama.cpp` submodule pointer

Python-only `model-runner-v4` changes need **pull + recreate only**.

---

## Rollback

```bash
ssh bot@ai.local
cd /media/data/projects/model-runner-v4
git log -5 --oneline
git checkout <previous-sha>
docker compose up -d --force-recreate lucebox
```

Same pattern for `lucebox-hub-src` and `ai-platform`. Rebuild C++ if the
hub SHA changed.

---

## For AI coding assistants

1. **Make all code edits in the local git checkout** (dev machine workspace).
2. **Do not `scp`** to `bot@ai.local` — ever.
3. After changes, tell the user to **commit, push, and pull on ai.local**, or
   run the pull/recreate steps via SSH only after the push exists on remote.
4. Host `.env` edits are OK via SSH when tuning runtime knobs the user
   requested (document what was changed).

---

## Related docs

- [deployment-flow.md](./deployment-flow.md) — topology, mounts, C++ build flow
- [engine-certification-plan.md](./engine-certification-plan.md) — post-deploy gates
- [inference-engine-north-star.md](./inference-engine-north-star.md) — perf targets
