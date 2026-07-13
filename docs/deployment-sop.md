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

SSH as **`bot@ai.local`** for day-to-day host ops. `bot` is in the `users`
group, so repo `.git` trees may be owned by **`david:users`** as long as they
stay **group-writable**. Mixed per-user object dirs (some `bot:bot`, some
`david:david`) break `git pull` / `git fetch`
(“insufficient permission for adding an object to repository database”).

One-time repair (run on the host with sudo — expand paths, do not use `...`):

```bash
#!/bin/bash
set -euo pipefail

repos=(
  /media/data/projects/model-runner-v4
  /media/data/projects/lucebox-hub-src
  /media/data/projects/ai-platform
)

for repo in "${repos[@]}"; do
  sudo chown -R david:users "$repo/.git"
  sudo chmod -R g+w "$repo/.git"
  sudo find "$repo/.git" -type d -exec chmod g+s {} \;
  git -C "$repo" config core.sharedRepository group
done
```

After that, either `bot@` or `david@` can fetch/pull. Prefer **`bot@`** for
agents and documented deploy steps so ownership does not drift again.

Host remotes must use a **wildcard fetch refspec**. A narrow refspec (e.g.
only `feature/dual-gpu-…`) leaves `refs/remotes/origin/<deploy-branch>`
stale: `git fetch origin <branch>` updates `FETCH_HEAD` but not
`origin/<branch>`, so tip checks lie. Fix once per repo:

```bash
git -C /media/data/projects/lucebox-hub-src \
  config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git -C /media/data/projects/model-runner-v4 \
  config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git -C /media/data/projects/ai-platform \
  config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
```

`lucebox-hub-src` must track a **single pushed branch tip**. Host-only commits
are forbidden. If a fix exists only on the host, cherry-pick it onto the deploy
branch on the **dev machine**, push, then `git pull --ff-only` on ai.local.

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

If two fixes live on different commits/branches, **merge or cherry-pick them
in the dev repo into one deployable SHA**, then push. Do not combine them on
the host.

### 2. `ai.local` — pull and recreate

```bash
ssh bot@ai.local

# model-runner-v4 (Python patch, compose, entrypoints, scripts)
cd /media/data/projects/model-runner-v4
git fetch origin
git pull --ff-only origin feat/vision
docker compose up -d --force-recreate lucebox

# ai-platform (proxy changes)
cd /media/data/projects/ai-platform
git pull --ff-only
docker compose up -d --force-recreate ai-platform-proxy   # service name may vary; use docker ps

# lucebox-hub (C++ only — rebuild if src/ changed)
cd /media/data/projects/lucebox-hub-src
git fetch origin
git pull --ff-only origin feat/native-mmproj
# MUST pass pre-rebuild gate below, then CUDA rebuild (deployment-flow.md Flow B)
# then: docker compose -f ... up -d --force-recreate lucebox
```

### 3. Verify

```bash
curl -sf http://127.0.0.1:8000/health
docker logs --tail 30 model-runner-v4-lucebox
# Warm-path / tool-split gate (required after hub or patch changes that
# touch restore/snap):
#   scripts/run-engine-certification.sh
# or at minimum two chat turns and confirm logs:
#   - turn 1: tool KV pinned + no "deferred conv snap failed"
#   - turn 2: RESTORE_CHAIN thick=<N> thin=[...]  (NOT thick=-1 after a cold pin)
```

---

## Do / Don't

| Do | Don't |
|----|-------|
| Commit + push from dev machine | `scp` files to `lucebox-patch/` or `server/scripts/` |
| `git pull --ff-only` on `ai.local` | Hand-edit Python/C++ source on the host |
| `docker compose up -d --force-recreate` after pull | `docker restart` only (misses compose/env changes) |
| Edit `.env` on host for runtime knobs | Commit secrets or host-specific `.env` to git |
| Rebuild C++ only from a **clean tree equal to the pushed tip** | Rebuild from a dirty or hybrid working tree |
| Merge/cherry-pick on the **dev machine**, push one SHA | `git checkout <ref> -- path/to/file` on ai.local to “grab one fix” |
| SSH as `bot@ai.local` | Mix `david@` / other users for git + rebuild ops |
| Use `git status` before pull | Assume pull will overwrite untracked files safely |

### Why path-limited checkout is banned

`git checkout <ref> -- <files>` on the host creates a **hybrid tree that is
not any commit**. A July 2026 vision deploy used that pattern to pull
`supports_multimodal()` from `feat/native-mmproj` and silently reverted a
host-local `RESTORE_CHAIN` `base_pos` fix. The rebuilt `test_dflash` matched
no git SHA; deferred thick snaps failed and warm TTFT collapsed to full
prefill. Treat path checkout as a hand-edit.

---

## C++ pre-rebuild gate (mandatory)

Before any CUDA rebuild of `test_dflash` / `dflash_server`:

```bash
ssh bot@ai.local
cd /media/data/projects/lucebox-hub-src
git fetch origin
DEPLOY_REF="${DEPLOY_REF:-origin/feat/native-mmproj}"

git status -sb
test -z "$(git status --porcelain)" || {
  echo "REFUSING rebuild: dirty working tree"; git status --porcelain; exit 1
}
git pull --ff-only "$DEPLOY_REF" || {
  echo "REFUSING rebuild: cannot ff-only to $DEPLOY_REF"; exit 1
}
test "$(git rev-parse HEAD)" = "$(git rev-parse "$DEPLOY_REF")" || {
  echo "REFUSING rebuild: HEAD ($(git rev-parse --short HEAD)) != $DEPLOY_REF"; exit 1
}

BUILT_SHA="$(git rev-parse HEAD)"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "built_sha=$BUILT_SHA" | tee server/build-mmproj/BUILD_PROVENANCE
echo "built_at=$BUILT_AT" | tee -a server/build-mmproj/BUILD_PROVENANCE
echo "built_ref=$DEPLOY_REF" | tee -a server/build-mmproj/BUILD_PROVENANCE
# proceed with CUDA rebuild (deployment-flow.md Flow B), then recreate lucebox
```

After recreate, confirm provenance:

```bash
cat /media/data/projects/lucebox-hub-src/server/build-mmproj/BUILD_PROVENANCE
docker logs model-runner-v4-lucebox 2>&1 | grep -E "deferred conv snap failed|RESTORE_CHAIN thick=" | tail -20
```

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
git pull --ff-only
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

Always run the **pre-rebuild gate** above. Python-only `model-runner-v4`
changes need **pull + recreate only**.

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
hub SHA changed (pre-rebuild gate still applies: clean tree at that SHA).

---

## For AI coding assistants

1. **Make all code edits in the local git checkout** (dev machine workspace).
2. **Do not `scp`** to `bot@ai.local` — ever.
3. After changes, tell the user to **commit, push, and pull on ai.local**, or
   run the pull/recreate steps via SSH only after the push exists on remote.
4. Host `.env` edits are OK via SSH when tuning runtime knobs the user
   requested (document what was changed).
5. On ai.local use **`bot@ai.local`** only: `git pull --ff-only`, clean-tree
   rebuild, compose recreate.
6. **Never** `git checkout <ref> -- <paths>` on the host to combine fixes.
7. If two fixes live on different commits/branches, merge them in the **dev**
   repo, push one SHA, then pull that SHA on the host.
8. After hub/patch deploys that touch restore/snap/cache, fail the deploy if
   logs show `deferred conv snap failed` or turn-2 `RESTORE_CHAIN thick=-1`
   right after a successful cold tool pin.

---

## Related docs

- [deployment-flow.md](./deployment-flow.md) — topology, mounts, C++ build flow
- [engine-certification-plan.md](./engine-certification-plan.md) — post-deploy gates
- [inference-engine-north-star.md](./inference-engine-north-star.md) — perf targets
