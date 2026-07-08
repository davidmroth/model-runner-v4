# Feedback: Tool-Split Prefix Cache Serves Stale Benchmark KV to Agent Requests

**Date:** July 8, 2026  
**Reporter:** David Roth (Hermes Agent deployment on `ai.local`, 2×RTX 3090)  
**Stack:** `model-runner-v4` lucebox / dflash tool-split server  
**Severity:** Critical — agent and cron replies become empty or HumanEval Python code  
**Status:** Confirmed on production; workaround is `DFLASH_TOOL_SPLIT_ENABLED=0`

---

## Summary

After enabling tool-split (`DFLASH_TOOL_SPLIT_ENABLED=1`), Hermes agent and cron
sessions intermittently return:

- **Empty assistant content** (`completion_tokens=0`, `finish_reason=stop`)
- **HumanEval `has_close_elements` Python** (the standard Lucebox decode benchmark)

The lucebox log shows the failure mode:

```text
empty prompt
  [daemon] [snap] restored slot=3 cur_pos=20
[pc] lookup hit slot=3 prefix_len=20 (of 46 total)
```

The prefix cache restores a **stale snapshot** from an earlier benchmark or short
probe. The target model then decodes against wrong KV — either emitting nothing
or continuing memorized benchmark code.

This is independent of PFlash conversation compression (`DFLASH_PREFILL_MODE=off`
was already set on the affected host).

## Reproduction (via ai-platform proxy)

```bash
# On ai.local — with tool-split enabled, most proxy calls fail after the first hit
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-autoround","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20,"stream":false}'
```

Direct engine access without going through a polluted slot usually works; rapid
repeated calls reproduce `empty prompt` + 0-token responses.

## Workaround

```bash
# model-runner-v4/.env
DFLASH_TOOL_SPLIT_ENABLED=0

cd /media/data/projects/model-runner-v4
docker compose --profile serve up -d --force-recreate lucebox
```

Verify startup banner shows `tool-split = off` and logs no longer print
`empty prompt` on normal agent traffic.

## Defaults (model-runner-v4)

| Setting | Safe default | Notes |
|---------|--------------|-------|
| `DFLASH_TOOL_SPLIT_ENABLED` | `0` | Do not enable until cache restore is session-safe |
| `DFLASH_PREFILL_MODE` | `off` | See `feedback-pflash-agent-regression.md` |
| `DFLASH_TOOL_SPLIT_COMPRESS_CONV` | `0` | Conversation compression breaks agents |

## Root-cause area (lucebox-hub patch)

Fix belongs in `prefix_cache.py` / `server_tools.py` — cache lookup must not
reuse slots across unrelated prompts (benchmark vs agent). Until then, keep
tool-split disabled on production agent hosts.
