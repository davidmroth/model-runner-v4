# Tool-split cache (split tool KV from conversation for PFlash)

Pins tool-schema KV separately from the growing conversation so PFlash can
compress agent history without re-prefilling static tool definitions.

## Enable

```bash
# CLI (server_tools.py)
python3 scripts/server_tools.py --tool-split auto --prefill-compression auto ...

# Environment (model-runner-v4 / compose)
DFLASH_TOOL_SPLIT_ENABLED=1
DFLASH_TOOL_SPLIT_PROFILE=auto   # or qwen3, laguna, plugin:my_vendor
```

## Built-in profiles

| Profile | Detects | Notes |
|---------|---------|-------|
| `qwen3` | `qwen35` / `qwen36` arch, Qwen HF tokenizer | System + tools prefix split |
| `laguna` | `laguna` arch, Poolside tokenizer | XML boundary fallback |

`auto` picks the first adapter whose `detect()` matches the loaded GGUF.

## User plugins

1. Copy `example_user_plugin.py` to a directory, e.g. `~/.lucebox/tool_split_plugins/my_llm.py`.
2. Implement `ToolSplitAdapter` and decorate with `@register_adapter("my_llm")`.
3. Start the server with:

```bash
--tool-split on \
--tool-split-profile plugin:my_llm \
--tool-split-plugin-dir ~/.lucebox/tool_split_plugins
```

Or via env:

```bash
export DFLASH_TOOL_SPLIT_ENABLED=1
export DFLASH_TOOL_SPLIT_PROFILE=plugin:my_llm
export DFLASH_TOOL_SPLIT_PLUGIN_DIR=$HOME/.lucebox/tool_split_plugins
```

## Settings

| Flag / env | Default | Purpose |
|------------|---------|---------|
| `--tool-split` / `DFLASH_TOOL_SPLIT_ENABLED` | off | Master switch (`auto` enables when adapter matches) |
| `--tool-split-profile` / `DFLASH_TOOL_SPLIT_PROFILE` | `auto` | Adapter name |
| `--tool-split-plugin-dir` / `DFLASH_TOOL_SPLIT_PLUGIN_DIR` | — | User `.py` plugins |
| `--tool-split-pinned-slots` / `DFLASH_TOOL_SPLIT_PINNED_SLOTS` | `2` | Tool KV slots (daemon) |
| `--tool-split-compress-conv` / `DFLASH_TOOL_SPLIT_COMPRESS_CONV` | on | PFlash on conversation suffix |

## Daemon flow (wired)

1. **First request** with a tool fingerprint: full prefill with inline
   ``snap=<tool_prefix_len>:<pin_slot>`` (default, ``DFLASH_TOOL_INLINE_SNAP_PIN=1``),
   or legacy ``SNAPSHOT_THIN <slot> 0 <tool_prefix_len>`` when inline pin is off.
2. **Later requests** with the same tools: ``RESTORE_CHAIN -1 <tool_slot> <prompt> <n_gen>``
   restores tool KV and prefills only the conversation suffix.
3. **Conversation prefix hit** + tool hit:
   ``RESTORE_CHAIN <conv_slot> <tool_slot> <prompt> <n_gen>``.

Slot layout (8 daemon slots max): ``[prefix LRU][full compress][tool pins]``.

## Requirements

- Layer-split + legacy daemon (`DFLASH_LEGACY_DAEMON=1`, default in
  ``entrypoint-tool-split-serve.sh``): SNAPSHOT_THIN / RESTORE_CHAIN supported.
- Layer-split **without** tool-split: prefix/full cache slots are disabled.
- PFlash enabled (`--prefill-compression auto`) for conversation compression.

## Architecture

```
ToolSplitAdapter.split_prompt() → PromptSplit
  tool_prefix_ids  → pinned ToolSlotCache (RESTORE_CHAIN thin slot / SNAPSHOT_THIN)
  conversation_ids → PFlash + PrefixCache inline snap
```

Implement `split_prompt()` for each LLM family; the orchestrator and daemon
command formatting are shared.

## Session-scoped prefix cache

Thick conversation snapshots (`PrefixCache`) are keyed by **cache scope**:

- **`X-Conversation-Id`** (or `X-Hermes-Conversation-Id`, etc.) — multi-turn
  reuse within one agent session.
- **No conversation header** — ephemeral scope per distinct prompt+tools hash;
  benchmarks and one-off probes cannot pollute agent slots.

Thin tool KV remains keyed by tools fingerprint only (shared across sessions
with the same tool definitions).
