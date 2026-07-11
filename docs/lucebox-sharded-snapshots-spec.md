# lucebox-hub: sharded snapshot protocol (Phase 2)

**Audience:** C++ implementers in `test_dflash` (legacy inline daemon loop)  
**Trigger:** `SNAPSHOT_THIN` unsafe at 20K+ KV depth on `--target-gpus` layer-split  
**Consumer:** `model-runner-v4` `server_tools.py` + `tool_split/` (protocol unchanged on wire)

---

## Context

Layer-split places target layers and their KV on multiple GPUs:

```
GPU0: layers 0..31  + KV shard 0
GPU1: layers 32..63 + KV shard 1
```

The single-GPU snapshot implementation assumes one contiguous KV buffer. Inline thick `snap=<cut>:<slot>` during prefill partially works at 20K+; post-hoc `SNAPSHOT_THIN <slot> 0 <kv_end>` crashes or corrupts state.

Python workaround (Phase 1c): pin tool KV via inline `snap=` into tool slots 4–5 on turn 1. Phase 2 makes **both** inline and `SNAPSHOT_THIN` correct on sharded KV and removes the 16K guard.

---

## Wire protocol (unchanged)

| Command | Args | Purpose |
|---------|------|---------|
| `SNAPSHOT_THIN` | `slot kv_start kv_end` | Pin KV range into thin slot |
| `RESTORE_CHAIN` | `thick thin_csv path gen_len` | `thick=-1` for tool-only restore |
| inline | `snap=cut:slot` on prefill line | Capture at depth during prefill |
| ack thin | `[snap] thin slot=N kv=start,end` | Thin snap success |
| ack inline | `[snap] inline slot=N cur_pos=M` | Inline snap success |
| ack chain | `[snap] chain restored cur_pos=M` | Chain restore success |

Optional future: `snap=cut:slot,snap=cut2:slot2` — **not required** if Python keeps one snap per request.

---

## Data model

Each **logical slot** `N` maps to:

```cpp
struct ShardedSnapshot {
    int kv_start;
    int kv_end;           // committed depth for this slot
    std::vector<ShardBlob> shards;  // one per target GPU shard
};

struct ShardBlob {
    int device_id;
    int layer_begin;      // global layer index
    int layer_end;        // exclusive
    size_t bytes;
    std::unique_ptr<uint8_t[]> host_ram;  // CPU backend
};
```

**Right-sizing:**

```
bytes(shard) = n_head_kv(shard_layers) * head_dim * (kv_end - kv_start) * elem_size
```

Never allocate `max_ctx` per slot in VRAM.

---

## SNAPSHOT_THIN (sharded)

For each GPU shard `s` owning layers `[L0, L1)`:

1. Validate `0 <= kv_start < kv_end <= cur_pos`
2. For each layer in shard, D2H copy KV cells for token positions `[kv_start, kv_end)`
3. Store in `ShardedSnapshot` for `slot`
4. Reply `[snap] thin slot=N kv=kv_start,kv_end` or `[snap] thin slot=N error=...`

**Failure:** Must not terminate process. Live KV unchanged on error.

**Optimization:** Share copy kernel with inline `snap=` path (same per-shard gather).

---

## Inline snap (sharded)

When prefill command includes `snap=cut:slot`:

1. At prefill position `cut`, snapshot each shard for `[0, cut)` (or `[kv_start, cut)` if range semantics differ for thin slots)
2. Emit `[snap] inline slot=N cur_pos=cut`
3. Track `slot_committed_depth[N] = cut` on all shards

Tool pin slots (4–5) use the same machinery; Python uses `cut = tool_prefix_len`.

---

## RESTORE_CHAIN (sharded)

`RESTORE_CHAIN thick thin path gen_len`:

1. **Parse** `thick` (-1 = skip thick), `thin` (comma-separated slot ids)
2. **For each shard**, H2D restore thin slot ranges then thick slot (if `thick >= 0`) into live KV buffers
3. **Verify** merged `cur_pos` matches thick depth or max thin `kv_end` per merge rules
4. **Prefill** suffix from `path` (existing behavior)
5. Ack `[snap] chain restored cur_pos=M`

Merge order for tool-split (prompt layout `[tool_prefix | conversation]`):

- Thin tool slots cover `[0, tool_prefix_len)`
- Thick conv slot covers `[0, conv_prefix_len)` where `conv_prefix_len >= tool_prefix_len` typically
- Overlap: thick includes tool KV when depth > tool_prefix_len; chain must not double-write conflicting positions

Document exact merge in C++ comments; Python assumes today's single-GPU semantics.

---

## Slot lifecycle

| Op | Behavior |
|----|----------|
| `LIST_SLOTS` | Return populated logical slots (both shards committed) |
| `FREE_SNAPSHOT N` | Free host RAM for all shards of slot N |
| Slot refresh in-place | Update `kv_end`; invalidate shallower hash mappings (Python already handles LRU side) |

---

## Depth / stale guards

Mirror Python `PrefixCache` stale eviction:

- Per-slot `committed_depth` — reject restore if requested cut ≠ committed
- On in-place refresh at deeper boundary, old shallow mappings invalid

---

## Tests (add in lucebox-hub)

| Test | Layer-split | kv_end | Pass |
|------|-------------|--------|------|
| `thin_snap_4k` | yes | 4096 | ack, no crash |
| `thin_snap_20k` | yes | 20590 | ack, round-trip restore |
| `restore_chain_tool_only` | yes | -1, thin=4 | cur_pos ≈ 20590 |
| `restore_chain_both` | yes | thick=1, thin=4 | suffix prefill < 2s |
| `inline_vs_thin_equiv` | yes | 20590 | same KV hash / logits on restore |

---

## Integration checklist

- [ ] Implement sharded copy helpers (`kv_shard_copy_to_host`, `kv_shard_copy_from_host`)
- [ ] Route `SNAPSHOT_THIN` through sharded path when `target_sharding` active
- [ ] Route inline `snap=` through same helpers
- [ ] `RESTORE_CHAIN` fans out to all shards atomically before ack
- [ ] Soft error replies; no `abort()` on allocation failure
- [ ] `LIST_SLOTS` / `FREE_SNAPSHOT` shard-aware
- [ ] Rebuild `test_dflash` on ai.local; run `scripts/repro-snapshot-thin.sh` with `DFLASH_TOOL_INLINE_SNAP_PIN=0`
- [ ] Run `run-engine-certification.sh` through proxy

---

## References in tree

- Python thin IPC: `lucebox-patch/dflash/scripts/tool_split/daemon_bridge.py`
- Chain compose: `lucebox-patch/dflash/scripts/server_tools.py` `_compose_daemon_cmd`
- Architecture: `docs/research-nextgen-archecture.md` § sharded snapshot protocol
- Whitepaper: `docs/whitepaper-agent-inference-cache.md` §§ 3–5
