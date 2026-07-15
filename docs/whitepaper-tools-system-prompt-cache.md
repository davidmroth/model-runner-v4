# Caching System Prompts and Tool Parameters for Multi-Request Agent Inference

**Cross-Turn Snapshot Pins versus Live Shared Prefix Pages for Hermes-Style
Workloads on Qwen3.6-27B (2×RTX 3090, TQ3 KV)**

*July 2026 — model-runner-v4 / lucebox-hub*

---

## Abstract

Hermes-style agent requests devote a large fraction of each prompt to a
**mostly static head**: system instructions plus tool-parameter / tool-schema
text. That head is expensive to recompute and, under concurrent admission,
expensive to **duplicate in live VRAM** when each `DFLASH_TARGET_CACHE_SLOTS`
entry holds a private full-attention KV cache.

This whitepaper separates two caching problems that are easy to conflate:

1. **Cross-turn reuse** — already shipped as tool-split thin pins +
   `RESTORE_CHAIN` (CPU/RAM snapshots restored into a live slot).
2. **Cross-request live sharing** — not yet shipped: refcounted full-attention
   (FA) KV pages so concurrent slots reference one physical copy of an
   identical system+tools prefix.

For Qwen3.6-27B under TQ3, a representative ~20k-token tools/system head
produces on the order of **~0.23–0.30 GiB of length-proportional FA KV**
(total across both layer-split GPUs). Sharing that head across four identical
toolsets removes about **three duplicate copies (~0.7–0.9 GiB)**. That is a
real capacity unlock for `N=3–4` live slots, not a “half the card” win, and it
is not vector-level deduplication of arbitrary K/V floats.

---

## 1. What We Mean by “Tools Parameters and System Prompt Cache”

An agent prompt is not one blob. Conceptually:

```text
[ system prompt ] [ tool schemas / parameters ] [ conversation + latest turn ]
      \________________ static-ish head ________________/   \___ private tail ___/
```

| Segment | Stability | Typical Hermes size | Cache role |
|---------|-----------|---------------------|------------|
| System prompt | Often stable; **can be dynamic** (clock, session facts) | ~2–4k tokens | Inside thin pin today; see §2.1.1 |
| Tool schemas / parameters | Stable per toolset fingerprint | ~15–20k tokens (measured pin depth ~20.5k) | Dominant share candidate; **lookup key today** |
| Conversation + turn | Per session / request | grows | Always private |

“Tools parameters” here means the serialized tool definitions the model sees
(names, descriptions, JSON schemas, enums) — not runtime tool *results*. Tool
results belong in the conversation tail.

---

## 2. Two Layers of Caching (Do Not Conflate)

### 2.1 Cross-turn snapshot cache (shipped)

Tool-split already pins the tool head into thin snapshot storage and restores
it on later turns via `RESTORE_CHAIN`. Goals:

- Avoid re-prefilling 15–20k tools every warm turn.
- Keep thick conversation snapshots separate from thin tool pins.
- Allow multiple prompt families without thrashing a single monolithic cache.

On the Qwen adapter path, the pinned segment is **everything before the first
user turn** — system header **and** tool schemas (`tool_prefix_ids`). The
conversation / user / tool-result suffix is not in the thin pin.

Properties:

- Storage is primarily **host RAM / disk-backed snapshot backends**.
- Live GPU KV is still filled by restore + suffix prefill into a **target-cache
  slot**.
- This is a **time** optimization (TTFT / prefill), not a **multiplicity**
  optimization for concurrent live slots.

#### 2.1.1 Correctness hazard: tools-only fingerprint, system-inclusive pin

**Pinned KV contents ≠ cache key.**

| | What is included |
|--|------------------|
| Thin pin (`tool_prefix_ids`) | System prompt + tool schemas |
| Lookup / fingerprint today | **Tools JSON only** (`tools_fingerprint`) |

If the system prompt is **dynamic** while the toolset is unchanged — for
example injecting wall-clock time, “today’s date,” or per-request session
facts into the system message — a warm `RESTORE_CHAIN` can hit the thin pin
for the **same tools fingerprint** and restore **stale system KV** (old time,
old rules). The model then attends as if that outdated system text were still
in context.

This is not merely a cache-efficiency miss: it can **quietly degrade response
quality and factual freshness** for anything that lives only in the system
prompt.

Mitigations (product / engine):

1. **Fingerprint the full pin** — key thin slots by a hash of
   `tool_prefix_ids` (or system+tools), so any system change forces a new pin.
2. **Keep volatiles out of the pin** — put clock / session facts in the
   conversation suffix (after the first-user-turn boundary), not in the system
   header that gets pinned.
3. **Split system from tools** (future) — separate snapshot/page keys if system
   churn must coexist with a long-lived tools pin.

Until (1) or (2) is enforced, treat “dynamic system + warm tool-split” as a
known correctness risk. Live shared FA pages (§2.3) have the same requirement:
the share key must cover every token in the shared head, not tools alone.

### 2.2 Live target-cache slots (shipped at N≥2)

`DFLASH_TARGET_CACHE_SLOTS=N` admits up to N overlapping requests. Today each
slot owns a **private live TargetCache** (FA KV + per-slot DeltaNet/SSM-related
state). Weights are shared; live KV is not.

Properties:

- This is a **concurrency** optimization (chat ∩ cron).
- Without live prefix sharing, N concurrent same-toolset requests pay
  **N copies** of the tools/system FA KV in VRAM after restore/prefill.

### 2.3 Live shared prefix pages (proposed)

Refcounted FA pages keyed by **token-prefix identity** (block hash of tokens +
parent hash), as in vLLM Automatic Prefix Caching / paged attention:

```text
Physical page pool
  P0..Pk  system+tools FA pages   ← one copy, refcount = #slots using it
  Pa…     chat A private pages
  Pb…     chat B private pages

Slot A map: [P0..Pk, Pa…]
Slot B map: [P0..Pk, Pb…]
```

Properties:

- Does **not** replace target-cache slots (slots still name who’s admitted /
  generating).
- Does **not** mean two chats write one contiguous private buffer (unsafe).
- Does **not** search VRAM for equal floating-point K or V vectors.

Safe reuse unit: **identical token prefix → identical FA page chain**.

---

## 3. Why Individual K/V Vector Dedup Is the Wrong Target

Attention “keys” and “values” are per-layer, per-head, per-position tensors
produced from the token *and its prefix* (RoPE / position, prior state). Two
requests may theoretically emit similar vectors at unrelated positions; systems
do not generally content-address those floats.

Pooling is:

> Reuse a previously computed **sequence** of FA KV tensors for an identical
> prompt head.

It is not:

> Intern every duplicate float across the GPU.

That distinction bounds expected savings to **shared prefix depth × (N−1)**,
not to “any redundancy somewhere in many GB of KV.”

---

## 4. Quantifying the System+Tools Head on This Stack

### 4.1 Model shape (Qwen3.6-27B)

Hybrid layout (64 layers):

```text
16 × ( 3 × Gated DeltaNet → FFN ,  1 × Gated Attention → FFN )
```

Length-proportional **full-attention** KV exists on **16 FA layers** only
(GQA: 4 KV heads, head dim 256). Gated DeltaNet / SSM state is largely
**per-slot recurrent state**, not “20k × same FA formula,” and is out of scope
for the first shared-page spike.

### 4.2 FA KV size for ~20k tools/system tokens

Approximation (K+V, all FA layers):

```text
bytes ≈ 2 × n_fa_layers × n_kv_heads × head_dim × (bits/8) × T
      ≈ 2 × 16 × 4 × 256 × (bits/8) × 20000
```

| Element size | FA KV @ 20k tokens | Notes |
|--------------|--------------------|-------|
| FP16 (16-bit) | ~1.22 GiB | Upper reference |
| TQ3 (~3-bit) | **~0.23 GiB (~234 MiB)** | Production path (`DFLASH27B_KV_TQ3`) |
| ~4-bit packed | ~0.30 GiB | Packing / scale overhead band |

Split across layer-split GPUs (roughly half per card). Measured Hermes thin-pin
depths around **20.5k** tokens sit in this band.

### 4.3 Relative to full `max_ctx`

At `max_ctx=131072`, 20k is ~**15%** of context depth. If four long, divergent
chats each fill toward max context, shared-cover savings are a **minority** of
total live FA KV. If four agent jobs mostly share tools and keep short tails,
the cover dominates the **resident** FA KV and sharing removes most of the
duplication.

---

## 5. Savings Model for Live Pooling

Let `C` = FA KV bytes of the shared system+tools head (~0.23–0.30 GiB TQ3).
Let `N` = live target-cache slots using that identical head concurrently.

| N | Copies without sharing | Copies with sharing | Bytes removed |
|---|------------------------|---------------------|---------------|
| 1 | 1 | 1 | 0 |
| 2 | 2 | 1 | ~1×C (~0.23–0.30 GiB) |
| 4 | 4 | 1 | ~3×C (~0.7–0.9 GiB) |

**What this buys:** headroom to run `N=3–4` with the same toolset without
multiplying the agent cover in VRAM — alongside existing ~11–13 GB free-GPU
headroom at N=2 idle-ish on ai.local.

**What this does not buy:** higher decode tok/s (decode remains time-sliced);
sharing across **different** toolset fingerprints; DeltaNet state sharing in
v1; automatic wins if each slot still **boot-reserves** a private contiguous
`max_ctx` buffer with no page pool (pooling requires page allocation /
refcounting).

---

## 6. Relation to Current Production Knobs

| Knob / mechanism | Layer | Role for system+tools |
|------------------|-------|------------------------|
| Tool-split thin pins + fingerprint | Cross-turn snapshot | Cache tools head across turns / sessions |
| `RESTORE_CHAIN` | Cross-turn restore | Materialize tools (+ optional thick conv) into a live slot |
| `DFLASH_TARGET_CACHE_SLOTS` | Live admit | How many private live KV notebooks exist |
| Shared FA pages (proposed) | Live VRAM | How many physical cover copies those notebooks need |

Operational intent: keep one product knob for concurrency (`TARGET_CACHE_SLOTS`)
and auto-manage tagged demux / drop-exclusive when N>1. Prefix sharing should
likewise activate under a single capability flag once implemented — not a pile
of independent per-page knobs.

---

## 7. Implementation Direction (Narrow First)

Aligned with the next-gen multi-request plan’s Phase 4:

1. **Page pool** for FA K/V blocks (fixed block size; hash = tokens in block +
   parent hash).
2. **Slot page maps** so identical system+tools prefixes refcount the same
   pages after restore/prefill.
3. **COW / private allocation** on first divergent conversation token.
4. **Scope:** FA tools/system pages only; DeltaNet/SSM remains per-slot.
5. **Exit gate:** two concurrent agent requests with identical tool fingerprint
   show tools FA GPU bytes ≈ 1× vs private baseline; correctness vs private-KV
   short generations; no use-after-free on cancel/reuse.

Industry analogue: [vLLM Automatic Prefix Caching](https://docs.vllm.ai/) with
paged attention. The destination memory model is the same; this stack remains
on dflash / layer-split / tool-split unless the engine is replaced.

---

## 8. Conclusions

1. System prompts and tool-parameter schemas form a **large, repeatable FA KV
   head** on Hermes agent traffic (~15–20k tokens; ~0.25 GiB TQ3 FA on this
   model).
2. **Cross-turn** caching of that head is already a first-class feature
   (tool-split pins + restore). The pin includes **system + tools**, but today’s
   lookup key is **tools-only** — dynamic system text (e.g. injected time) can
   restore stale system KV and hurt quality (§2.1.1).
3. **Cross-request live** caching of that head is the missing multiplier for
   multi-slot VRAM: reference-counted FA pages, not float dedup, not one shared
   writable notebook. Share keys must cover the full shared head.
4. At **N=4** same toolset, expected live-VRAM savings are about
   **0.7–0.9 GiB** of duplicated cover FA KV — enough to matter for stretch
   concurrency, not enough to treat as unlimited free slots at 131k.

Together, snapshot pins (time) and shared prefix pages (multiplicity) are the
complete “tools + system prompt cache” story for this hardware — provided the
cache key matches everything that was pinned or shared.

---

## References (internal)

- [whitepaper-agent-inference-cache.md](./whitepaper-agent-inference-cache.md) —
  tool-split, `RESTORE_CHAIN`, CPU snapshots, decode tuning.
- [nextgen-multi-request-shared-kv-plan.md](./nextgen-multi-request-shared-kv-plan.md) —
  target-cache slots, Phase 4 shared tool-prefix pages, VRAM guidance.
- [lucebox-sharded-snapshots-spec.md](./lucebox-sharded-snapshots-spec.md) —
  sharded thin/thick snapshot protocol on layer-split KV.
