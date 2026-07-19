# Efficient Agent Workload Handling to Improve TTFT

**Splitting and reusing static agent context so follow-up turns stop re-reading
what they already know**

*July 2026 — model-runner-v4 / lucebox-hub · Qwen3.6-27B on 2×RTX 3090*  
*Audience: product, engineering, and operators evaluating agent inference latency*  
*Companion technical papers: [whitepaper-agent-inference-cache.md](./whitepaper-agent-inference-cache.md), [whitepaper-tools-system-prompt-cache.md](./whitepaper-tools-system-prompt-cache.md)*

---

## Abstract

Agent assistants such as Hermes do not send a short chat message and wait.
Every turn re-submits a large **fixed workload**—tool manuals, system setup,
and a skills index—plus a growing conversation. Time to first token (TTFT) on
follow-up turns is dominated by how much of that fixed work the engine
recomputes.

This paper argues from a simple premise: **within a Hermes session, most of
that fixed workload is intentionally stable across turns.** Efficient agent
handling therefore means paying for it once and reusing it—separating tools
from conversation, restoring prior progress on later turns, and (as a
supporting measure) warm-starting after restarts. The headline win is faster
**follow-up** TTFT in a live agent session, not a claim that the first cold
turn becomes free.

---

## 1. Premise: Agent Turns Repeat a Large Fixed Workload

Without this section, the rest of the paper has no reason to exist.

### 1.1 What each Hermes request carries

Conceptually every agent completion looks like:

```text
[ system setup ] [ tool schemas ] [ skills index ] [ conversation… ] [ what’s new ]
 \____________________ largely fixed within a session ____________________/   \__ grows __/
```

| Piece | Within one session | Across sessions |
|-------|--------------------|-----------------|
| Tool schemas | Built once and reused each API call | Often the same if tools/config unchanged; can change with MCP, plugins, or `hermes tools` |
| System prompt | Cached and kept **byte-stable** by design | Rebuilt (cwd guides, memory snapshot, model, time) |
| Skills list (index) | Frozen with the system prompt | Updates when skills are installed; normally takes effect next session |
| Conversation + tool results | Grows every turn | New / empty |

Hermes deliberately **avoids** rewriting the system prompt or swapping tools
mid-session so provider- and engine-side prefix reuse stays valid. Ephemeral
extras (memory prefetch, a slash-loaded full skill body) are attached to the
**current user message**, not by mutating the frozen system prompt.

**Qualified statement for this paper:**  
*Within a session, treat tools + system + skills index as stable by design—not
mathematically immutable (compression rebuilds and explicit “apply now”
invalidation exist), but constant on the normal multi-turn path.*

### 1.2 How large is the fixed part?

Figures below are **qualified**: they depend on which tools are enabled,
cwd context files, and skills on disk. They are not universal constants.

| Observation | Value | Source / caveat |
|-------------|-------|-----------------|
| Tool schemas alone (many tools) | ~20–30K tokens | Hermes `estimate_request_tokens_rough` guidance for 50+ tools |
| Measured thin tool-pin depth (Hermes-style) | ~20.5K tokens | Live `tool KV pinned … len=20590` on ai.local (Jul 2026) |
| Smaller toolsets observed | ~16.0K and ~20.5K pins; snapshots with 12–38 tools | Same host; fingerprint changes with toolset |
| Skills **index** (names + short descriptions) | ~3K tokens order of magnitude | Hermes docs; full skill bodies are *not* in the index |
| System / identity / memory / AGENTS.md | Typically smaller than tools; AGENTS.md capped at 20,000 characters | Varies by workspace |
| Early-turn share | Fixed head often **dominates** prompt size when history is still short | By construction of agent prompts |

So the paper’s bet is concrete: on follow-up turns, a large slice of “reading”
before the first answer is **re-reading the same agent packaging**, not reading
something new.

---

## 2. What TTFT Means Here

**TTFT (time to first token)** = wall time from sending the request until the
first useful output token streams back.

For agent UX, the hero metric is:

> **Follow-up turn TTFT** — turn 2+ in the same conversation, after tools have
> already been learned once.

Cold turn 1 (or first turn after a process restart with an unseen toolset) can
still be expensive. That is acknowledged; it is not the headline.

---

## 3. Principle

**Learn once, reuse often, only process what’s new.**

Anything that recomputes tens of thousands of unchanged tokens before every
assistant word fights the structure of Hermes workloads. Anything that restores
prior work and prefills only the delta matches that structure.

---

## 4. Split the Workload (Tools vs Conversation)

### 4.1 Idea

Agent packaging and chat history are different jobs:

- **Tools (and the stable head that travels with them)** get learned once and
  pinned.
- **Conversation** grows separately and is restored/refreshed on later turns.

Without the split, every message drags the full tool manual through the same
path as the latest “yes, run that,” forcing full or near-full re-reading.

### 4.2 What we measured (qualified)

| Situation | What logs showed | Latency / size (qualified) |
|-----------|------------------|----------------------------|
| Cold tool learn (Hermes-sized pin) | `cold tool prefill … tool_prefix_len=20590` | Prefill on order of **~90 s** for that pin depth in observed Jul 2026 runs (`prefill_s≈91.8` on a related cold path) |
| Same tools, later turn with restore | `RESTORE_CHAIN` with tool slot hit | Avoids re-paying the full ~20.5K tool prefill |
| Warm UI example (same long session family) | ~25.3K read, **~20.7K cached (~82%)**, TTFT ≈ reading time for the uncached tail | Uncached ~4.6K at ~140 t/s → TTFT on order of **~30 s**—still real work, but not another full tool cold |

**Qualification:** Absolute seconds depend on GPU load, competing traffic, and
exact prompt shape. The directional claim is stable: **follow-up turns should
not repeat a full tools cold prefill when the pin hits.**

### 4.3 Lab / certification targets (not every live chat)

Under controlled certification conditions for this stack, published targets
include warm prefill around **8.5K tokens ≤ ~0.5 s** (vs ~11 s cold/naive in
prior whitepaper measurements) and multi-turn prefill speedups of **≥ 5×**
turn-3 vs turn-1. Those are **program targets / lab gates**, not a promise that
every production Hermes turn meets 0.5 s TTFT.

---

## 5. Reuse Across Turns (Remember Conversation Progress)

### 5.1 Idea

After tools are warm, the conversation also needs memory: otherwise turn 2
restores tools but still rebuilds almost the entire chat.

Efficient handling saves conversation progress so later turns only process the
new delta (new user text, new tool results).

### 5.2 What we measured (qualified)

| Marker | Meaning | Example from live traffic |
|--------|---------|---------------------------|
| `RESTORE_CHAIN thick=0` (or denser thick slot) | Conversation progress restored with tools | Post–tool-pin turns in Hermes scopes |
| `prefix_len≈20.8K–21.7K` on ~25K prompts | Engine reports large restored prefix | Same sessions after pin |
| Deferred conversation save after cold tool pin | Turn 1 spends its one “save” on tools; conversation save follows quickly | `tail_len=201` deferred save → next turn `thick=0` |

In one observed path, a deferred conversation save of only ~200 tokens after
the cold tool pin was enough for the **next** turn to show conversation
restore (`thick=0`) instead of tools-only restore (`thick=-1`).

**Qualification:** If conversation depth grows far past save budgets, or a save
fails, follow-ups may temporarily fall back to tools-only restore and look
slower until a conversation save succeeds again.

---

## 6. Ready After Restart (Supporting, Not Hero)

After a process restart, in-memory pins are gone. The fixed tool workload is
often **still the same** across sessions when config is unchanged.

Supporting measure:

1. On a successful real (session-scoped) tool learn, **persist** the tools
   fingerprint + definitions.
2. On startup, **preload** that toolset in the background when possible.
3. First user message then aims for a tools hit instead of another cold learn.

**Status (qualified, Jul 2026):** Persistence has been observed on ai.local
(`tools snapshot saved … n_tools=…`). Startup preload helps only **after** a
snapshot exists; the first fingerprint after a blank slate still pays cold
once. Cross-session identity is **best-effort**—a different tool enablement
fingerprint is a different learn.

This section completes the loop; it is **not** the paper’s headline metric.

---

## 7. What Better Looks Like

| Experience | Good | Still expected |
|------------|------|----------------|
| Follow-up agent turns | First answer begins after processing **mostly what’s new**; tools not fully re-learned | Some prefill for new tokens / history growth |
| Right after cold tools learn | Next turn soon shows conversation reuse | The cold learn itself remains expensive once per fingerprint lifetime |
| After restart with known tools | Startup preload can move cost off the user | First-ever toolset, or toolset change, still cold once |
| Lab / cert | Targets such as ≤0.5 s prefill @ ~8.5K and ≥5× multi-turn prefill | Not every live chat meets lab isolation |

---

## 8. What This Paper Is Not

| Topic | Why it’s out of scope here |
|-------|----------------------------|
| Busy / 503 queuing | Admission control, not “reuse static agent context” |
| Deploy / rebuild SOP | Operational integrity, not TTFT mechanics |
| Vision / multimodal | Separate capability path |
| Lossy prompt compression (PFlash) | Can shorten cold reads but drops agent instructions; kept **off** for Hermes-style agents |

Related engineering detail lives in the companion cache whitepapers.

---

## 9. Summary

1. **Premise:** Hermes-style agent requests repeat a large fixed packaging
   (tools + system + skills index) across turns within a session.
2. **Problem:** Follow-up TTFT suffers when that packaging is recomputed every
   time.
3. **Approach:** Split tools from conversation; restore prior work; only
   process deltas; optionally warm-start known tools after restart.
4. **Hero metric:** Faster **follow-up** TTFT in a real agent session.
5. **Evidence (qualified):** ~20.5K-token Hermes-like tool pins; cold learns on
   the order of a minute at that depth; warm follow-ups showing high cache hit
   shares (~80%+ in observed UI timings) and conversation restore markers
   (`thick=0`); lab targets up to ~20× prefill improvement at designated sizes.

Efficient agent workload handling is not a smaller tool catalog and not a
promise of zero wait. It is matching the engine to how agents already work:
**most of the prompt does not change every turn—so neither should the cost.**

---

## Appendix A — Figure checklist (for illustrations)

Suggested figures when packaging this paper:

1. **Bar: static vs growing** — early-turn prompt composition (tools / system /
   skills index / short history).
2. **Timeline: cold then warm** — turn 1 tools learn → deferred conversation
   save → turn 2+ restore.
3. **Before/after follow-up TTFT** — same session, cold vs warm path, with
   error bars and workload notes.
4. **Caveat callout** — “within session by design” vs “across session
   best-effort.”

## Appendix B — Metric hygiene

When citing numbers externally:

- Always state **model**, **GPU**, **tool fingerprint / n_tools**, and
  **whether the turn was cold or warm**.
- Prefer ranges and “order of” language for production screenshots.
- Separate **lab/cert targets** from **opportunistic live logs**.
