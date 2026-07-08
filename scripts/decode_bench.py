"""Decode TPS vs context length bench against the lucebox server.

Sends tool-bearing chat requests of increasing prompt size (avoids the
PFlash compress path) and reports the server-side decode timings.
"""
import json
import sys
import time
import urllib.request

BASE = "http://model-runner-v4-lucebox:8080"

TOOLS = [{"type": "function", "function": {
    "name": "terminal",
    "description": "Run a shell command and return stdout.",
    "parameters": {"type": "object",
                   "properties": {"cmd": {"type": "string"}},
                   "required": ["cmd"]}}}]

# Semi-realistic prose so the draft can't cheat on pure repetition.
PARA = (
    "The deployment pipeline validates each artifact against the schema "
    "registry before promotion. When a contract test fails, the release "
    "manager receives a notification with the diff and the offending commit. "
    "Rollbacks are automated below the orchestration layer, but database "
    "migrations still require a human approval step because reversible "
    "migrations cannot be guaranteed for destructive schema changes. "
)


def run(n_paras: int, tag: str, max_tokens: int = 200):
    body_text = " ".join(PARA for _ in range(n_paras))
    msgs = [
        {"role": "system", "content": "You are a concise infrastructure assistant."},
        {"role": "user", "content": body_text +
         "\n\nSummarize the deployment policy above in a numbered list."},
    ]
    payload = {"model": "dflash", "messages": msgs, "tools": TOOLS,
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    dt = time.time() - t0
    u = r.get("usage", {})
    t = u.get("timings", {})
    print(f"{tag:>8}  prompt={u.get('prompt_tokens'):>6}  "
          f"gen={u.get('completion_tokens'):>4}  wall={dt:6.1f}s  "
          f"prefill={t.get('prefill_ms', 0)/1000:6.1f}s  "
          f"decode_tps={t.get('decode_tokens_per_sec', '?'):>6}  "
          f"accept={t.get('draft_accept_pct', '?'):>5}%  "
          f"commit/step={t.get('avg_commit_per_step', '?')}",
          flush=True)
    steps = {k[8:]: v for k, v in t.items() if k.startswith("step_ms_")}
    if steps:
        parts = "  ".join(f"{k}={v}" for k, v in sorted(steps.items(),
                                                        key=lambda kv: -kv[1]))
        print(f"          step-ms: {parts}", flush=True)


if __name__ == "__main__":
    sizes = [(3, "0.5k"), (15, "2k"), (60, "8k"), (120, "16k"), (180, "24k")]
    if len(sys.argv) > 1:
        keep = set(sys.argv[1].split(","))
        sizes = [s for s in sizes if s[1] in keep]
    for n, tag in sizes:
        run(n, tag)
