"""Hermes-shaped request: big tool-bearing system prompt + prose answer."""
import json
import time
import urllib.request

BASE = "http://model-runner-v4-lucebox:8080"

TOOLS = [{"type": "function", "function": {
    "name": n,
    "description": f"Tool {n}: " + "does something useful for agents. " * 6,
    "parameters": {"type": "object", "properties": {
        "arg1": {"type": "string", "description": "first argument " * 8},
        "arg2": {"type": "integer", "description": "second argument " * 8},
    }, "required": ["arg1"]}}}
    for n in ("terminal", "read_file", "write_file", "search_files", "web_search",
              "web_extract", "vision_analyze", "delegate_task", "memory_search",
              "cronjob", "send_message", "skill_manage", "todo_write", "patch",
              "browser_navigate", "browser_click", "execute_code", "image_gen",
              "tts_speak", "clarify")]

SYS = ("You are Hermes, a capable AI agent. " +
       "Follow the operating principles carefully. " * 200)

MSGS = [
    {"role": "system", "content": SYS},
    {"role": "user", "content": "Explain how prefix caching speeds up "
     "multi-turn agent conversations. Write about 250 words."},
]

for run in (1, 2):
    payload = {"model": "dflash", "messages": MSGS, "tools": TOOLS,
               "max_tokens": 350, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    dt = time.time() - t0
    u = r.get("usage", {})
    t = u.get("timings", {})
    print(f"run{run}: prompt={u.get('prompt_tokens')} gen={u.get('completion_tokens')} "
          f"wall={dt:.1f}s prefill={t.get('prefill_ms', 0)/1000:.1f}s "
          f"decode_tps={t.get('decode_tokens_per_sec')} "
          f"accept={t.get('draft_accept_pct')}% "
          f"commit/step={t.get('avg_commit_per_step')}", flush=True)
