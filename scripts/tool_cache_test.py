"""Three-turn tool-calling cache test against the lucebox tool-split server.

Run from a container on the ai-inference docker network:
  docker run --rm --network ai-inference -v $PWD:/w:ro python:3.12-slim python3 /w/tool_cache_test.py
"""
import json
import time
import urllib.request

BASE = "http://model-runner-v4-lucebox:8080"
filler = ("Here is a detailed instruction manual section. " * 40 + "\n") * 24
tools = [
    {"type": "function", "function": {
        "name": "terminal",
        "description": "Run a shell command " + ("with detailed semantics. " * 60),
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string", "description": "the command"}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file " + ("with detailed semantics. " * 60),
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]


def chat(msgs, tag):
    body = json.dumps({"model": "dflash", "messages": msgs, "tools": tools,
                       "max_tokens": 60, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    print(tag, round(dt, 1), "s usage=", r.get("usage"), flush=True)
    return r["choices"][0]["message"]


msgs = [{"role": "system", "content": "You are Hermes. " + filler},
        {"role": "user", "content": "Run ls in the terminal."}]
m1 = chat(msgs, "turn1")
msgs.append({k: v for k, v in m1.items() if v is not None})
if m1.get("tool_calls"):
    for tc in m1["tool_calls"]:
        msgs.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": "file_a.txt file_b.txt"})
else:
    msgs.append({"role": "user", "content": "Assume it printed file_a.txt file_b.txt."})
msgs.append({"role": "user", "content": "Now say DONE."})
m2 = chat(msgs, "turn2")
msgs.append({k: v for k, v in m2.items() if v is not None})
msgs.append({"role": "user", "content": "Say DONE once more."})
m3 = chat(msgs, "turn3")
