"""Single >16k-token request to exercise PFlash with draft colocated on GPU0."""
import json
import time
import urllib.request

BASE = "http://model-runner-v4-lucebox:8080"
PARA = ("Sprint retrospective notes: the ingestion service dropped events "
        "during the failover window because the consumer group rebalanced "
        "twice. Action items were assigned to the platform team with a "
        "deadline of next Thursday, pending capacity review. ")

msgs = [{"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": " ".join(PARA for _ in range(320)) +
         "\n\nWhat were the action items? Answer in two sentences."}]
payload = {"model": "dflash", "messages": msgs, "max_tokens": 120,
           "temperature": 0}
req = urllib.request.Request(BASE + "/v1/chat/completions",
                             json.dumps(payload).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1800))
dt = time.time() - t0
u = r.get("usage", {})
print("wall", round(dt, 1), "s", json.dumps(u))
print(r["choices"][0]["message"]["content"][:200])
