#!/usr/bin/env bash
# Benchmark Lucebox decode + prefill via proxy (:8000) or direct engine (:8080 in container).
set -euo pipefail

BASE="${BENCHMARK_URL:-http://127.0.0.1:8000}"
MODEL="${BENCHMARK_MODEL:-qwen3.6-27b-autoround}"
PROMPT_LEN="${BENCHMARK_PROMPT_TOKENS:-520}"
GEN="${BENCHMARK_MAX_TOKENS:-256}"
RUNS="${BENCHMARK_RUNS:-3}"

# ~4 chars/token rough HumanEval-sized prompt filler
FILLER="$(python3 -c "print('x ' * ($PROMPT_LEN // 2))")"

bench_once() {
  local stream="$1"
  local t0 t1 body
  t0=$(python3 -c "import time; print(time.time())")
  body=$(curl -sf "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${FILLER}\\n\\nWrite a short Python function that returns the sum of a list.\"}],\"max_tokens\":${GEN},\"temperature\":0,\"stream\":${stream}}")
  t1=$(python3 -c "import time; print(time.time())")
  python3 - "$body" "$t0" "$t1" <<'PY'
import json, sys
body, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
wall = (t1 - t0) * 1000
try:
    data = json.loads(body)
except json.JSONDecodeError:
    print(f"wall_ms={wall:.0f} parse_error=1")
    raise SystemExit(0)
usage = data.get("usage") or {}
timings = data.get("timings") or {}
ut = usage.get("timings") or {}
prefill_ms = ut.get("prefill_ms") or timings.get("prompt_ms")
decode_tps = ut.get("decode_tokens_per_sec") or timings.get("predicted_per_second")
prefill_tps = timings.get("prompt_per_second")
pt = usage.get("prompt_tokens", 0)
ct = usage.get("completion_tokens", 0)
print(
    f"wall_ms={wall:.0f} prompt_tokens={pt} completion_tokens={ct} "
    f"prefill_ms={prefill_ms} prefill_tps={prefill_tps} decode_tps={decode_tps}"
)
PY
}

echo "== Lucebox benchmark base=${BASE} model=${MODEL} runs=${RUNS} =="
for i in $(seq 1 "$RUNS"); do
  echo "-- run $i non-stream --"
  bench_once false
done
