#!/usr/bin/env bash
# rebuild-and-probe.sh — Rebuild test_dflash with timing instrumentation and run the timing probe.
#
# The binary is compiled inside a Docker build container (nvidia/cuda devel image)
# because ninja/cmake are not installed on the host — they live in the builder
# stage of the Dockerfile.
#
# Run ON ai.local (any user with docker access):
#   bash /media/data/projects/model-runner-v4/scripts/rebuild-and-probe.sh
set -e

SRC=/media/data/projects/lucebox-hub-src
BUILD=$SRC/server/build-mmproj
MODEL_RUNNER=/media/data/projects/model-runner-v4

# Builder image: matches the FROM in lucebox-hub-src/Dockerfile builder stage.
# Must have cmake + ninja-build available via apt (it's the -devel image).
BUILDER_IMAGE="nvidia/cuda:12.8.1-devel-ubuntu22.04"

echo "=== Step 1: Verify source has timing instrumentation ==="
if ! grep -q "\[timing\] per-step averages" "$SRC/server/src/common/dflash_spec_decode.cpp"; then
    echo "ERROR: timing printf not found in dflash_spec_decode.cpp"
    echo "Source may not have been patched. Check the file manually."
    exit 1
fi
echo "OK: timing printf found in source"

echo ""
echo "=== Step 2: Stale object will be removed inside the build container ==="
echo "Source mtime: $(stat -c '%y' "$SRC/server/src/common/dflash_spec_decode.cpp")"
echo "Object mtime: $(stat -c '%y' "$BUILD/CMakeFiles/dflash_common.dir/src/common/dflash_spec_decode.cpp.o" 2>/dev/null || echo 'not found')"
echo "Binary before: $(stat -c '%y %s bytes' "$BUILD/test_dflash")"

echo ""
echo "=== Step 3: Build inside Docker builder container ==="
echo "Image: $BUILDER_IMAGE"
echo "(Installing ninja-build, deleting stale object to force recompile, then building...)"
docker run --rm --gpus all \
    -v "$SRC":/src:ro \
    -v "$BUILD":/build \
    "$BUILDER_IMAGE" \
    bash -c "
        set -e
        apt-get update -qq && apt-get install -y -q ninja-build 2>&1 | grep -E 'ninja|Setting up|error' || true
        echo 'ninja version:' && ninja --version
        OBJ=/build/CMakeFiles/dflash_common.dir/src/common/dflash_spec_decode.cpp.o
        echo \"Removing stale object to force recompile: \$OBJ\"
        rm -f \"\$OBJ\"
        ninja -C /build test_dflash
    "
echo "Binary after:  $(stat -c '%y %s bytes' "$BUILD/test_dflash")"

echo ""
echo "=== Step 4: Verify binary has timing strings ==="
if strings "$BUILD/test_dflash" | grep -q "per-step averages"; then
    echo "OK: [timing] per-step averages found in binary"
else
    echo "WARNING: timing string not found in binary — build may not have recompiled"
    exit 1
fi

echo ""
echo "=== Step 5: Redeploy lucebox ==="
cd "$MODEL_RUNNER"
docker compose --profile serve up -d --force-recreate lucebox
echo "Waiting for lucebox to be healthy..."
timeout 120 bash -c "until docker exec model-runner-v4-lucebox curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; do sleep 3; done"
echo "Lucebox is healthy"

echo ""
echo "=== Step 6: Run timing probe ==="
SYS_PROMPT=$(python3 -c "print('You are a helpful AI assistant with extensive knowledge. ' * 150)")
PAYLOAD=$(python3 -c "
import json, sys
msg = [
    {'role': 'system', 'content': sys.argv[1]},
    {'role': 'user',   'content': 'Count from 1 to 10.'}
]
print(json.dumps({'model':'dflash','messages':msg,'max_tokens':128,'stream':False}))
" "$SYS_PROMPT")

echo "Sending probe request (~${#SYS_PROMPT} char system prompt)..."
RESPONSE=$(docker exec model-runner-v4-lucebox \
    curl -s --max-time 180 -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "content-type: application/json" \
    -d "$PAYLOAD" 2>/dev/null)

echo ""
echo "=== Timing results ==="
echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)

err = d.get('error')
if err:
    print('ERROR:', err)
    sys.exit(1)

t = d.get('usage', {}).get('timings', {})
step = {k: v for k, v in t.items() if 'step_ms' in k}
other = {k: v for k, v in t.items() if 'step_ms' not in k}

if step:
    print('Per-step phase breakdown:')
    total = 0
    for k, v in sorted(step.items()):
        name = k.replace('step_ms_', '')
        print(f'  {name:20s}: {v:8.2f} ms')
        if 'sum' not in k:
            total += v
    print(f'  {\"(sum of phases)\":20s}: {total:8.2f} ms')
    print()
    # Key diagnosis
    verify = step.get('step_ms_verify_compute', 0)
    copyfeat = step.get('step_ms_draft_copyfeat', 0)
    replay = step.get('step_ms_replay_compute', 0)
    total_step = step.get('step_ms_sum', total)
    print('=== Diagnosis ===')
    print(f'  verify_compute  {verify:7.1f} ms  (expected ~87ms at 1.5K ctx if layer-split ~=whitepaper)')
    print(f'  draft_copyfeat  {copyfeat:7.1f} ms  (expected ~2ms if mirror fix working; ~706ms if broken)')
    print(f'  replay_compute  {replay:7.1f} ms')
    print(f'  sum             {total_step:7.1f} ms  (expected ~200ms at 1.5K ctx)')
    print()
    if copyfeat > 100:
        print('>> FINDING: draft_copyfeat is HIGH — feature mirror overhead not eliminated')
    elif verify > 130:
        print('>> FINDING: verify_compute dominates — layer-split pipeline penalty is the bottleneck')
    else:
        print('>> FINDING: phases look reasonable; check sum vs expected')
else:
    print('No step_ms_* keys in response.')
    print('All timing keys:', list(t.keys()))
    if not t:
        print('No timings at all — daemon may not have run speculative decode')
        print('Response choices:', d.get('choices', [{}])[0].get('message', {}).get('content', '')[:100])

print()
print('Other timings:')
for k, v in sorted(other.items()):
    print(f'  {k}: {v}')
"
