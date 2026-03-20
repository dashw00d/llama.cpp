#!/bin/bash
# ============================================================
# TEST 32: EP np=16 Throughput
# ============================================================
#
# WHAT: Maximum throughput test with Expert Parallelism.
#       16 concurrent users on 96-expert REAM model.
#
# WHY:  THIS IS THE END GOAL. EP's real advantage is at high
#       concurrency. With 16 tokens per dispatch:
#       - 16 × 8 = 128 expert activations per layer
#       - ~43 activations per GPU (32 local experts)
#       - GPU compute finally utilizes significant fraction
#         of Arc A770's execution units
#       - Kernel launch overhead amortized across 16 tokens
#
# EXPECTED (when EP is fixed):
#   Per-user: ~400-600ms (1.5-2.5 t/s each)
#   Aggregate: 24-40 t/s total
#   GPU utilization: 30-50% (up from 1.5% at np=1)
#
# COMPARISON:
#   TP np=16: ~8-11 t/s aggregate (AllReduce bottleneck)
#   EP np=16: ~24-40 t/s aggregate (no MoE AllReduce)
#   Improvement: 2-4x throughput from EP at scale
#
# BLOCKED BY: OQ-1 (EP np=1 corruption). Fix np=1 first.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="32-ep-np16-throughput"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="

if [ ! -x "$EPTP_SERVER" ]; then
    echo "SKIP: eptp build not found"
    exit 0
fi

kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$EPTP_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 16 -c 8192 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG" 180
warmup

echo "--- 16 concurrent requests (30 tokens each) ---"
PROMPTS=("math" "physics" "chemistry" "biology" "history" "geography"
         "music" "art" "literature" "philosophy" "astronomy" "cooking"
         "sports" "technology" "medicine" "architecture")

for i in $(seq 0 15); do
    run_inference "Write about ${PROMPTS[$i]}" 30 > "/tmp/ep16_r$((i+1)).json" &
done
wait

echo "--- Results ---"
python3 -c "
import json
total_tps = 0; clean = 0; garbled = 0
for i in range(1, 17):
    try:
        d = json.load(open(f'/tmp/ep16_r{i}.json'))
        t = d.get('timings', {})
        ms = t.get('predicted_per_token_ms', 0)
        tps = t.get('predicted_per_second', 0)
        c = d.get('choices', [{}])[0].get('message', {}).get('content', '')
        r = sum(1 for x in c if ord(x)<128)/max(len(c),1)
        ok = 'CLEAN' if r>0.8 and len(c)>5 else 'GARBLED'
        if ok == 'CLEAN': clean += 1
        else: garbled += 1
        total_tps += tps
        print(f'  Slot {i:2d}: {ms:6.0f}ms | {tps:5.1f} t/s | {ok} | {c[:40]}')
    except Exception as e:
        print(f'  Slot {i:2d}: FAILED ({e})')
        garbled += 1
n = clean + garbled
print()
print(f'  Clean: {clean}/16  Garbled: {garbled}/16')
print(f'  Aggregate: {total_tps:.1f} t/s')
print(f'  Per-user:  {total_tps/max(n,1):.1f} t/s')
if clean == 16:
    print('  ✅ EP np=16 FULLY WORKING!')
elif clean > 0:
    print('  ⚠️  Partial success — some slots garbled')
else:
    print('  ❌ All garbled — EP corruption not fixed')
"

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
