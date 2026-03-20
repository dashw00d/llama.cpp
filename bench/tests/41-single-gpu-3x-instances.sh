#!/bin/bash
# ============================================================
# TEST 41: 3x Independent Instances — One Per GPU
# ============================================================
#
# WHAT: Run 3 separate llama-server instances, each pinned to
#       one Arc A770 via ZE_AFFINITY_MASK. Each serves independently.
#
# WHY:  If the REAM model fits on 1 GPU (13GB < 16GB), we can
#       run 3 copies simultaneously with ZERO cross-GPU sync.
#       Each instance has its own KV cache, its own dispatch,
#       and no AllReduce at all.
#
#       Theoretical aggregate: 3 × single-GPU t/s
#       vs TP aggregate at np=3: much lower due to AllReduce
#
# EXPECTED: Unknown — first test.
#       If single GPU gets 4 t/s → aggregate 12 t/s (3×4)
#       vs TP np=3 aggregate: ~4-5 t/s
#       That would be a 2.5-3x throughput improvement.
#
# ARCHITECTURE:
#       GPU0 → llama-server :18410 (user requests 1, 4, 7, ...)
#       GPU1 → llama-server :18411 (user requests 2, 5, 8, ...)
#       GPU2 → llama-server :18412 (user requests 3, 6, 9, ...)
#       Load balancer (round-robin) in front
#
# TRADEOFF:
#       - 3x VRAM usage (3 copies of model weights)
#       - Each instance limited to ~3GB KV cache
#       - No shared prompt cache across instances
#       - Only works for models that fit in single GPU VRAM
#
# NOTE: arcllm-proxy could orchestrate this in production.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="41-single-gpu-3x-instances"
LOG0="/tmp/bench_${TEST_NAME}_gpu0.log"
LOG1="/tmp/bench_${TEST_NAME}_gpu1.log"
LOG2="/tmp/bench_${TEST_NAME}_gpu2.log"

echo "=== $TEST_NAME ==="
kill_server

if [ ! -f "$MODEL_30B_REAM" ]; then
    echo "SKIP: REAM model not found"
    exit 0
fi

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

echo "--- Starting 3 instances (one per GPU) ---"

ZE_AFFINITY_MASK=0 "$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    -ngl 99 -np 1 -c 512 --port 18410 --no-warmup > "$LOG0" 2>&1 &
PID0=$!

ZE_AFFINITY_MASK=1 "$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    -ngl 99 -np 1 -c 512 --port 18411 --no-warmup > "$LOG1" 2>&1 &
PID1=$!

ZE_AFFINITY_MASK=2 "$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    -ngl 99 -np 1 -c 512 --port 18412 --no-warmup > "$LOG2" 2>&1 &
PID2=$!

echo "  GPU0 PID=$PID0 :18410"
echo "  GPU1 PID=$PID1 :18411"
echo "  GPU2 PID=$PID2 :18412"

# Wait for all 3
for LOG in "$LOG0" "$LOG1" "$LOG2"; do
    wait_ready "$LOG" 180 || true
done
echo "All 3 ready"

# Warmup each
for PORT in 18410 18411 18412; do
    warmup $PORT 2
done

echo "--- 3 concurrent requests (one per GPU) ---"
run_inference "Write about math" 50 18410 > /tmp/3x_r1.json &
run_inference "Write about physics" 50 18411 > /tmp/3x_r2.json &
run_inference "Write about biology" 50 18412 > /tmp/3x_r3.json &
wait

echo "--- Results ---"
for i in 1 2 3; do
    echo -n "  GPU$((i-1)): "
    cat "/tmp/3x_r${i}.json" | parse_result
done

python3 -c "
import json
tot=0; n=0
for i in range(1,4):
    try:
        d=json.load(open(f'/tmp/3x_r{i}.json'))
        t=d['timings']['predicted_per_second']
        tot+=t; n+=1
    except: pass
print(f'  Aggregate: {tot:.1f} t/s | Per-instance: {tot/max(n,1):.1f} t/s')
print(f'  vs TP np=3 estimate: ~4-5 t/s aggregate')
"

echo "--- META per GPU ---"
for LOG in "$LOG0" "$LOG1" "$LOG2"; do
    grep META "$LOG" | tail -1
done

# Cleanup all 3
kill $PID0 $PID1 $PID2 2>/dev/null || true
sleep 2
export ZE_AFFINITY_MASK=0,1,2
echo "=== DONE ==="
