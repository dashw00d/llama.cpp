#!/bin/bash
# ============================================================
# TEST 04: TP Scaling — REAM np=4
# ============================================================
#
# WHAT: 4 concurrent users on 96-expert REAM with TP.
#
# WHY:  Tests how TP handles concurrent requests. Under TP,
#       ALL GPUs process ALL experts for every token. With
#       np>1, the batch dimension grows → more compute per
#       META dispatch → but same AllReduce overhead.
#
# EXPECTED: Per-user ~600-800ms (2x slower than np=1)
#           Aggregate ~5-6 t/s
#           This is the TP scaling limitation that EP was
#           designed to solve.
#
# COMPARISON: EP np=4 should give better per-user latency
#             because each GPU only computes its 32 local
#             experts, and batching amortizes the dispatch.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="04-tp-scaling-ream-np4"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 4 -c 2048 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"
warmup

echo "--- 4 concurrent requests ---"
PROMPTS=("Write about math" "Write about physics" "Write about history" "Write about art")
for i in 0 1 2 3; do
    run_inference "${PROMPTS[$i]}" 50 > "/tmp/np4_r$((i+1)).json" &
done
wait

echo "--- Results ---"
TOTAL_TPS=0
for i in 1 2 3 4; do
    echo -n "  Slot $i: "
    cat "/tmp/np4_r${i}.json" | parse_result
done

python3 -c "
import json
tot=0; n=0
for i in range(1,5):
    try:
        d=json.load(open(f'/tmp/np4_r{i}.json'))
        t=d['timings']['predicted_per_second']
        tot+=t; n+=1
    except: pass
print(f'  Aggregate: {tot:.1f} t/s | Per-user: {tot/max(n,1):.1f} t/s | Slots: {n}/4')
"

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
