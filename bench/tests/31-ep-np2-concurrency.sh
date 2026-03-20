#!/bin/bash
# ============================================================
# TEST 31: EP np=2 Concurrency
# ============================================================
#
# WHAT: EP with 2 concurrent users. Tests the Q8 cache
#       poisoning fix (invalidate_q8_cache per expert).
#
# WHY:  Before the fix, np=2 produced garbled output because
#       the Q8 quantization cache keyed on pointer address.
#       src1_contiguous is reused across experts with different
#       content, but same pointer → stale quantized data from
#       Expert A used for Expert B's matmul.
#
# FIX:  ctx.invalidate_q8_cache() before each expert matmul
#       in the batch path (2 lines added).
#
# BLOCKED BY: OQ-1 (EP np=1 corruption). Fix np=1 first.
#
# EXPECTED (when np=1 is fixed):
#   Per-user: ~300-400ms (only 13-25% slower than np=1)
#   Aggregate: ~5-6 t/s
#   vs TP np=2: ~640ms per user (2x degradation)
#   EP should show MUCH better scaling than TP.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="31-ep-np2-concurrency"
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
    --split-mode tensor -ngl 99 -np 2 -c 1024 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"
warmup

echo "--- 2 concurrent requests ---"
run_inference "What is 2+2? Answer briefly." 30 > /tmp/ep_np2_r1.json &
run_inference "What is 3+3? Answer briefly." 30 > /tmp/ep_np2_r2.json &
wait

for i in 1 2; do
    echo -n "  Slot $i: "
    cat "/tmp/ep_np2_r${i}.json" | parse_result
done

# Check quality
python3 -c "
import json
clean=0
for i in range(1,3):
    d=json.load(open(f'/tmp/ep_np2_r{i}.json'))
    c=d['choices'][0]['message']['content']
    r=sum(1 for x in c if ord(x)<128)/max(len(c),1)
    if r>0.8 and len(c)>5: clean+=1
print(f'  Clean: {clean}/2')
if clean==2: print('  ✅ EP np=2 WORKS!')
else: print('  ❌ EP np=2 still garbled — check OQ-1')
"

kill_server
echo "=== DONE ==="
