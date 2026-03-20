#!/bin/bash
# ============================================================
# TEST 02: TP Baseline — 0.6B Dense
# ============================================================
#
# WHAT: Tensor Parallelism on Qwen3-0.6B Q8_0 (dense, no MoE)
#       using batched command lists, no graphs.
#
# WHY:  Fast iteration model. Tests TP, AllReduce, batched
#       dispatch without MoE complexity. Also the baseline
#       for graph speedup tests (see test 03).
#
# EXPECTED: ~54ms/token, ~18.5 t/s, CLEAN output
#           57 subgraphs
#
# NOTES: This model loads in ~2 seconds. Perfect for quick
#        sanity checks after code changes.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="02-tp-baseline-06b"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_06B" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"
warmup

echo "--- Benchmark (3 runs) ---"
for i in 1 2 3; do
    RESULT=$(run_inference "Write about physics" 100)
    echo "  Run $i: $(echo "$RESULT" | parse_result)"
done

save_result "$TEST_NAME" "$RESULT" '{"model":"0.6b","config":"batched","np":1}'

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
