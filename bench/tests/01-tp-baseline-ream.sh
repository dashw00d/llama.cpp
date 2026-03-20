#!/bin/bash
# ============================================================
# TEST 01: TP Baseline — 96-expert REAM model
# ============================================================
#
# WHAT: Tensor Parallelism on 30B REAM (96 experts, 8 active)
#       using batched command lists, no graphs.
#
# WHY:  This is our production baseline. Every other test
#       compares against this number. If this regresses,
#       something fundamental broke.
#
# EXPECTED: ~330ms/token, ~3.0 t/s, CLEAN output
#           97 subgraphs, ~3.4ms/subgraph
#
# HISTORY: Started at 2.42 t/s (128-expert, immediate mode)
#          Batched cmdlists: +20% → 2.9 t/s
#          REAM model (smaller): 3.0-3.5 t/s
#
# CONFIG:  GGML_SYCL_DISABLE_GRAPH=1 (no graphs — MoE breaks them)
#          SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 (batched)
#          --split-mode tensor (TP across 3 GPUs)
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="01-tp-baseline-ream"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"
warmup

echo "--- Benchmark (3 runs) ---"
for i in 1 2 3; do
    RESULT=$(run_inference "Write a paragraph about mathematics" 100)
    echo "  Run $i: $(echo "$RESULT" | parse_result)"
done

# Save last run
save_result "$TEST_NAME" "$RESULT" '{"model":"30b-ream","config":"batched","np":1}'

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
