#!/bin/bash
# ============================================================
# TEST 03: TP + Graphs — 0.6B Dense
# ============================================================
#
# WHAT: SYCL graph recording + replay on 0.6B dense model.
#
# WHY:  Graphs eliminate kernel launch overhead by recording
#       the full dispatch sequence and replaying it. On 0.6B:
#       - 57 subgraphs × ~0.6ms replay = ~34ms total
#       - vs ~54ms without graphs = +53% speedup
#       - Also eliminates warmup: first request runs at full speed
#
# EXPECTED: ~35ms/token, ~26-28 t/s, CLEAN output
#           99% graph replay rate
#
# WHY ONLY 0.6B: Graphs DON'T work on larger models:
#   - 30B MoE: MoE expert routing shifts tensor pointers every
#     token → graph replay sees pointer mismatch → falls back
#     to immediate dispatch. Net effect: +3% overhead (SLOWER).
#   - 32B dense: needs ~192 graphs/device but Arc A770 hardware
#     limit is ~127. Cache thrashing = 0% replay, SLOWER.
#   - 0.6B needs ~57 graphs/device → fits within limit → works.
#
# MEASUREMENT NOTE: The META timing (28.6 t/s) excludes AllReduce.
#   The eval timing (~26 t/s) includes AllReduce. Use eval timing
#   for apples-to-apples comparison with other tests.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="03-tp-graphs-06b"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=0  # <-- GRAPHS ENABLED
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_06B" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"
warmup

echo "--- Benchmark (3 runs) ---"
for i in 1 2 3; do
    RESULT=$(run_inference "Write about chemistry" 100)
    echo "  Run $i: $(echo "$RESULT" | parse_result)"
done

save_result "$TEST_NAME" "$RESULT" '{"model":"0.6b","config":"graphs","np":1}'

echo "--- META ---"
grep META "$LOG" | tail -3
echo "--- Graph stats ---"
grep -i "graph\|replay\|record\|finalize" "$LOG" | tail -5
kill_server
echo "=== DONE ==="
