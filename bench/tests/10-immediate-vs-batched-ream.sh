#!/bin/bash
# ============================================================
# TEST 10: A/B — Immediate vs Batched Command Lists (REAM)
# ============================================================
#
# WHAT: Direct comparison of immediate vs batched dispatch
#       on the 30B REAM model.
#
# WHY:  Batched command lists (SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_
#       COMMANDLISTS=0) batch kernel submissions into L0 command
#       lists before hardware submission. This reduces per-kernel
#       overhead from ~61µs to ~48µs (21% reduction).
#
# EXPECTED:
#   Immediate: ~400ms/token, ~2.5 t/s
#   Batched:   ~330ms/token, ~3.0 t/s (+20%)
#
# HISTORY: This was one of the first major wins. Discovered that
#   batched mode was 30% faster for inference but made loading
#   10x slower. The loading issue was fixed by removing per-tensor
#   .wait() serialization (see TECHNIQUES.md D-2).
#
# SIDE EFFECT: Batched mode compiles DIFFERENT L0 kernels than
#   immediate mode. The NEO compiler cache is mode-specific.
#   First run after switching modes will be slower (JIT).
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="10-immediate-vs-batched-ream"
LOG_IMM="/tmp/bench_${TEST_NAME}_imm.log"
LOG_BAT="/tmp/bench_${TEST_NAME}_bat.log"

echo "=== $TEST_NAME ==="

# --- Test A: Immediate ---
echo ""
echo "--- A: Immediate mode ---"
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # IMMEDIATE

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG_IMM" 2>&1 &

wait_ready "$LOG_IMM"
warmup

for i in 1 2 3; do
    RESULT_IMM=$(run_inference "Write about biology" 50)
    echo "  Immediate $i: $(echo "$RESULT_IMM" | parse_result)"
done

META_IMM=$(grep META "$LOG_IMM" | tail -1)
echo "  META: $META_IMM"
kill_server

# --- Test B: Batched ---
echo ""
echo "--- B: Batched mode ---"

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0  # BATCHED

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG_BAT" 2>&1 &

wait_ready "$LOG_BAT"
warmup

for i in 1 2 3; do
    RESULT_BAT=$(run_inference "Write about biology" 50)
    echo "  Batched $i: $(echo "$RESULT_BAT" | parse_result)"
done

META_BAT=$(grep META "$LOG_BAT" | tail -1)
echo "  META: $META_BAT"
kill_server

echo ""
echo "--- Comparison ---"
echo "  Immediate: $(echo "$RESULT_IMM" | parse_result)"
echo "  Batched:   $(echo "$RESULT_BAT" | parse_result)"
echo "=== DONE ==="
