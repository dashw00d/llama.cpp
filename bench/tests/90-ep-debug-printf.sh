#!/bin/bash
# ============================================================
# TEST 90: EP Debug — Printf Tracing
# ============================================================
#
# WHAT: Runs EP with debug output to trace expert dispatch.
#       Adds stderr logging to see what each GPU is doing.
#
# WHY:  EP output is garbled. We need to see:
#       - What ne02 (local expert count) each GPU sees
#       - What expert_offset each GPU gets from op_params
#       - Which expert IDs pass the local filter
#       - Whether the local index remapping is correct
#
# HOW TO USE:
#   1. First, add debug prints to ggml-sycl.cpp (see below)
#   2. Rebuild: cd eptp/build-sycl && cmake --build . --target llama-server -j$(nproc)
#   3. Run this test
#   4. Check /tmp/bench_90-ep-debug-printf.log for EP DEBUG lines
#
# DEBUG PRINTS TO ADD (in ggml_sycl_mul_mat_id, after reading op_params):
#
#   // Add after line ~4205 in ggml-sycl.cpp:
#   static int ep_debug_count = 0;
#   if (ep_flag && ep_debug_count < 10) {
#       fprintf(stderr, "EP DEBUG: ne02=%ld expert_offset=%d n_local=%ld "
#               "src0_ne=[%ld,%ld,%ld,%ld] dst_ne=[%ld,%ld,%ld,%ld]\n",
#               ne02, expert_offset, n_local_experts,
#               src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
#               dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
#       ep_debug_count++;
#   }
#
# WHAT TO LOOK FOR:
#   - ne02 should be 32 (local experts per GPU), NOT 96
#   - expert_offset should be 0/32/64 for GPU 0/1/2
#   - src0_ne[2] should be 32 (matches ne02)
#   - If ne02 is 96: the simple tensor still has global shape → bug in meta backend tensor init
#   - If expert_offset is wrong: bug in meta backend offset calculation
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="90-ep-debug-printf"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="

if [ ! -x "$EPTP_SERVER" ]; then
    echo "SKIP: eptp build not found at $EPTP_SERVER"
    echo "Build with: cd $EPTP_WORKTREE/build-sycl && cmake --build . --target llama-server -j\$(nproc)"
    exit 0
fi

kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$EPTP_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG"

# Single short request to trigger EP dispatch
echo "--- Single request (10 tokens) ---"
RESULT=$(run_inference "What is 2+2?" 10)
echo "  $(echo "$RESULT" | parse_result)"

echo ""
echo "--- EP Debug output ---"
grep "EP DEBUG" "$LOG" 2>/dev/null | head -20 || echo "  No EP DEBUG lines found. Did you add the fprintf? See header comments."

echo ""
echo "--- Expert info from model load ---"
grep "n_expert" "$LOG" | head -3

kill_server
echo "=== DONE ==="
