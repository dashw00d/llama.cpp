#!/bin/bash
# ============================================================
# TEST 30: EP vs TP — Same Model, Same Hardware
# ============================================================
#
# WHAT: Side-by-side comparison of Expert Parallelism vs Tensor
#       Parallelism on the 96-expert REAM model.
#
# WHY:  EP splits experts across GPUs (32 each) instead of
#       splitting weight matrices. This eliminates MoE AllReduce
#       (48 fewer sync points per token) and reduces per-GPU
#       compute (only 2-3 active experts per GPU vs 8).
#
# EXPECTED:
#   TP:  ~330ms, ~3.0 t/s, CLEAN output (verified working)
#   EP:  ~250ms, ~3.95 t/s, ??? output (WAS garbled — needs fix)
#
# EP CORRUPTION STATUS (as of 2026-03-19):
#   Root cause investigation identified 5 potential bugs:
#   1. Tensor rotation vs expert_offset mismatch (TESTED — didn't fix alone)
#   2. op_params slot mismatch (latent, works by coincidence)
#   3. ne02 on simple tensor may not match local expert count
#   4. Fused aggregation may not handle EP correctly
#   5. Rotation not applied in offset calculation
#   See docs/EP-DEBUG.md for full debug guide.
#
# THIS TEST WILL:
#   1. Run TP on stable build → confirm clean baseline
#   2. Run EP on eptp build → check if output is clean or garbled
#   3. Compare speed and output quality
#
# IF EP OUTPUT IS STILL GARBLED: the bug is NOT fixed yet.
#   See docs/EP-DEBUG.md "Debug Strategy" for next steps.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="30-ep-vs-tp-ream"
LOG_TP="/tmp/bench_${TEST_NAME}_tp.log"
LOG_EP="/tmp/bench_${TEST_NAME}_ep.log"

echo "=== $TEST_NAME ==="

# --- Test A: TP (stable build) ---
echo ""
echo "--- A: Tensor Parallelism (stable build) ---"
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG_TP" 2>&1 &

wait_ready "$LOG_TP"
warmup

RESULT_TP=$(run_inference "What is 2+2? Explain your reasoning step by step." 50)
echo "  TP: $(echo "$RESULT_TP" | parse_result)"
kill_server

# --- Test B: EP (eptp build) ---
echo ""
echo "--- B: Expert Parallelism (eptp build) ---"

if [ ! -x "$EPTP_SERVER" ]; then
    echo "  SKIP: eptp build not found at $EPTP_SERVER"
    echo "  Build with: cd $EPTP_WORKTREE/build-sycl && cmake --build . --target llama-server -j\$(nproc)"
    exit 0
fi

"$EPTP_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG_EP" 2>&1 &

wait_ready "$LOG_EP"
warmup

RESULT_EP=$(run_inference "What is 2+2? Explain your reasoning step by step." 50)
echo "  EP: $(echo "$RESULT_EP" | parse_result)"

echo ""
echo "--- Comparison ---"
echo "  TP: $(echo "$RESULT_TP" | parse_result)"
echo "  EP: $(echo "$RESULT_EP" | parse_result)"
echo ""

# Check if EP is clean
EP_CLEAN=$(echo "$RESULT_EP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
c=d['choices'][0]['message']['content']
r=sum(1 for x in c if ord(x)<128)/max(len(c),1)
print('PASS' if r>0.8 and len(c)>5 else 'FAIL')
" 2>/dev/null)

if [ "$EP_CLEAN" = "PASS" ]; then
    echo "  ✅ EP OUTPUT IS CLEAN — corruption bug is FIXED!"
else
    echo "  ❌ EP OUTPUT IS GARBLED — see docs/EP-DEBUG.md"
fi

echo "--- META ---"
echo "  TP META:"
grep META "$LOG_TP" | tail -1
echo "  EP META:"
grep META "$LOG_EP" | tail -1

kill_server
echo "=== DONE ==="
