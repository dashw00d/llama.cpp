#!/bin/bash
# ============================================================
# TEST 40: Single GPU — REAM Model (No Cross-GPU Sync)
# ============================================================
#
# WHAT: Run the 96-expert REAM model entirely on ONE Arc A770.
#       13.18GB model fits in 16GB VRAM with ~3GB left for KV.
#
# WHY:  With TP across 3 GPUs, the token time budget is:
#         57.5% kernel launch overhead
#         14.0% AllReduce (cross-GPU sync)
#          0.6% actual GPU math
#       
#       On a single GPU, AllReduce is ELIMINATED entirely.
#       No cross-GPU communication, no host staging, no sync waits.
#       
#       MoE active reads per token: ~1.46GB (485MB × 3 GPUs with TP)
#       On single GPU: ~485MB active reads against 560 GB/s = ~0.87ms
#       Theoretical single-GPU decode: ~186 t/s (!!!)
#       Even at 1% utilization: faster than TP's 3.0 t/s
#
# EXPECTED: Unknown — this is the FIRST test of this config.
#       Likely limited by kernel launch overhead on single GPU.
#       But no AllReduce = -48ms/token = potentially 4-5 t/s.
#       KV cache limited to ~3GB (~512 context tokens).
#
# TRADEOFF: Single GPU = can't run larger models.
#       This only works because REAM is 13GB (fits in 16GB).
#       The 128-expert model (~18GB) does NOT fit on one GPU.
#
# FUTURE: Could use GPU1+GPU2 for KV cache overflow or
#       run 3 independent instances (see test 41).
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="40-single-gpu-ream"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export ZE_AFFINITY_MASK=0  # SINGLE GPU ONLY

# Use -ngl 99 but only 1 GPU available
# --split-mode none would be ideal but may not be supported
# Try row split with 1 device
"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG" 120

echo "--- Model load info ---"
grep "n_expert\|VRAM\|model size\|total" "$LOG" | head -5

warmup 18404 2  # fewer warmup since single GPU

echo "--- Benchmark (3 runs) ---"
for i in 1 2 3; do
    RESULT=$(run_inference "What is 2+2? Explain briefly." 30)
    echo "  Run $i: $(echo "$RESULT" | parse_result)"
done

save_result "$TEST_NAME" "$RESULT" '{"model":"30b-ream","config":"single-gpu","np":1,"gpus":1}'

echo "--- META ---"
grep META "$LOG" | tail -3

echo ""
echo "--- Comparison targets ---"
echo "  TP (3 GPU): ~330ms, ~3.0 t/s (with AllReduce overhead)"
echo "  Single GPU: see above (no AllReduce)"

kill_server
# Restore multi-GPU
export ZE_AFFINITY_MASK=0,1,2
echo "=== DONE ==="
