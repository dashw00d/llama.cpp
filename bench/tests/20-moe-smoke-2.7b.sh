#!/bin/bash
# ============================================================
# TEST 20: MoE Smoke Test — 2.7B MoE (Qwen1.5-MoE-A2.7B)
# ============================================================
#
# WHAT: Quick MoE code path validation on a tiny model.
#       64 experts (4 shared + 60 routed), 2.7B active params.
#       Q2_K quantization, ~5.9GB total.
#
# WHY:  The 30B models take 10-30 seconds to load. This model
#       loads in 2-3 seconds. Use it for:
#       - Validating MoE dispatch changes don't crash
#       - Quick output quality check after code changes
#       - Testing MUL_MAT_ID code path without waiting
#
# EXPECTED: Fast load, clean output, MoE dispatch exercised.
#           Speed will be high (small model) — exact number TBD
#           (this is the first time we're testing this model).
#
# NOTE: 64 experts is NOT divisible by 3, so EP won't work
#       on this model. TP only. But it exercises the same
#       MUL_MAT_ID code path that EP modifies.
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="20-moe-smoke-2.7b"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="

if [ ! -f "$MODEL_27B_MOE" ]; then
    echo "SKIP: 2.7B MoE model not found at $MODEL_27B_MOE"
    echo "Download: curl -L -o '$MODEL_27B_MOE' 'https://huggingface.co/RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf/resolve/main/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf'"
    exit 0
fi

kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_27B_MOE" \
    --split-mode tensor -ngl 99 -np 1 -c 512 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG" 60

echo "--- Model info ---"
grep "n_expert" "$LOG" | head -3

warmup

echo "--- Benchmark (3 runs) ---"
for i in 1 2 3; do
    RESULT=$(run_inference "What is the capital of France?" 30)
    echo "  Run $i: $(echo "$RESULT" | parse_result)"
done

save_result "$TEST_NAME" "$RESULT" '{"model":"2.7b-moe","config":"batched","np":1}'

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
