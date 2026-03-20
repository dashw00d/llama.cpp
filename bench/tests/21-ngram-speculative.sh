#!/bin/bash
# ============================================================
# TEST 21: N-Gram Self-Speculative Decoding
# ============================================================
#
# WHAT: Self-speculative decoding using n-gram pattern matching.
#       No draft model needed — reuses the target model's own
#       prior output to predict next tokens.
#
# WHY:  Draft-model speculative decoding FAILED because Qwen3-0.6B
#       and Qwen3-30B have incompatible tokenizers (0% acceptance).
#       N-gram mode avoids this entirely — zero VRAM overhead,
#       zero tokenizer risk. Estimated 5-15% gain on repetitive
#       content (code, structured text).
#
# EXPECTED: Baseline ~3.0 t/s → maybe 3.3-3.5 t/s (+10-15%)
#           Works best on repetitive/structured prompts.
#           May show zero gain on creative/diverse prompts.
#
# MODES AVAILABLE:
#   --spec-type ngram-mod    — modular n-gram (lightest, 16MB pool)
#   --spec-type ngram-simple — simple frequency counting
#   --spec-type ngram-cache  — cached n-gram lookup
#
# NOTE: This is COMPLETELY UNTESTED as of 2026-03-19.
#       We're the first to try it on this stack.
#
# STATUS: OQ-5 in OPEN-QUESTIONS.md — "untested free fruit"
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="21-ngram-speculative"
LOG_BASE="/tmp/bench_${TEST_NAME}_base.log"
LOG_NGRAM="/tmp/bench_${TEST_NAME}_ngram.log"

echo "=== $TEST_NAME ==="

# Use a repetitive/structured prompt for best n-gram chance
PROMPT="Write a numbered list of 20 facts about water. Number each fact."

# --- Baseline (no speculation) ---
echo ""
echo "--- A: Baseline (no speculation) ---"
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 1 -c 2048 \
    --port $BENCH_PORT --no-warmup > "$LOG_BASE" 2>&1 &

wait_ready "$LOG_BASE"
warmup

RESULT_BASE=$(run_inference "$PROMPT" 200)
echo "  Baseline: $(echo "$RESULT_BASE" | parse_result)"
kill_server

# --- N-gram speculative ---
echo ""
echo "--- B: N-gram speculative (ngram-mod) ---"

# Check if server supports --spec-type
if "$STABLE_SERVER" --help 2>&1 | grep -q "spec-type"; then
    "$STABLE_SERVER" -m "$MODEL_30B_REAM" \
        --split-mode tensor -ngl 99 -np 1 -c 2048 \
        --spec-type ngram-mod --draft-max 48 --spec-ngram-size-n 16 \
        --port $BENCH_PORT --no-warmup > "$LOG_NGRAM" 2>&1 &

    wait_ready "$LOG_NGRAM"
    warmup

    RESULT_NGRAM=$(run_inference "$PROMPT" 200)
    echo "  N-gram:   $(echo "$RESULT_NGRAM" | parse_result)"

    echo ""
    echo "--- Comparison ---"
    echo "  Baseline: $(echo "$RESULT_BASE" | parse_result)"
    echo "  N-gram:   $(echo "$RESULT_NGRAM" | parse_result)"

    # Check for speculation stats in log
    echo "--- Speculation stats ---"
    grep -i "accept\|reject\|draft\|spec" "$LOG_NGRAM" | tail -5
else
    echo "  SKIP: --spec-type not supported in this build"
fi

kill_server
echo "=== DONE ==="
