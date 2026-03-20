#!/bin/bash
set -euo pipefail

# Usage: ./bench/run-bench.sh <model> <config> [np] [max_tokens]
# model: 0.6b | 30b-moe | 30b-ream
# config: immediate | batched | graphs
# np: number of parallel slots (default 1)
# max_tokens: tokens to generate per request (default 100)

MODEL_NAME="${1:?Usage: run-bench.sh <model> <config> [np] [max_tokens]}"
CONFIG="${2:?Usage: run-bench.sh <model> <config> [np] [max_tokens]}"
NP="${3:-1}"
MAX_TOKENS="${4:-100}"
PORT=18404
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE="$(dirname "$SCRIPT_DIR")"
SERVER="$WORKTREE/build-sycl/bin/llama-server"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Model paths
case "$MODEL_NAME" in
  0.6b)     MODEL="/home/ryan/llm-stack/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf" ;;
  30b-moe)  MODEL="/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf" ;;
  30b-ream) MODEL="/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf" ;;
  2.7b-moe) MODEL="/home/ryan/llm-stack/models/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf" ;;
  *)        echo "Unknown model: $MODEL_NAME (use 0.6b, 2.7b-moe, 30b-moe, 30b-ream)"; exit 1 ;;
esac

# Config env vars
case "$CONFIG" in
  immediate) export GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 ;;
  batched)   export GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 ;;
  graphs)    export GGML_SYCL_DISABLE_GRAPH=0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 ;;
  *)         echo "Unknown config: $CONFIG (use immediate, batched, graphs)"; exit 1 ;;
esac

# Context size scales with np
CTX=$((512 * NP))
[ "$CTX" -lt 512 ] && CTX=512

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/${TIMESTAMP}_${MODEL_NAME}_${CONFIG}_np${NP}.json"
LOG_FILE="/tmp/bench_${TIMESTAMP}.log"

echo "============================================"
echo "  BENCHMARK: $MODEL_NAME / $CONFIG / np=$NP"
echo "  Max tokens: $MAX_TOKENS | Context: $CTX"
echo "  Result: $RESULT_FILE"
echo "============================================"

# Pre-test cleanup
pkill -x llama-server 2>/dev/null || true
sleep 2
if pgrep -x llama-server >/dev/null; then
    echo "ERROR: llama-server still running!" >&2; exit 1
fi

# Start server
source /home/ryan/llm-stack/env.sglang-xpu.sh
"$SERVER" -m "$MODEL" --split-mode tensor -ngl 99 -np "$NP" -c "$CTX" --port "$PORT" --no-warmup \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for ready
START=$SECONDS
while true; do
    if grep -q "all slots are idle" "$LOG_FILE" 2>/dev/null; then
        LOAD_TIME=$((SECONDS - START))
        echo "Ready in ${LOAD_TIME}s"
        break
    fi
    if [ $((SECONDS - START)) -gt 300 ]; then
        echo "TIMEOUT waiting for server"
        tail -10 "$LOG_FILE"
        kill "$SERVER_PID" 2>/dev/null
        exit 1
    fi
    sleep 2
done

# Warmup: 3 sequential requests (gets past JIT compilation)
echo "Warmup (3 sequential)..."
for i in 1 2 3; do
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":10}" > /dev/null 2>/dev/null
done
echo "Warmup done"

# Benchmark: fire NP concurrent requests
echo "Benchmark: $NP concurrent requests, $MAX_TOKENS tokens each..."
PROMPTS=(
    "Write about mathematics" "Write about physics" "Write about chemistry"
    "Write about biology" "Write about history" "Write about geography"
    "Write about music" "Write about art" "Write about literature"
    "Write about philosophy" "Write about astronomy" "Write about cooking"
    "Write about sports" "Write about technology" "Write about medicine"
    "Write about architecture"
)

BENCH_START=$(date +%s%N)
for i in $(seq 1 "$NP"); do
    IDX=$(( (i - 1) % ${#PROMPTS[@]} ))
    PROMPT="${PROMPTS[$IDX]}"
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}" \
        > "/tmp/bench_r${i}.json" 2>/dev/null &
done
wait
BENCH_END=$(date +%s%N)
WALL_MS=$(( (BENCH_END - BENCH_START) / 1000000 ))

# Collect results
echo ""
echo "--- Results ---"
python3 -c "
import json, sys, os

results = []
total_tps = 0
total_tokens = 0
clean = 0
garbled = 0

for i in range(1, $NP + 1):
    try:
        d = json.load(open(f'/tmp/bench_r{i}.json'))
        t = d.get('timings', {})
        ms = t.get('predicted_per_token_ms', 0)
        tps = t.get('predicted_per_second', 0)
        tokens = t.get('predicted_n', 0)
        content = d.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Basic garble detection: high ratio of non-ASCII or very short
        ascii_ratio = sum(1 for c in content if ord(c) < 128) / max(len(content), 1)
        is_clean = ascii_ratio > 0.8 and len(content) > 10
        
        status = 'clean' if is_clean else 'GARBLED'
        if is_clean: clean += 1
        else: garbled += 1
        
        total_tps += tps
        total_tokens += tokens
        results.append({'slot': i, 'ms_per_tok': ms, 'tps': tps, 'tokens': tokens, 'status': status, 'content_preview': content[:60]})
        print(f'  Slot {i:2d}: {ms:7.1f} ms/tok | {tps:6.2f} t/s | {tokens:3d} tok | {status} | {content[:40]}')
    except Exception as e:
        results.append({'slot': i, 'error': str(e)})
        print(f'  Slot {i:2d}: FAILED ({e})')

print()
print(f'  Clean: {clean}/{$NP}  Garbled: {garbled}/{$NP}')
print(f'  Aggregate throughput: {total_tps:.1f} t/s')
print(f'  Per-user average: {total_tps / max(clean + garbled, 1):.1f} t/s')
print(f'  Wall time: {$WALL_MS}ms')
print(f'  Total tokens: {total_tokens}')

# Save full result
output = {
    'timestamp': '$TIMESTAMP',
    'model': '$MODEL_NAME',
    'config': '$CONFIG',
    'np': $NP,
    'max_tokens': $MAX_TOKENS,
    'context': $CTX,
    'load_time_s': $LOAD_TIME,
    'wall_time_ms': $WALL_MS,
    'aggregate_tps': total_tps,
    'per_user_tps': total_tps / max(clean + garbled, 1),
    'clean_slots': clean,
    'garbled_slots': garbled,
    'total_tokens': total_tokens,
    'slots': results
}
json.dump(output, open('$RESULT_FILE', 'w'), indent=2)
print(f'  Saved: $RESULT_FILE')
"

echo ""
echo "--- META ---"
grep META "$LOG_FILE" | tail -3

# Cleanup
kill "$SERVER_PID" 2>/dev/null || true
sleep 2
echo "Done."
