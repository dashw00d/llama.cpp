#!/bin/bash
# ============================================================
# Common environment for all bench tests
# Source this at the top of every test script:
#   source "$(dirname "$0")/../env.sh"
# ============================================================

# SYCL environment (Intel oneAPI + conda)
source /home/ryan/llm-stack/env.sglang-xpu.sh

# Paths
export LLM_STACK="/home/ryan/llm-stack"
export STABLE_WORKTREE="$LLM_STACK/llama.cpp-stable"
export EPTP_WORKTREE="$LLM_STACK/llama.cpp-eptp"
export STABLE_SERVER="$STABLE_WORKTREE/build-sycl/bin/llama-server"
export EPTP_SERVER="$EPTP_WORKTREE/build-sycl/bin/llama-server"
export RESULTS_DIR="$STABLE_WORKTREE/bench/results"
mkdir -p "$RESULTS_DIR"

# Models
export MODEL_06B="$LLM_STACK/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
export MODEL_27B_MOE="$LLM_STACK/models/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf"
export MODEL_30B_MOE="$LLM_STACK/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf"
export MODEL_30B_REAM="$LLM_STACK/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf"

# Default port
export BENCH_PORT=18404

# ============================================================
# Helper functions
# ============================================================

kill_server() {
    pkill -x llama-server 2>/dev/null || true
    sleep 2
    if pgrep -x llama-server >/dev/null; then
        echo "ERROR: llama-server still running!" >&2
        return 1
    fi
}

wait_ready() {
    local log="$1" timeout="${2:-120}"
    local start=$SECONDS
    while true; do
        grep -q "all slots are idle" "$log" 2>/dev/null && return 0
        if [ $((SECONDS - start)) -gt "$timeout" ]; then
            echo "TIMEOUT waiting for server (${timeout}s)" >&2
            tail -5 "$log" >&2
            return 1
        fi
        sleep 2
    done
}

warmup() {
    local port="${1:-$BENCH_PORT}" count="${2:-3}"
    for i in $(seq 1 "$count"); do
        curl -s --max-time 120 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"warmup","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' \
            > /dev/null 2>/dev/null
    done
}

run_inference() {
    # Usage: run_inference <prompt> <max_tokens> [port]
    local prompt="$1" max_tokens="$2" port="${3:-$BENCH_PORT}"
    curl -s --max-time 300 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"bench\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens}" \
        2>/dev/null
}

parse_result() {
    # Usage: echo "$json" | parse_result
    # Outputs: ms_per_tok t/s tokens content_preview is_clean
    python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
ms = t.get('predicted_per_token_ms', 0)
tps = t.get('predicted_per_second', 0)
tokens = t.get('predicted_n', 0)
c = d.get('choices', [{}])[0].get('message', {}).get('content', '')
ascii_r = sum(1 for x in c if ord(x) < 128) / max(len(c), 1)
clean = 'CLEAN' if ascii_r > 0.8 and len(c) > 5 else 'GARBLED'
print(f'{ms:.0f}ms | {tps:.1f} t/s | {tokens} tok | {clean} | {c[:60]}')
"
}

save_result() {
    # Usage: save_result <test_name> <json_string> [extra_fields]
    local test_name="$1" result="$2" extra="${3:-{}}"
    local timestamp=$(date +%Y-%m-%d_%H%M%S)
    local outfile="$RESULTS_DIR/${timestamp}_${test_name}.json"
    python3 -c "
import json, sys
result = json.loads('''$result''')
extra = json.loads('''$extra''')
t = result.get('timings', {})
c = result.get('choices', [{}])[0].get('message', {}).get('content', '')
ascii_r = sum(1 for x in c if ord(x) < 128) / max(len(c), 1)
output = {
    'test': '$test_name',
    'timestamp': '$timestamp',
    'ms_per_tok': t.get('predicted_per_token_ms', 0),
    'tps': t.get('predicted_per_second', 0),
    'tokens': t.get('predicted_n', 0),
    'clean': ascii_r > 0.8 and len(c) > 5,
    'content_preview': c[:100],
    **extra,
    'raw_timings': t
}
json.dump(output, open('$outfile', 'w'), indent=2)
print(f'Saved: $outfile')
" 2>/dev/null
}
