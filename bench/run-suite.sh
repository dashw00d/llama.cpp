#!/bin/bash
set -euo pipefail

# Full benchmark suite — runs all model/config/np combinations
# Usage: ./bench/run-suite.sh [model_filter]
# model_filter: 0.6b | 30b-moe | 30b-ream | all (default: all)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTER="${1:-all}"

echo "=========================================="
echo "  FULL BENCHMARK SUITE"
echo "  Filter: $FILTER"
echo "  Started: $(date)"
echo "=========================================="

run() {
    local model="$1" config="$2" np="$3" tokens="$4"
    echo ""
    echo ">>> $model / $config / np=$np / ${tokens}tok"
    bash "$SCRIPT_DIR/run-bench.sh" "$model" "$config" "$np" "$tokens"
    sleep 5  # Cool-down between tests
}

# 0.6B Dense — fast, good for TP/graph/batched testing
if [ "$FILTER" = "all" ] || [ "$FILTER" = "0.6b" ]; then
    echo ""
    echo "=== 0.6B DENSE ==="
    run 0.6b immediate 1 100
    run 0.6b batched   1 100
    run 0.6b graphs    1 100
    run 0.6b batched   4 100
    run 0.6b batched  16 50
    run 0.6b graphs    4 100
    run 0.6b graphs   16 50
fi

# 30B MoE (128 experts) — the production model
if [ "$FILTER" = "all" ] || [ "$FILTER" = "30b-moe" ]; then
    echo ""
    echo "=== 30B MoE (128 experts) ==="
    run 30b-moe immediate 1 100
    run 30b-moe batched   1 100
    run 30b-moe batched   4 50
    run 30b-moe batched  16 30
fi

# 30B REAM (96 experts) — EP candidate
if [ "$FILTER" = "all" ] || [ "$FILTER" = "30b-ream" ]; then
    echo ""
    echo "=== 30B REAM (96 experts) ==="
    run 30b-ream immediate 1 100
    run 30b-ream batched   1 100
    run 30b-ream batched   4 50
    run 30b-ream batched  16 30
fi

echo ""
echo "=========================================="
echo "  SUITE COMPLETE: $(date)"
echo "  Results in: $SCRIPT_DIR/results/"
echo "=========================================="

# Summary
echo ""
echo "=== SUMMARY ==="
for f in "$SCRIPT_DIR"/results/*.json; do
    python3 -c "
import json, os
d=json.load(open('$f'))
name=os.path.basename('$f')
print(f\"  {d['model']:10s} {d['config']:10s} np={d['np']:2d} | {d['aggregate_tps']:6.1f} agg t/s | {d['per_user_tps']:5.1f} per-user | clean={d['clean_slots']}/{d['np']} | wall={d['wall_time_ms']}ms\")
" 2>/dev/null
done
