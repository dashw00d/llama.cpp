#!/bin/bash
# ============================================================
# TEST 05: TP Scaling — REAM np=16
# ============================================================
#
# WHAT: 16 concurrent users on 96-expert REAM with TP.
#
# WHY:  Maximum concurrent load test. This is where EP's real
#       advantage should show — TP's AllReduce becomes the
#       bottleneck with 16 concurrent tokens being processed.
#
# EXPECTED (TP):
#   Per-user: ~1500-2000ms (~0.5-0.7 t/s per user)
#   Aggregate: ~8-11 t/s total
#   Each META dispatch processes 16 tokens → 16x more compute
#   but same AllReduce count → AllReduce becomes larger % of time
#
# COMPARISON TARGET (EP, when fixed):
#   Per-user: ~400-600ms (~1.5-2.5 t/s per user)
#   Aggregate: ~24-40 t/s total
#   Each GPU only processes 2-3 active experts × 16 tokens
#   GPU utilization goes from ~1.5% to ~30-50%
#
# TOKEN TIME BUDGET CONTEXT:
#   At np=1:  57.5% kernel launch, 14% AllReduce, 0.6% GPU math
#   At np=16: kernel launch amortized, GPU math grows 16x,
#             AllReduce stays ~constant → GPU actually does work
#
# ============================================================
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TEST_NAME="05-tp-scaling-ream-np16"
LOG="/tmp/bench_${TEST_NAME}.log"

echo "=== $TEST_NAME ==="
kill_server

export GGML_SYCL_DISABLE_GRAPH=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0

"$STABLE_SERVER" -m "$MODEL_30B_REAM" \
    --split-mode tensor -ngl 99 -np 16 -c 8192 \
    --port $BENCH_PORT --no-warmup > "$LOG" 2>&1 &

wait_ready "$LOG" 180
warmup

echo "--- 16 concurrent requests (30 tokens each) ---"
PROMPTS=("math" "physics" "chemistry" "biology" "history" "geography"
         "music" "art" "literature" "philosophy" "astronomy" "cooking"
         "sports" "technology" "medicine" "architecture")

for i in $(seq 0 15); do
    run_inference "Write about ${PROMPTS[$i]}" 30 > "/tmp/np16_r$((i+1)).json" &
done
wait

echo "--- Results ---"
python3 -c "
import json
total_tps = 0; clean = 0; garbled = 0
for i in range(1, 17):
    try:
        d = json.load(open(f'/tmp/np16_r{i}.json'))
        t = d.get('timings', {})
        ms = t.get('predicted_per_token_ms', 0)
        tps = t.get('predicted_per_second', 0)
        c = d.get('choices', [{}])[0].get('message', {}).get('content', '')
        r = sum(1 for x in c if ord(x)<128)/max(len(c),1)
        ok = 'CLEAN' if r>0.8 and len(c)>5 else 'GARBLED'
        if ok == 'CLEAN': clean += 1
        else: garbled += 1
        total_tps += tps
        print(f'  Slot {i:2d}: {ms:6.0f}ms | {tps:5.1f} t/s | {ok} | {c[:40]}')
    except Exception as e:
        print(f'  Slot {i:2d}: FAILED ({e})')
        garbled += 1
n = clean + garbled
print()
print(f'  Clean: {clean}/16  Garbled: {garbled}/16')
print(f'  Aggregate: {total_tps:.1f} t/s')
print(f'  Per-user:  {total_tps/max(n,1):.1f} t/s')
"

echo "--- META ---"
grep META "$LOG" | tail -3
kill_server
echo "=== DONE ==="
