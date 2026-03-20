# Bench Tests — Individual Hypothesis Tests

Each test is a self-contained script that:
1. Documents WHAT it's testing and WHY (in the header comments)
2. Sources the common env from `../env.sh`
3. Runs ONE specific test
4. Saves JSON results to `../results/`
5. Prints PASS/FAIL with the key number

## Running Tests

```bash
# Run a single test
bash bench/tests/01-tp-baseline-ream.sh

# Run all tests in order
for t in bench/tests/[0-9]*.sh; do bash "$t"; done
```

## Test Categories

### 0x — Baselines (what works today)
- `01-tp-baseline-ream.sh` — TP on 96-expert REAM, batched, np=1
- `02-tp-baseline-06b.sh` — TP on 0.6B dense, batched, np=1
- `03-tp-graphs-06b.sh` — TP + graphs on 0.6B (should be ~26 t/s)
- `04-tp-scaling-ream-np4.sh` — TP on REAM, np=4 concurrent
- `05-tp-scaling-ream-np16.sh` — TP on REAM, np=16 concurrent

### 1x — Dispatch Optimizations
- `10-immediate-vs-batched-ream.sh` — A/B: immediate vs batched cmdlists
- `11-graphs-vs-nograph-06b.sh` — A/B: graphs vs no-graphs on 0.6B

### 2x — MoE Specific
- `20-moe-smoke-2.7b.sh` — MoE code path on tiny 2.7B model
- `21-ngram-speculative.sh` — N-gram self-speculation (untested free fruit)

### 3x — EP (on eptp branch — currently broken)
- `30-ep-vs-tp-ream.sh` — EP vs TP on same model (tests corruption fix)
- `31-ep-np2-concurrency.sh` — EP np=2 after Q8 cache fix
- `32-ep-np16-throughput.sh` — EP np=16 aggregate throughput

### 9x — Debug / Investigation
- `90-ep-debug-printf.sh` — EP with debug logging (ne02, expert_offset per GPU)
