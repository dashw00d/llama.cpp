# Bench Tests — Organized by Status

Each test is self-contained with rich context in header comments.
Run with: `bash bench/tests/<test>.sh`

## Common env
All tests source `bench/env.sh` which sets up paths, models, and helpers.

---

## ✅ PROVEN — Known working, use as regression tests

| Test | What | Expected |
|------|------|----------|
| `01-tp-baseline-ream.sh` | TP on 96-expert REAM, batched, np=1 | ~330ms, ~3.0 t/s, CLEAN |
| `02-tp-baseline-06b.sh` | TP on 0.6B dense, batched, np=1 | ~54ms, ~18.5 t/s, CLEAN |
| `03-tp-graphs-06b.sh` | TP + graphs on 0.6B dense | ~35ms, ~26 t/s, CLEAN |
| `10-immediate-vs-batched.sh` | A/B: immediate vs batched dispatch | Batched ~20% faster |
| `20-moe-smoke-2.7b.sh` | MoE code path on tiny 2.7B model | Fast load, CLEAN output |

## ❌ FAILING — Known broken, needs debugging

| Test | What | Status |
|------|------|--------|
| `30-ep-vs-tp-ream.sh` | EP vs TP same model | EP garbled, TP clean |
| `31-ep-np2-concurrency.sh` | EP np=2 after Q8 fix | Blocked by np=1 corruption |
| `32-ep-np16-throughput.sh` | EP np=16 scaling | Blocked by np=1 corruption |
| `90-ep-debug-printf.sh` | EP debug tracing | Diagnostic tool for EP bugs |

## 🔬 PLANNED — Not yet tested, ready to run

| Test | What | Why |
|------|------|-----|
| `04-tp-scaling-ream-np4.sh` | TP np=4 concurrent | Baseline for EP scaling comparison |
| `05-tp-scaling-ream-np16.sh` | TP np=16 concurrent | Maximum TP scaling reference |
| `21-ngram-speculative.sh` | N-gram self-speculation | Untested free fruit (5-15% est.) |
| `40-single-gpu-ream.sh` | REAM on 1 GPU, offload KV to others | 13GB fits in 16GB, no AllReduce |
| `41-single-gpu-3x-instances.sh` | 3 independent servers, 1 per GPU | 3x throughput, no cross-GPU sync |

---

## Models Available

| Model | Size | Experts | Status |
|-------|------|---------|--------|
| Qwen3-0.6B Q8_0 | ~600MB | dense | ✅ Ready |
| Qwen1.5-MoE-A2.7B Q2_K | 5.89GB | 64 MoE | ✅ Ready (just downloaded) |
| Qwen3-30B-A3B-REAM Q4_K_M | 13.18GB | 96 MoE | ✅ Ready |
| Qwen3-30B-A3B-abliterated Q4_K_M | ~18GB | 128 MoE | ✅ Ready |

## Running Tests

```bash
# Single test
bash bench/tests/01-tp-baseline-ream.sh

# All proven tests
for t in bench/tests/0[1-3]*.sh bench/tests/10*.sh bench/tests/20*.sh; do bash "$t"; done

# All planned tests
for t in bench/tests/04*.sh bench/tests/05*.sh bench/tests/21*.sh bench/tests/4*.sh; do bash "$t"; done
```
