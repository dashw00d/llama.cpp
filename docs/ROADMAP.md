# Roadmap — Intel Arc A770 × 3 LLM Inference

## The Goal

Run large MoE language models on 3x Intel Arc A770 GPUs at **competitive inference speeds**. The target is **15+ tokens/second** on Qwen3-30B-A3B (a 30B parameter MoE model with 8 active experts per token).

We started at **0.6 t/s**. We're at **3.0–3.95 t/s**. There's a clear engineering path to 15+ t/s, and the physics ceiling is **222 t/s** — we're leaving 98% of the hardware on the table.

## Why This Matters

Nobody runs LLM inference on Intel Arc GPUs at scale. NVIDIA dominates this space. If we can make 3x $350 consumer GPUs match or beat a $1,500 NVIDIA GPU for LLM serving, that changes the economics of self-hosted AI.

## The Hardware

```
3x Intel Arc A770 (16GB VRAM each, 48GB total)
Intel i9-7900X / X299 motherboard
44 PCIe 3.0 lanes (x16/x16/x8)
```

Each GPU has 560 GB/s memory bandwidth. For MoE inference, only ~485MB of weights are "active" per token (8 of 96 experts). That means each token should theoretically take <1ms of GPU memory reads. We're at 330ms. The gap is pure software overhead.

## Where We Are (March 19, 2026)

### What Works
| Config | Speed | Status |
|--------|-------|--------|
| 30B REAM (96 exp) — TP, batched | **3.0 t/s** | ✅ Production ready |
| 0.6B dense — TP, batched | **18.5 t/s** | ✅ Production ready |
| 0.6B dense — TP, batched + graphs | **26 t/s** | ✅ Production ready |
| Batched cmdlists vs immediate | **+20%** | ✅ Proven |
| Model loading (JIT cached) | **15s** (was 180s) | ✅ Production ready |

### What's Broken
| Config | Issue |
|--------|-------|
| Expert Parallelism (EP) | Output corruption at np=1. Code audited twice — reads correct but runtime output is garbled. TP on same model = clean. Printf debug is next step. |

### What's Untested
| Config | Potential |
|--------|-----------|
| N-gram self-speculation | 5-15% free speedup, zero VRAM cost |
| Single GPU (13GB REAM) | No AllReduce = potentially faster than 3-GPU TP |
| 3x independent instances | 3× single-GPU throughput |
| 2.7B MoE smoke test | Fast MoE iteration (loads in 2s vs 15s) |

## The Token Time Budget (Why We're Slow)

For a single token on the 30B MoE at 3.0 t/s (330ms):

```
What the GPU spends its time on:

  57.5%  Kernel launch overhead       198ms   ← THIS IS THE PROBLEM
  14.0%  AllReduce (cross-GPU sync)    48ms   ← EP eliminates half of this
   6.4%  MoE host sync (D2H copy)     22ms
   6.4%  MoE quantize/pool overhead   22ms
   2.9%  Flash Attention execution     10ms
   2.3%  Meta backend dispatch          8ms
   0.6%  GPU math (actual computation) <2ms   ← THE GPU IS IDLE 99.4% OF THE TIME
   9.8%  Other overhead               34ms

  TOTAL                               330ms
```

The GPU does <2ms of real work per token. Everything else is software overhead: launching kernels, synchronizing across GPUs, copying data between host and device.

## The Plan

### Phase 1: Fix EP Corruption (CURRENT PRIORITY)

Expert Parallelism gives each GPU ownership of 32 experts (instead of all 96 with TP). This means:
- No MoE AllReduce (-48ms per token)
- Each GPU only computes 2-3 active experts (vs 8 with TP)
- At np=16, GPU utilization jumps from 1.5% to 30-50%

**EP is implemented and runs at 3.95 t/s** — but output is corrupted. Two code audits say the logic is correct, yet runtime output is garbled while TP on the same model is clean.

**Next step:** Add printf debug logging (test `90-ep-debug-printf.sh`) to trace what each GPU actually sees at runtime: `ne02`, `expert_offset`, which experts pass the filter, and the local index used.

**Validation strategy:** Use the 2.7B MoE model for fast iteration. It loads in 2 seconds (vs 15s for 30B), exercises the same MUL_MAT_ID code path, and has 64 experts. Note: 64 % 3 ≠ 0, so EP won't work on it directly — but it validates the TP MoE path after any code changes. For EP testing, use the 96-expert REAM model.

**The gate test:** `bench/tests/30-ep-vs-tp-ream.sh` — runs both TP and EP on the same model, same prompt, checks output quality. When this test shows ✅, EP is fixed.

### Phase 2: EP Multi-User Scaling

Once EP np=1 produces clean output:

1. **Run test `31-ep-np2-concurrency.sh`** — validates the Q8 cache fix works
2. **Run test `32-ep-np16-throughput.sh`** — the real payoff
3. Compare against TP scaling: tests `04-tp-scaling-ream-np4.sh` and `05-tp-scaling-ream-np16.sh`

Expected: EP np=16 aggregate throughput 2-4x higher than TP np=16 because MoE AllReduce is eliminated and expert compute is distributed.

### Phase 3: Kernel Fusion (Medium-Term)

The token time budget shows 57.5% is kernel launch overhead (198ms). Each MoE layer dispatches 31+ individual kernels. Fusing them reduces launches:

| Fusion | Saves | Impact |
|--------|-------|--------|
| Fused MoE kernel (all experts in 1 launch) | -170ms | **Highest ROI** |
| Expert aggregation (8 ADDs → 1 reduce) | -55ms | Easy win |
| Shared Q8 quantization (Q/K/V reuse) | -14ms | |
| RMSNorm + Quantize fusion | -14ms | |
| Top-K select fusion | -14ms | |

The fused MoE kernel already exists in the codebase (`fused-moe-mmvq.hpp`) but is blocked from activating in TP mode by a buffer guard. In EP mode, it would fire — another reason to fix EP first.

**Stacked estimate:**
```
Current:                    3.0 t/s
+ EP:                       3.95 t/s  (+32%)
+ Fused MoE kernel:        ~5.7 t/s  (+90%)
+ Expert aggregation:      ~6.5 t/s  (+117%)
+ All kernel fusions:      ~8-10 t/s (+230%)
+ Speculative decoding:    ~15-20 t/s (+500%)
```

### Phase 4: Speculative Decoding (Longer-Term)

Draft-model speculation failed because Qwen3-0.6B and Qwen3-30B have incompatible tokenizers (0% acceptance rate). Two paths forward:

1. **N-gram self-speculation** (test `21-ngram-speculative.sh`) — no draft model, 5-15% gain, ready to test TODAY
2. **Same-family draft model** — find a small Qwen3 model with identical tokenizer. This could give 2-3x effective speedup.

### Phase 5: Architecture Explorations

These are promising but secondary to EP and kernel fusion:

| Approach | Test | Notes |
|----------|------|-------|
| Single GPU (no AllReduce) | `40-single-gpu-ream.sh` | REAM fits in 16GB. Eliminates ALL cross-GPU overhead. |
| 3x instances (one per GPU) | `41-single-gpu-3x-instances.sh` | 3× single-GPU throughput. No shared state. |
| Pre-split weight snapshots | Not built | Load time: 10 min → 2 seconds (mmap + bulk DMA) |
| Device-side expert routing | Not built | Eliminates D2H sync in MoE → enables graph recording |

## How to Work on This

### Setup
```bash
# Source environment
source /home/ryan/llm-stack/env.sglang-xpu.sh

# Two worktrees:
#   llama.cpp-stable  — all working optimizations, no EP (branch: stable-baseline)
#   llama.cpp-eptp    — EP implementation, experimental (branch: ep-tp-combined)
```

### Making Changes
```bash
# Edit code in the appropriate worktree
cd /home/ryan/llm-stack/llama.cpp-stable   # for TP/general work
cd /home/ryan/llm-stack/llama.cpp-eptp     # for EP work

# Build
cd build-sycl && cmake --build . --target llama-server -j$(nproc)

# ALWAYS before testing:
pkill -x llama-server; sleep 2; pgrep -x llama-server && echo "LEAK!" || echo "clear"
```

### Running Tests
```bash
cd /home/ryan/llm-stack/llama.cpp-stable

# Quick sanity (2 seconds to load)
bash bench/tests/02-tp-baseline-06b.sh

# MoE sanity (2 seconds to load)  
bash bench/tests/20-moe-smoke-2.7b.sh

# Production baseline
bash bench/tests/01-tp-baseline-ream.sh

# EP gate test (is EP fixed?)
bash bench/tests/30-ep-vs-tp-ream.sh
```

### Key Files to Know

| File | What | When to touch |
|------|------|---------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | SYCL backend: graphs, matmul, AllReduce, MUL_MAT_ID | Kernel optimization, EP dispatch |
| `ggml/src/ggml-backend-meta.cpp` | Meta backend: TP split, subgraphs, AllReduce orchestration | Split logic, EP dispatch |
| `src/llama-model.cpp` | Tensor split axis assignment | EP tensor split |
| `ggml/src/ggml-sycl/fused-moe-mmvq.hpp` | Fused MoE kernel | Kernel fusion work |

### Anti-Patterns (Learn from Our Mistakes)

1. **Don't declare something dead without root-causing.** Batched mode "made loading slow" → turned out to be a `.wait()` bug, not batched mode itself.
2. **Don't trust garbled = "model is bad."** Always test the same model on stable TP first.
3. **Always kill the server before starting a new one.** Zombie processes contaminate results.
4. **Don't fly blind.** Run GPU monitor alongside benchmarks: `python3 /home/ryan/llm-stack/scripts/bench_gpu_monitor.py --log /tmp/gpu.jsonl`
5. **Save results as JSON.** The bench suite does this automatically. Don't rely on terminal scrollback.
6. **Use the 0.6B or 2.7B model for fast iteration.** Don't wait 15 seconds for the 30B to load when you're debugging MoE dispatch logic.

## File Map

```
llama.cpp-stable/
├── docs/
│   ├── README.md          ← Start here
│   ├── ROADMAP.md         ← This file (goals + plan)
│   ├── TECHNIQUES.md      ← 60+ techniques, what worked/failed
│   ├── HISTORY.md         ← Git archaeology, 6 phases
│   ├── BENCHMARKS.md      ← All performance numbers
│   ├── ENV-VARS.md        ← 8 env toggles, code locations
│   ├── OPEN-QUESTIONS.md  ← Solved mysteries + remaining leads
│   └── EP-DEBUG.md        ← EP corruption debug guide
│
├── bench/
│   ├── env.sh             ← Common env (source this)
│   ├── run-bench.sh       ← Generic bench runner
│   ├── run-suite.sh       ← Full test matrix
│   ├── results/           ← JSON results (timestamped)
│   └── tests/
│       ├── README.md      ← Test catalog (proven/failing/planned)
│       ├── 01-05          ← Baseline + scaling tests
│       ├── 10-11          ← A/B dispatch tests
│       ├── 20-21          ← MoE + speculation tests
│       ├── 30-32          ← EP tests (currently failing)
│       ├── 40-41          ← Single-GPU explorations
│       └── 90             ← Debug tools
```
