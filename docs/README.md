# Intel Arc A770 × 3 — LLM Inference Optimization Docs

## What Is This?
This is the complete documentation and benchmark suite for running LLM inference on 3x Intel Arc A770 GPUs using a custom SYCL backend for llama.cpp.

## Hardware
- 3x Intel Arc A770 (16GB VRAM each, 48GB total)
- Intel i9-7900X / X299 motherboard
- 44 PCIe 3.0 lanes (x16/x16/x8 slot configuration)

## Quick Start
```bash
# Source the environment
source /home/ryan/llm-stack/env.sglang-xpu.sh

# Run a single benchmark
cd /home/ryan/llm-stack/llama.cpp-stable
bash bench/run-bench.sh 30b-ream batched 1 100

# Run the full test suite
bash bench/run-suite.sh all
```

## Directory Layout
```
llama.cpp-stable/
├── bench/                    # Benchmark suite
│   ├── README.md             # How to run benchmarks
│   ├── run-bench.sh          # Single test: model × config × np
│   ├── run-suite.sh          # Full matrix of all combinations
│   └── results/              # JSON results (timestamped)
│
├── docs/                     # You are here
│   ├── README.md             # This file — start here
│   ├── ARCHITECTURE.md       # How the multi-GPU pipeline works
│   ├── TECHNIQUES.md         # Every optimization: what, why, status
│   ├── HISTORY.md            # Timeline of what we tried and learned
│   ├── BENCHMARKS.md         # All performance numbers, progression
│   └── ENV-VARS.md           # Every env var toggle, what it does
│
├── build-sycl/               # Build directory
│   └── bin/llama-server      # The server binary
│
└── ggml/src/ggml-sycl/       # SYCL backend source (where the magic is)
```

## Models
| Model | Experts | Active | Size (Q4_K_M) | Path |
|-------|---------|--------|---------------|------|
| Qwen3-0.6B | dense | — | ~600MB | `models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` |
| Qwen3-30B-A3B | 128 MoE | 8 | ~18GB | `models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf` |
| Qwen3-30B-A3B-REAM | 96 MoE | 8 | ~13GB | `models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf` |

All model paths are relative to `/home/ryan/llm-stack/`.

## Current Best Performance
| Model | Config | np=1 t/s | Notes |
|-------|--------|----------|-------|
| 0.6B dense | batched+graphs | ~28.6 | Graph replay works great on small dense |
| 30B REAM (96 exp) | batched | ~3.0 | TP across 3 GPUs |
| 30B MoE (128 exp) | batched | ~3.0 | TP across 3 GPUs |

## Key Concepts
- **TP (Tensor Parallelism)**: Split weight matrices across GPUs, AllReduce to sync
- **EP (Expert Parallelism)**: Each GPU owns a subset of MoE experts — experimental, on `ep-tp-combined` branch
- **Batched command lists**: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` — 30% faster dispatch
- **SYCL graphs**: Record+replay kernel sequences — great for small dense, not viable for MoE
- **AllReduce**: GPU-to-GPU sync after each TP subgraph boundary

## Branches
| Branch | What | Status |
|--------|------|--------|
| `stable-baseline` | All working optimizations, no EP | ✅ PRODUCTION |
| `ep-tp-combined` | Expert Parallelism experiment | ⚠️ EXPERIMENTAL (output corruption) |
| `tensor-parallelism-upstream` | Original TP base | 📦 ARCHIVE |
