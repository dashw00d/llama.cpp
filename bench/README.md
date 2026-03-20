# Benchmark Suite — llama.cpp-stable

## Worktree
- Path: `/home/ryan/llm-stack/llama.cpp-stable/`
- Branch: `stable-baseline` (from `ac12736a2`)
- Contains: All TP/AllReduce/batched/graph optimizations from March 18-19
- Does NOT contain: EP implementation (experimental, output corruption)
- Binary: `build-sycl/bin/llama-server`

## Hardware
- 3x Intel Arc A770 (16GB VRAM each, 48GB total)
- i9-7900X / X299 (44 PCIe 3.0 lanes: x16/x16/x8)

## Models
- **0.6B** (dense): `/home/ryan/llm-stack/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`
- **30B MoE** (128 experts): `/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf`
- **30B REAM** (96 experts): `/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf`

## Test Configs
| Config | Env Vars | Description |
|--------|----------|-------------|
| immediate | `GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | Baseline immediate dispatch |
| batched | `GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` | Batched command lists (30% faster on small models) |
| graphs | `GGML_SYCL_DISABLE_GRAPH=0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` | SYCL graph recording + replay |

## Running Benchmarks
```bash
./bench/run-bench.sh <model> <config> [np] [max_tokens]
# Examples:
./bench/run-bench.sh 0.6b batched 1 100
./bench/run-bench.sh 30b-moe batched 4 50
./bench/run-bench.sh 30b-ream batched 16 30
```

## Results
All results saved to `bench/results/` with timestamped filenames:
```
bench/results/YYYY-MM-DD_HHMMSS_<model>_<config>_np<N>.json
```
