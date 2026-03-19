# Orchestrator Brief: SYCL Tensor Parallelism Optimization

## The Goal

Get Qwen3-30B-A3B MoE from **2.5 t/s to 15+ t/s** on 3x Intel Arc A770 using `--split-mode tensor` (PR #19378's meta backend).

## The Bottleneck

**86µs per SYCL kernel launch** × 4,365 launches per token = 375ms of pure launch overhead. Actual compute is ~3ms. We're spending 99% of token time on dispatch, not math.

## The Proven Data Point

SYCL graph replay on 0.6B model: **8ms** for a full forward pass (vs 5143ms for first compile+run). Graph replay eliminates kernel launch overhead. The crash on request 2 (stale tensor pointers) is the bug to fix.

## The 4 Optimizations (priority order)

### A. Fix SYCL Graph Recording for TP (PRIMARY — 10-50x speedup potential)

Record per-device command graphs within each meta backend subgraph. Replay on subsequent tokens instead of re-dispatching individual kernels.

**Current status:** Graph cache struct added to `common.hpp`. Instrumentation partially implemented. The multi-device check in `check_graph_compatibility` was removed (TP dispatches per-device). The crash is stale tensor data pointers — graph was recorded with token 1's addresses, token 2 has different addresses.

**Key challenge:** `MUL_MAT_ID` (MoE expert dispatch) does host memcpy inside graph recording, which breaks SYCL graphs. Solution: split graphable vs non-graphable nodes — graph the attention ops, dispatch MUL_MAT_ID directly.

**Files:**
- `ggml/src/ggml-sycl/ggml-sycl.cpp` — `graph_compute` (~line 4490), `graph_compute_impl` (~line 4410), `check_graph_compatibility` (~line 4448)
- `ggml/src/ggml-sycl/common.hpp` — `graph_cache_entry` struct (already added)

### B. Reduce Subgraph Count (30-50% fewer sync points)

Meta backend creates 97 subgraphs for 48 layers (~2 per layer). Each boundary = AllReduce + sync. Target: 50-64 subgraphs by eliminating redundant AllReduces.

**Files:**
- `ggml/src/ggml-backend-meta.cpp` — subgraph splitting logic
- `src/llama-model.cpp` — `llama_meta_device_get_split_state` / `get_tensor_config`

### C. Kernel Fusion (20-30% compute reduction)

Fuse ops like RMSNorm+matmul, SiLU+mul, residual+norm. Low priority — only matters after launch overhead is solved.

### D. Batched Command Lists (3-5x launch speedup, no code changes)

`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` groups kernel submissions. TP path uses simple `queue.memcpy + queue.wait` (no events), so batched mode should be safe. Just an env var test.

## Task Pipeline

```
#1 [in_progress] Instrument graph_compute — log what changes between tokens
#2 [pending]     Fix graph update for pointer changes (blocked by #1)
#3 [pending]     Validate 0.6B graph replay — 10+ requests no crash (blocked by #2)
#4 [pending]     Test 30B MoE with graph replay (blocked by #3)
#5 [pending]     Split graphable vs non-graphable nodes for MUL_MAT_ID (blocked by #4)
#6 [pending]     Test batched command lists (independent — can run anytime)
#7 [pending]     Analyze and reduce subgraph count (independent)
#8 [pending]     Combine best optimizations, measure final t/s (blocked by #5, #6, #7)
```

## Environment

```
Worktree: /home/ryan/llm-stack/llama.cpp-eptp/
Build:    cd build-sycl && source /home/ryan/llm-stack/env.sglang-xpu.sh && cmake --build . --target llama-server -j$(nproc)
0.6B:     /home/ryan/llm-stack/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
30B MoE:  /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf
Server:   GGML_SYCL_DISABLE_GRAPH=0 ./bin/llama-server -m MODEL --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404
Request:  curl -s --max-time 60 -X POST http://127.0.0.1:18404/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
Kill:     pkill -f 'llama-server.*18404'
Cache:    --slot-save-path /home/ryan/llm-stack/cache/slots/ (instant model reload)
```

## Agent Roles

**Fixer** — reads code, implements changes, builds. Does NOT start servers or send requests. See `scripts/graph-fix-fixer.sh`.

**Tester** — starts servers, sends requests, captures crash logs and timing. Does NOT edit C++ code. See `scripts/graph-fix-tester.sh`. Creates new bug tasks when tests fail.

**Coordination:** Both agents share a TaskList. Fixer marks implementation tasks done → unblocks test tasks → Tester claims and runs them → if fail, Tester creates bug task → Fixer picks it up. Loop until all pass.

## Rules

- ALWAYS read files before editing
- ALWAYS build after code changes
- Do NOT regress to `queue.wait()` hacks
- Use `GGML_LOG_INFO` not `fprintf` for debug output
- Test on 0.6B first (fast iteration), then 30B MoE
- The `--slot-save-path` flag gives instant model reloads — USE IT
- Capture ALL crash data: backtrace, [META] timing, [KERN] counts, eval time
- Reference `OPTIMIZATION_PLAN.md` for detailed analysis of each optimization

## Current Measurements

| Config | t/s | Notes |
|--------|-----|-------|
| Legacy row-split | 0.6 | Host stall bound |
| TP naive | 1.93 | Per-copy malloc |
| TP + staging pool | 2.24 | Reused pinned buffers |
| TP + direct AllReduce | 2.51 | Host-staged sum |
| TP + graph replay (0.6B, req 2) | ~125 est | 8ms before crash |
| Target | 15+ | Graph replay + batched + fewer subgraphs |
