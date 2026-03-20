# Open Questions & Unexplored Leads

These are gaps, anomalies, and untested optimizations identified during the March 18-19 2026 sprint. Each needs investigation before we can call the optimization story complete.

---

## 🔴 Critical — Blocks EP Production Use

### OQ-1: EP Output Corruption at np=1
**Priority:** HIGHEST
**What:** Expert Parallelism produces garbled output even with a single slot (np=1). The Q8 cache invalidation fix (commit `5a590cbf7`) was targeted at np>1 concurrency, but np=1 garbling was confirmed on 2026-03-19 ~20:00.

**Known facts:**
- EP np=1 timing is correct: 250ms/token, 97 subgraphs, 3.95 t/s
- Output is consistently garbled (non-ASCII, random tokens)
- TP on the same REAM model produces clean output at 296ms / 3.38 t/s
- EP code touches 3 files: `llama-model.cpp`, `ggml-backend-meta.cpp`, `ggml-sycl.cpp`

**Likely suspects:**
1. Expert index remapping: `expert_id - expert_offset` to get local index — off-by-one or wrong offset?
2. Output pre-zeroing: `memset(dst, 0, ...)` may be zeroing at wrong time or wrong size
3. op_params encoding: Phase 2 writes `[1]=expert_offset, [2]=n_local_experts`, Phase 3 reads same — but are they reading the RIGHT tensor's op_params? (simple tensor vs original?)
4. AllReduce SUM: zeros from non-owned experts + real values from owned experts — is the SUM actually correct? Could be double-counting or missing contributions.

**Test plan:**
1. Add debug logging to Phase 3 SYCL code: print expert_offset, n_local for each GPU
2. For a single token, log which experts each GPU processes and what local index it uses
3. Compare against TP: same token should activate same 8 experts globally
4. Check the AllReduce output: is the summed result correct?

**Branch:** `ep-tp-combined` (commit `5a590cbf7`)

---

## 🟡 High Value — Unexplained Performance Anomalies

### OQ-2: The 0.6B 19→28.6 t/s Jump (50% Unattributed Speedup)
**Priority:** HIGH — if we understand this, we might replicate it on larger models
**What:** Between two test sessions on 2026-03-19, the 0.6B dense model went from 19.14 t/s (batched+graphs, A/B test) to 28.6 t/s (batched+graphs, "session wrap" measurement). That's a **50% improvement** with no known code change between tests.

**Timeline:**
- ~02:15 CDT: A/B test → 19.14 t/s avg (3 requests, steady state)
- ~08:30 CDT: "session wrap" → 28.6 t/s
- No commits between these timestamps

**Possible explanations:**
1. **JIT cache warming:** The 02:15 test may have been first-ever graph compilation. By 08:30, the L0 NEO compiler cache was fully populated → graph replay is faster?
2. **Different measurement method:** Was the 28.6 number from the server log `predicted_per_second` vs calculated differently?
3. **GPU thermal state:** 6 hours idle between tests → GPUs running cooler → higher boost clocks?
4. **Different context length:** Shorter prompt or fewer tokens generated = less KV overhead?
5. **Graph finalization caching:** Graphs finalized once are faster to replay on subsequent calls. First test did finalization; second test was pure replay.
6. **Server restart between tests:** Fresh server may have different memory layout, page table state, or L0 driver state

**How to investigate:**
```bash
# Test 1: cold start (kill server, clear JIT cache)
rm -rf /tmp/neo_compiler_cache
bash bench/run-bench.sh 0.6b graphs 1 100
# Note: first 3 requests are warmup in bench script

# Test 2: warm start (same server, second benchmark run)
bash bench/run-bench.sh 0.6b graphs 1 100
# Compare t/s between test 1 and test 2

# Test 3: check GPU temps
cat /sys/class/drm/card{1,2,3}/device/hwmon/hwmon{3,4,5}/temp1_input
# If temps are 40°C vs 70°C, thermal throttling could explain it
```

### OQ-3: The 18-19 t/s Garbled 30B MoE Result
**Priority:** HIGH — if this speed is achievable with correct output, it's our target
**What:** On 2026-03-19 ~02:00, 30B MoE batched mode hit 18.07 and 19.40 t/s. Output was garbled. We marked it as "measurement error" and moved on.

**But what if it wasn't an error?** 18 t/s on 30B MoE would mean something fundamental changed in how the kernels executed. The garbled output means a correctness bug, not necessarily a speed bug.

**Possible explanations:**
1. **Wrong model loaded:** Could the server have been running 0.6B but benchmarked with 30B prompt? (Unlikely — server logs should show model)
2. **Kernel compilation race:** Batched mode + graphs + fresh compilation = some kernels ran optimized while others ran fallback? The "correct" slow path wasn't taken?
3. **MoE expert dispatch was skipped:** If expert routing was broken and all tokens bypassed MoE layers, inference would be ~6x faster but produce garbage. This matches the numbers.
4. **AllReduce was no-oped:** If AllReduce returned without actually reducing, each GPU would compute independently (faster) but produce wrong results.

**How to investigate:**
```bash
# Reproduce with debug logging
GGML_SYCL_DEBUG=1 bash bench/run-bench.sh 30b-moe batched 1 10
# Check: are MoE expert dispatches actually happening?
# Check: are AllReduces actually running?
# Check: what does the META line show for subgraph count and ms/subgraph?
```

---

## 🟢 Untested Optimizations — Ready to Try

### OQ-4: Fused MoE Kernel — Built But Never Benchmarked on 30B
**Priority:** MEDIUM-HIGH — potentially -27ms/token (8% improvement)
**What:** `fused-moe-mmvq.hpp` is compiled into the stable branch. It replaces N individual expert kernel launches with 1 fused launch. But we never measured its impact on 30B.

**The code path:** In `ggml_sycl_mul_mat_id()`, the fused path is tried FIRST. If it returns false (unsupported quant type), falls back to per-expert dispatch.

**Questions:**
1. Is it actually activating for Q4_K_M? (Check with `GGML_SYCL_DEBUG=1`)
2. If not, why not? What quant types does it support?
3. If yes, is it faster than per-expert dispatch?

**Test:**
```bash
# With fused path
GGML_SYCL_DEBUG=1 bash bench/run-bench.sh 30b-moe batched 1 50
# Look for "fused" in debug output

# Force disable fused path (comment out fused call, rebuild, benchmark)
# Compare ms/subgraph and total ms/token
```

### OQ-5: N-Gram Self-Speculative Decoding
**Priority:** MEDIUM — zero cost to try, may give 1.3-2x speedup
**What:** llama.cpp supports `--lookup-cache-static` and `--lookup-cache-dynamic` for n-gram based self-speculation. No draft model needed. Lower acceptance rate than proper speculative decoding, but zero overhead.

**Test:**
```bash
# Dynamic n-gram lookup
./build-sycl/bin/llama-server \
  -m <30b-ream-model> \
  --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 \
  --lookup-cache-dynamic \
  --no-warmup > /tmp/ngram_test.log 2>&1 &
# Send a request with repetitive/predictable content
# Check: does predicted_per_second improve vs baseline?
```

### OQ-6: 32B Dense Graph Test with Proper Budget
**Priority:** MEDIUM — might recover 10-15% on dense models
**What:** The 32B dense graph test showed 98% replay rate but zero speedup. However, the test was run with a conservative graph budget that throttled compilation to 56 seconds of first-token latency. The "hang" was actually slow compilation, not a bug.

**The question:** After compilation finishes and steady-state replay kicks in, is there ANY speedup? The bandwidth-bound analysis says no, but we never measured it cleanly.

**Test:**
```bash
# Bump graph budget (or remove throttle), wait for compilation to finish
# Use Q3_K_M (fits in VRAM unlike Q4_K_M)
GGML_SYCL_DISABLE_GRAPH=0 bash bench/run-bench.sh 32b-dense graphs 1 200
# First ~50 tokens will be slow (compilation)
# Look at tokens 100-200 for steady-state speed
# Compare against batched no-graph baseline (5.88 t/s)
```

---

## 🔵 Designed But Not Built — Future Work

### OQ-7: Fused Expert Aggregation Kernel
**Estimated impact:** -55ms/token (16% improvement)
**What:** Replace 8 sequential `add_f32` + 1 `mul_f32` per MoE layer with single fused `reduce_experts` kernel.
**Status:** Designed, not built. Low blast radius — standalone kernel.

### OQ-8: Pre-Split Weight Snapshots
**Estimated impact:** Load time 10 min → 2 seconds
**What:** Dump per-GPU tensor shards to NVMe after first load. Subsequent loads: mmap + bulk DMA.
**Status:** Designed in detail (memory/caching-architecture.md). Not built.

### OQ-9: Device-Side Expert Routing (Eliminate D2H Sync)
**Estimated impact:** -22ms/token (6.4% improvement) + enables MoE graph recording
**What:** Pre-compute expert pointer table in device memory. Kernel reads routing IDs directly on device.
**Status:** Designed. Would enable MoE segments to be graph-recorded.

### OQ-10: Bounce Buffer Hardening
**What:** Direct mmap DMA is active but flagged as potentially unsafe (async DMA from mmap may outlive source). Needs correctness verification under stress.
**Status:** Code landed, correctness not verified.

---

## Investigation Checklist

When exploring these, use the bench suite:
```bash
cd /home/ryan/llm-stack/llama.cpp-stable
bash bench/run-bench.sh <model> <config> <np> <tokens>
# Results auto-saved to bench/results/ as JSON
```

Always run GPU monitor alongside:
```bash
python3 /home/ryan/llm-stack/scripts/bench_gpu_monitor.py --log /tmp/gpu.jsonl &
# After test:
python3 /home/ryan/llm-stack/scripts/bench_gpu_monitor.py --summarize /tmp/gpu.jsonl
```
