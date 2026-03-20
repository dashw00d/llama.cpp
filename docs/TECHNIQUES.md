# Technique Catalog — 3x Intel Arc A770 LLM Inference Optimization
**Generated:** 2026-03-19  
**System:** Qwen3-30B-A3B MoE, 3x Intel Arc A770, i9-7900X / X299 (44 PCIe 3.0 lanes: x16/x16/x8)  
**Branch:** `/home/ryan/llm-stack/llama.cpp-eptp/` (ep-tp-combined)  
**Sources:** MEMORY.md, memory/2026-03-18.md, memory/2026-03-19.md, optimization-inventory.md, caching-architecture.md, instant-model-swap.md, state-file-caching.md, outputs/*

---

## Performance Baseline Reference

| Config | t/s | ms/tok | Notes |
|--------|-----|--------|-------|
| 30B MoE immediate (original) | 0.6 | — | Host-stall bound (naive row split) |
| 30B MoE TP naive | 1.93 | — | Per-copy malloc AllReduce |
| 30B MoE TP + staging pool | 2.24 | — | Reused pinned buffers |
| 30B MoE TP + direct AllReduce | 2.51 | — | Host-staged sum |
| 30B MoE immediate (all 2026-03-18 fixes) | 2.40 | 416ms | Baseline day-start |
| 30B MoE batched no-graph (morning 3-19) | 2.99 | 335ms | After dpct/wait fixes |
| 30B MoE batched (proper benchmark) | 3.44 | 291ms | --no-warmup, 43% above baseline |
| 30B MoE batched (verified avg 3 runs) | 3.0 | 330ms | 323/332/335ms — consistent |
| 30B MoE batched (2 AM burst) | 18–19 | 51-55ms | **NOTE: likely misconfigured / measurement error per later investigation** |
| 0.6B batched no-graph | 17–18 | 54-58ms | Stable |
| 0.6B batched + graphs | 19.14 | 52ms | +10% over batched alone |
| 0.6B batched + graphs (final session) | 28.6 | 35ms | **Best measured** |
| 0.6B batched no-graph (final session) | 18.5 | 54ms | Final consistent |
| 32B dense batched no-graph | 2.38 | 420ms | Measurement-validated |
| 32B dense immediate no-graph | 2.15 | 465ms | Slower than batched |
| 96-expert REAM EP single-user | 3.95 | 250ms | **+65% over original 128-expert TP baseline** |
| 96-expert REAM TP single-user | 3.38 | 296ms | Same model, TP mode |
| EP np=2 (BROKEN — output corrupted) | — | — | Q8 cache poisoning bug |

---

## Category 1: Tensor Parallelism (TP)

### T-1. Tensor Parallelism via Meta Backend
- **Category:** Parallelism / Architecture
- **What it does:** Splits weight matrices across 3 GPUs along rows/columns. Each GPU computes a partial result. AllReduce combines shards. Enables running models larger than single-GPU VRAM.
- **Split axes:** `SPLIT_AXIS_0` (row split, e.g. output projections), `SPLIT_AXIS_1` (column split, e.g. up/gate projections), `SPLIT_AXIS_PARTIAL` (triggers AllReduce boundary)
- **Source files:** `ggml/src/ggml-backend-meta.cpp` (~1700 lines), `src/llama-model.cpp` (split assignment), `ggml/src/ggml-sycl/ggml-sycl.cpp`
- **Flag:** `--split-mode tensor`
- **Result:** Produces 97 subgraphs per token for 30B MoE (48 attention + 48 MoE FFN + 1 final)
- **Status:** ✅ WORKING / PRODUCTION
- **Why it works:** 3 GPUs in parallel with PCIe aggregation; enables 48GB effective VRAM for 19GB model

### T-2. AllReduce — Host-Staged with Pinned Staging Buffers
- **Category:** Communication / Parallelism
- **What it does:** Cross-device reduction via host USM memory: GPU→host (parallel), host sum (AVX-512), host→GPU (parallel). Reuses a single persistent staging pool allocation. BF16 format.
- **Source file:** `ggml/src/ggml-sycl/ggml-sycl.cpp:5628` (`ggml_backend_sycl_allreduce_tensor()`)
- **Called from:** `ggml/src/ggml-backend-meta.cpp:1158`
- **Evolution:**
  - v1 naive malloc: 1.93 t/s  
  - v2 staging pool: 2.24 t/s  
  - v3 host-staged sum: 2.51 t/s  
- **Bottleneck:** 96 AllReduce calls per token = ~48ms (14% of token time). Two synchronous `queue.wait()` per call.
- **PCIe limit:** 600KB tensor at 13GB/s PCIe 3.0 = ~90µs. Dominated by DMA latency, not bandwidth.
- **Why P2P not used:** Arc A770 has no GPU-to-GPU P2P DMA. Host-staged is the only option.
- **Env vars:** None specific — uses SYCL USM malloc_host
- **Status:** ✅ WORKING / NEAR-OPTIMAL for current hardware
- **Known waste:** AllReduce wait #2 (host→GPU step) is redundant per SYCL in-order queue semantics — provably safe to remove per analysis, but requires meta backend pipelining changes

### T-3. Delayed MoE AllReduce (get_i_delayed)
- **Category:** Communication / Subgraph Optimization
- **What it does:** Defers the AllReduce boundary past the MoE aggregation chain (ADD_ID, MUL, VIEW, expert-sum ADDs). Merges expert output weight application + aggregation into the same subgraph as the expert matmul, avoiding an extra subgraph break.
- **Source:** `ggml/src/ggml-backend-meta.cpp:917-983` (`get_i_delayed()`)
- **Commit:** `ae0334ffa`
- **Result:** Reduces MoE subgraph boundaries (from theoretically higher to current 97)
- **Status:** ✅ IMPLEMENTED / PRODUCTION

### T-4. AllReduce: Remove Redundant Wait #2
- **Category:** Communication / Latency Reduction
- **What it does:** The Step-3 wait (after host→GPU memcpy) in AllReduce is provably unnecessary — SYCL in-order queues guarantee ordering without explicit waits. Downstream compute on each GPU will implicitly order after its own queue's memcpy.
- **Estimated savings:** ~48ms/token (14% of token time) if all 96 AllReduces benefit
- **Blocker:** Requires meta backend pipelining to truly exploit — meta backend currently calls allreduce synchronously before next subgraph
- **Status:** 🔶 IDENTIFIED / NOT IMPLEMENTED (requires meta backend restructure)

### T-5. AllReduce: Fused 3-Way Host Sum
- **Category:** Performance Micro-Optimization
- **What it does:** Change `for i in 1..n_backends: acc[j] += src[i][j]` to `acc[j] = acc[j] + src1[j] + src2[j]` (single-pass, saves one memory pass over accumulator)
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp:5681-5687`
- **Impact:** Only measurable for tensors >1MB. Most AllReduce tensors are ~8KB (hidden_size), so negligible.
- **Status:** 🔶 IDENTIFIED / LOW PRIORITY

### T-6. AllReduce: Stack-Allocated Arrays in Meta Backend Caller
- **Category:** Memory / Micro-Optimization
- **What it does:** Replace `std::vector<ggml_backend_t>` / `std::vector<ggml_tensor*>` heap allocs on every AllReduce call with stack/static arrays (size GGML_SYCL_MAX_DEVICES)
- **Source:** `ggml/src/ggml-backend-meta.cpp:1162-1170`
- **Impact:** Saves 96 heap alloc/free cycles per token (trivial — 24-48 bytes each)
- **Status:** 🔶 IDENTIFIED / LOW PRIORITY

---

## Category 2: Expert Parallelism (EP)

### EP-1. Expert Parallelism — SPLIT_AXIS_2 (Expert Dimension Split)
- **Category:** Parallelism / Architecture
- **What it does:** Splits MoE expert weights across GPUs on the expert dimension instead of the weight dimension. GPU0 owns experts 0–31, GPU1 owns 32–63, GPU2 owns 64–95. Each GPU computes only its local experts. AllReduce SUM (zero-masked) combines results. Attention layers remain TP.
- **Prerequisite model:** 96-expert REAM model (96 ÷ 3 = 32 per GPU — clean division). 128-expert models fail (128 % 3 ≠ 0).
- **Model:** `/home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf` (96 experts, 8 active, 13.18 GiB)
- **Source files:**
  - `src/llama-model.cpp` — SPLIT_AXIS_2 assignment for `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (Phase 1)
  - `ggml/src/ggml-backend-meta.cpp` — EP dispatch + expert_offset in op_params (Phase 2)
  - `ggml/src/ggml-sycl/ggml-sycl.cpp` — EP-aware MUL_MAT_ID with local expert masking + offset (Phase 3)
- **op_params encoding:** `[1] = expert_offset`, `[2] = n_local_experts`
- **Phase 1:** Tensor split change (~50 LOC) ✅
- **Phase 2:** Meta backend EP dispatch (~50 LOC) ✅
- **Phase 3:** SYCL EP-aware MUL_MAT_ID (~100-150 LOC) ✅
- **Subgraph count fix:** 193 → 97 (meta backend boundary fix, +16% speedup) ✅
- **Results:**
  - EP single-user: **250ms / 3.95 t/s** (avg 3 runs: 251, 253, 255ms)
  - TP same model: 296ms / 3.38 t/s
  - EP improvement over TP: **+17%**
  - vs original 128-expert TP baseline (416ms): **+65% total**
- **np=2 concurrency:** BROKEN — Q8 cache poisoning bug (see EP-2 for fix)
- **Status:** ✅ IMPLEMENTED (single-user), ⚠️ np=2+ NEEDS TESTING after concurrency fix

### EP-2. EP Concurrency Fix — Q8 Cache Poisoning
- **Category:** Bug Fix / Correctness
- **What it does:** In `ggml_sycl_mul_mat_id()` batch path, the same `src1_contiguous` pointer was reused across experts, causing the Q8 quantization cache to return stale quantized data for subsequent experts in the same batch.
- **Fix:** Call `invalidate_q8_cache()` before each expert matmul (2 lines)
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_mul_mat_id` batch path)
- **Symptom:** Garbled/corrupted output with np=2, both slots; 500 errors from server
- **Status:** ✅ FIX LANDED — needs np=2 re-test to confirm

### EP-3. EP Subgraph Count Fix (193 → 97)
- **Category:** Bug Fix / Performance
- **What it does:** Meta backend was creating 193 subgraphs instead of 97 for EP mode due to incorrect boundary detection. Fix restored correct 97-subgraph behavior.
- **Source:** `ggml/src/ggml-backend-meta.cpp`
- **Impact:** +16% speedup (extra subgraphs = extra AllReduce overhead)
- **Status:** ✅ FIXED

### EP-4. EP np=16 Throughput (Theoretical)
- **Category:** Scalability / Future Work
- **What it does:** 16 parallel slots with EP. Expert routing with 16 tokens: 16 × 8 = 128 activations per layer → ~43 per GPU → GPU compute finally utilizes a significant fraction of Arc A770's EUs
- **Estimated:** ~60 t/s total throughput (vs ~42 t/s TP np=16 estimate)
- **Key insight:** np=16 + EP is the "real prize" — GPU utilization goes from ~1.5% to ~30-50%
- **Status:** 🔶 NOT YET TESTED — np=2 concurrency bug needs to be resolved first

---

## Category 3: SYCL Dispatch Optimization

### D-1. Batched Command Lists
- **Category:** Dispatch / SYCL Tuning
- **What it does:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` batches kernel submissions into L0 command lists before hardware submission. Reduces per-kernel overhead from ~61µs to ~48µs (21% reduction in launch overhead).
- **Env var:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`
- **Result:** 
  - 30B MoE: 416ms → 344ms (-72ms, +21%) immediately
  - 0.6B: 77ms → 56ms immediate mode → batched  
  - **30% faster** on inference once loaded
- **Side effect (FIXED):** Made model loading ~10× slower until `.wait()` serialization was fixed (see D-2)
- **Source:** No code changes — env var only
- **Status:** ✅ PRODUCTION — default in arcllm-proxy

### D-2. Per-Tensor `.wait()` Removal (Loading Path)
- **Category:** Loading / Dispatch Serialization Fix
- **What it does:** Removed `queues_wait_and_throw()` (full device drain) and `.wait()` after each tensor upload during model loading. Was serializing every tensor memcpy: 300+ batch-flush-wait cycles → ~5 flushes of 64 tensors each. Root cause of "batched mode makes loading slow."
- **Source file:** `ggml/src/ggml-sycl/ggml-sycl.cpp`
- **Lines fixed:** 
  - Line 449: `queues_wait_and_throw()` before each tensor — REMOVED (was full device drain)
  - Line 455: `.wait()` after each H2D memcpy — REMOVED (was blocking per-tensor)
  - Added: `pending_host_bufs` vector in buffer context, `flush_pending_host_bufs()` called every 64 tensors
  - Also fixed: Line 994 (split buffer set_tensor), Lines 177-178 (concat.cpp pipelining)
- **Impact:** Made batched cmdlists viable for loading. ~60× fewer batch-flush-wait cycles.
- **Result:** 0.6B batched load: ~10s (was ~250s+ without fix). 30B: viable vs was hanging.
- **Status:** ✅ IMPLEMENTED

### D-3. Per-Op `ggml_sycl_set_device()` Removal
- **Category:** Hot-Path Cleanup / Mutex Reduction
- **What it does:** Removed 13+ redundant `ggml_sycl_set_device(ctx.device)` calls from per-op functions. In TP mode, each backend is pinned to one device — these were no-op mutex acquisitions.
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (13 calls removed from hot path)
- **Estimated impact:** ~315 recursive_mutex acquisitions/token → ~15 across 3 GPUs
- **Status:** ✅ IMPLEMENTED
- **Codex note:** fp16 enforcement should be made explicit after removal

### D-4. Cached Device Capability Queries
- **Category:** Hot-Path Cleanup
- **What it does:** Replaced repeated `has_capability_or_fail(dev, {aspect::fp16})` SYCL runtime queries with `thread_local` cached bool. Device capabilities don't change at runtime.
- **Sites removed:** ~40 queries per GPU per token across mmq.cpp (20), convert.cpp (25), dmmv.cpp (7), others
- **Source files:** `ggml/src/ggml-sycl/mmq.cpp`, `dmmv.cpp`, `convert.cpp`, `rope.cpp`, `cpy.cpp`, `binbcast.cpp`
- **Status:** ✅ IMPLEMENTED (partial — capability caching added at graph_compute level)

### D-5. Peer-Access Loop Removal
- **Category:** Dead Code Removal
- **What it does:** Removed dead `ggml_sycl_set_peer_access()` loop that called `ggml_sycl_set_device(i)` for all devices even though peer access is not implemented on Intel GPUs (body was `#ifdef NDEBUG` with commented-out code).
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp:2593`
- **Status:** ✅ IMPLEMENTED

### D-6. Dead Kernel Counter Removal
- **Category:** Hot-Path Cleanup
- **What it does:** Removed per-graph kernel dispatch counters that were never used for anything.
- **Status:** ✅ IMPLEMENTED

### D-7. Lock-Free Instrumentation Fast Path
- **Category:** Hot-Path Cleanup
- **What it does:** Added lock-free fast path for `instrument_graph_compute` — after initial warmup, the mutex was acquired 3× per token for telemetry. Added check: skip locking entirely if instrumentation not needed.
- **Status:** ✅ IMPLEMENTED

### D-8. AllReduce Timing Gated
- **Category:** Hot-Path Cleanup
- **What it does:** `chrono::now()` calls in AllReduce path gated to first 5 calls + every 200th. Avoids syscall overhead on every AllReduce.
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp:5642-5648`
- **Status:** ✅ IMPLEMENTED

### D-9. AllReduce `__restrict__` Qualifiers
- **Category:** CPU Vectorization
- **What it does:** Added `__restrict__` to host AllReduce sum loop arrays, enabling auto-vectorization (AVX-512 on i9-7900X). `float * __restrict__ acc`, `const float * __restrict__ src`
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp:5681`
- **Impact:** Negligible in practice — host sum is not the bottleneck (<0.1ms). But correct to do.
- **Status:** ✅ IMPLEMENTED

---

## Category 4: SYCL Graph Recording & Replay

### G-1. SYCL Graph Recording + Replay (Dense Models)
- **Category:** Graph Optimization / Kernel Launch Elimination
- **What it does:** Records SYCL compute graphs per subgraph on first execution. Replays the pre-recorded graph on subsequent tokens, eliminating per-kernel dispatch overhead (~48µs/launch → ~1-2µs/replay).
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (`replay_or_record_segment()`)
- **Env var:** `GGML_SYCL_DISABLE_GRAPH=0` (enable), `GGML_SYCL_DISABLE_GRAPH=1` (disable)
- **Result on 0.6B dense:** ✅ **+33% speedup** (54ms → 35ms). 28.6 t/s vs 18.5 t/s. 99% replay rate, 57 subgraphs.
- **Result on 30B MoE:** ❌ **NEUTRAL / NEGATIVE** (-3% overhead). Only 1.9% replays; 72% permanent unstable fallbacks; 25.7% skip (too small). Net: +9ms overhead vs no-graph.
- **Result on 32B dense Q3_K_M:** ⚠️ **98% replay but 0% speedup** — memory bandwidth bound, not launch-overhead bound. 32B reads 20GB/token; launch overhead (~8ms) is <2% of 420ms token time.
- **Result on 32B dense Q4_K_M:** ❌ **CRASH** — VRAM exhaustion (381 graphs + 19GB weights > 48GB)
- **Key benefit beyond speed:** Eliminates JIT warmup — with graphs, first request runs at full speed. Without graphs, batched mode takes 2-3 requests to stabilize.
- **Status:** ✅ WORKING for models ≤ ~50 layers (0.6B). ❌ NOT VIABLE for 30B+ MoE. ❌ USELESS for large dense models (bandwidth-bound).

### G-2. Graph Cache Key — Topology + Shape Split
- **Category:** Graph Bug Fix
- **What it does:** Split cache key into two components: (1) topology key (stable — based on op types and structure) for cache lookup, (2) shape hash (validation only — checks tensor dimensions). Prevents KV cache growth from poisoning every cache entry.
- **Bug fixed:** KV cache `ne[]` changes every token on MoE → invalidated every key → no graph hits
- **Status:** ✅ FIXED (2026-03-18)

### G-3. Graph Key — Pointer Inclusion Fix (Unstable → Per-Layer Keys)
- **Category:** Graph Bug Fix / Root Cause
- **What it does:** Topology key collision: all 48 layers shared the same cache entry (same op structure). Fix: include tensor data pointers in topology key → per-layer cache entries.
- **Problem introduced:** Per-layer keys caused MoE segments to attempt recording, but `MUL_MAT_ID` is graph-incompatible (expert routing shifts pointers) → server hangs on token 4.
- **Root cause insight:** The collision was accidentally preventing crashes by making MoE segments skip graphing.
- **Correct fix needed:** Per-layer keying ONLY for segments that don't contain `MUL_MAT_ID`. MoE-containing segments must continue to skip graphing.
- **Status:** ⚠️ PARTIALLY BROKEN — needs MoE-segment exclusion logic

### G-4. Graph Pointer Stability Tracking
- **Category:** Graph Safety
- **What it does:** Before replaying a graph segment, validates that all tensor data pointers match recorded values. After 2 pointer mismatches, marks segment "unstable" — falls back to direct dispatch permanently for that segment.
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp:5000-5050`
- **Why needed:** MoE expert routing creates VIEW tensors pointing into weight arrays at expert-specific offsets. Different expert selection each token → different pointer → stale graph replay = DEVICE_LOST.
- **Status:** ✅ IMPLEMENTED (correctly prevents crashes)

### G-5. MKL GEMM → Graph-Compatible SYCL Kernels
- **Category:** Graph Bug Fix
- **What it does:** oneMKL `gemm()` calls `event.wait()` internally, which is incompatible with SYCL graph recording (graphs cannot contain blocking waits). Replaced with graph-compatible SYCL kernels.
- **Status:** ✅ FIXED (2026-03-18)

### G-6. Graph Assert Fix (Updatable Path)
- **Category:** Bug Fix
- **What it does:** Non-updatable pointer-mismatch path fell through to updatable graph code → assert crash. Fixed conditional to correctly separate updatable vs non-updatable replay paths.
- **Status:** ✅ FIXED (2026-03-18)

### G-7. Arc A770 Graph Cache Hardware Limit (L0 Executable Graph Limit)
- **Category:** Hardware Limitation
- **What it does:** Discovery: Arc A770 / L0 driver limits executable graphs per device to ~127. 0.6B model needs ~56/device (fits ✅). 32B dense needs ~192/device (exceeds → 100% thrashing ❌).
- **Evidence:** 32B with Q2_K: 104,190 evictions, 0 replays, ~400ms/token (WORSE than 2.38ms baseline)
- **Fix path:** Graph cache eviction (cap at ~100/device), or only graph first N layers
- **Conclusion:** Graph caching viable ONLY for models with ≤ ~50 layers on 3x Arc A770 TP setup
- **Status:** 🔴 HARDWARE LIMIT — no fix possible short of eviction strategy

### G-8. Graph Recording Budget (Too Conservative)
- **Category:** Graph Configuration
- **What it does:** Graph recording budget of 3/call (+1 replenishment) throttles compilation. For 32B dense: 774 graph finalizations × ~73ms each = ~56s first-token latency. The "hang" was actually the throttled compilation across hundreds of eval passes.
- **Fix:** Bump budget or remove it entirely; accept ~56s one-time first-token latency; JIT cache makes it one-time.
- **Status:** ⚠️ NEEDS FIX for 32B dense, irrelevant for MoE

### G-9. L0 Mutable Command Lists (Future / Uncertain)
- **Category:** Advanced Graph / L0 Interop
- **What it does:** L0 spec v1.9+ `ze_experimental_mutable_command_list` allows updating kernel arguments in a closed command list without full re-record. Would enable MoE graph segments to update expert pointers cheaply (~5-20µs vs ~300µs re-record).
- **DG2 support:** UNCERTAIN. `ext_oneapi_limited_graph` aspect strongly suggests mutable CL not supported on DG2. Needs runtime query test.
- **Status:** 🔬 RESEARCH ONLY — not implemented, support unconfirmed

### G-10. Buffer Pool Allocator for MoE Pointer Stability
- **Category:** Graph Enablement
- **What it does:** Pre-allocate fixed pool of device memory for active expert tensors. Expert routing maps experts to pool slots (stable addresses). Different experts reuse same addresses → SYCL graph replays without pointer mismatches. Eliminates root cause of graph instability for MoE.
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED — most architecturally correct long-term fix

---

## Category 5: MoE Expert Dispatch Optimization

### M-1. MoE Dispatch Phase 1 — Persistent Pinned Host Buffer
- **Category:** MoE / Memory
- **What it does:** Replaced `std::vector<char>` heap allocation per `MUL_MAT_ID` call with persistent pinned host buffer (`sycl::malloc_host`) that grows as needed and persists across calls.
- **Eliminates:** 96 heap alloc/free cycles per token (one per MoE layer)
- **Source:** `ggml/src/ggml-sycl/common.hpp` (added `ids_pinned_buf`, `ids_pinned_buf_size`, `get_ids_pinned()`)
- **Status:** ✅ IMPLEMENTED

### M-2. MoE Dispatch Phase 1 — Event-Specific Wait
- **Category:** MoE / Dispatch
- **What it does:** Replaced `stream->wait()` (drains entire queue) with `event.wait()` (waits only for the specific D2H routing-ID memcpy). Allows previously submitted GPU work to continue executing during the wait.
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_mul_mat_id`)
- **Eliminates:** 96 full-queue drains per token → 96 targeted event waits
- **Status:** ✅ IMPLEMENTED

### M-3. MoE Dispatch Phase 1 — Single-Pass Expert Counting
- **Category:** MoE / CPU Hot Path
- **What it does:** Replaced O(n_as × n_tokens × n_ids) triple-nested loop that scanned ALL 128 experts to count active tokens with O(n_tokens × n_ids) single-pass that builds `expert_token_counts[]` then iterates only active experts.
- **For decode (1 token, 8 active):** 8 iterations vs 1,024 iterations saved
- **For prefill (512 tokens, 8 active):** 4,096 vs 524,288 iterations saved
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_mul_mat_id` batch path)
- **Status:** ✅ IMPLEMENTED

### M-4. Fused MoE MMVQ Kernel (Phase 2 — Batch Expert Dispatch)
- **Category:** MoE / Kernel Fusion
- **What it does:** Single fused SYCL kernel replaces N individual per-expert kernel launches. Grid layout: `(n_experts, 1, block_num_y)`. Each work-group handles one row of one expert's mat-vec multiply. The expert dimension is in `group(0)`, so all experts execute in one launch but with independent work-groups. Shared Q8 quantization of src1 across all experts.
- **New files:** `ggml/src/ggml-sycl/fused-moe-mmvq.hpp`, `ggml/src/ggml-sycl/fused-moe-mmvq.cpp`
- **Modified:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (added `ggml_sycl_mul_mat_id_fused()`, used as first-try path in `ggml_sycl_mul_mat_id()`)
- **Supported types:** Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (standard + reorder variants)
- **Savings per MoE op (decode, n_expert_used=4):** 3 launches × 48µs = 144µs + quantize reuse + reduced D2H
- **Estimated savings:** ~27ms/token vs fast-path (8% reduction from 344ms baseline)
- **Status:** ✅ BUILDS — impact **not yet benchmarked on 30B**. Fused expert aggregation NOT activating per latest benchmark (needs investigation).

### M-5. MoE Phase 2: Device-Side Expert Routing (Future)
- **Category:** MoE / Kernel Architecture
- **What it does:** Eliminate D2H copy of routing IDs entirely. Pre-compute expert pointer table (128 × sizeof(void*)) at model load time in device memory. Kernel reads routing IDs directly from `ids` tensor on device; indexes into pre-computed pointer table on-device.
- **Impact:** Eliminates last D2H sync in MUL_MAT_ID → makes MoE graph-compatible
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### M-6. MoE Phase 3: Fully Fused Expert Kernel (Future)
- **Category:** MoE / Kernel Architecture
- **What it does:** Single SYCL kernel: reads routing IDs on device, gathers expert weights, computes all 8 expert matmuls, scatters/aggregates results. One kernel per layer vs current 8+.
- **Estimated savings:** -170ms/token (eliminates 4,464+ kernel submissions for MoE) → ~5.7 t/s from 2.9
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED — highest individual ROI optimization

### M-7. Expert Aggregation Reduction (Future Kernel Fusion)
- **Category:** MoE / Kernel Fusion
- **What it does:** Replace 7 sequential `add_f32` kernels (aggregating 8 expert outputs) + 1 `mul_f32` (weight scaling) with a single fused `reduce_experts` kernel: `dst[i] = Σ(weights[k] × expert_outputs[k][i])`
- **Savings:** 8 kernels × 48 layers × 3 GPUs = 1,152 submissions × 48µs = **~55ms/token**
- **Implementation:** Simple parallel reduction over 8 vectors of size 2048. Low blast radius.
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED — **highest ROI kernel fusion**

---

## Category 6: Kernel Fusion (General)

### K-1. Fused MMQ (GGML_SYCL_FUSED_MMQ)
- **Category:** Matmul / Memory Optimization
- **What it does:** Dequantize inside the matmul vector math instead of separate dequant kernel + matmul. Saves a full memory round-trip (no intermediate dequant buffer).
- **Source files:** `ggml/src/ggml-sycl/mmvq.cpp`, `mmq.cpp`, `vecdotq.hpp`, `dequantize.hpp`
- **Env var:** `GGML_SYCL_FUSED_MMQ=1`
- **Status:** ✅ PRODUCTION — enabled in arcllm-proxy

### K-2. Flash Attention
- **Category:** Attention / Memory
- **What it does:** Tiled memory-efficient attention. Reduces memory I/O for attention computation.
- **Source:** `ggml/src/ggml-sycl/fattn-common.hpp` and related
- **Flag:** `-fa on`
- **Measured cost:** ~10ms/token (2.9% of 344ms) — not a bottleneck
- **Status:** ✅ PRODUCTION

### K-3. Shared Q/K/V src1 Quantization (Future Fusion)
- **Category:** Attention / Kernel Fusion
- **What it does:** Q, K, V projections all quantize the same input (`attn_norm` output) to Q8_1 format independently. Cache/reuse the quantized result.
- **Savings:** 2 × 48 layers × 3 GPUs = 288 submissions × 48µs = **~14ms/token**
- **Implementation:** Hash src1 pointer + check quantized-tensor pool cache
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### K-4. RMSNorm + Quantize_Q8_1 Fusion (Future)
- **Category:** Kernel Fusion
- **What it does:** Fuse `rms_norm_f32(inp, cur)` + `quantize_q8_1(cur, q8)` into single kernel. Eliminates intermediate f32 buffer and one kernel launch per layer.
- **Savings:** 1 × 96 subgraphs × 3 GPUs = 288 submissions = **~14ms/token**
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### K-5. Residual ADD + RMSNorm Fusion (Future)
- **Category:** Kernel Fusion
- **What it does:** `add_and_rms_norm(a, b, normed, sum)` — writes both residual sum (for skip connection) and normalized output. Replaces 2 kernels at every layer boundary.
- **Savings:** 1 × 96 subgraphs × 3 GPUs = 288 submissions = **~14ms/token**
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### K-6. RMSNorm(Q) + RoPE(Q) and RMSNorm(K) + RoPE(K) Fusion (Future)
- **Category:** Attention / Kernel Fusion
- **What it does:** Fuse `rms_norm_f32(Q_raw, Q_normed)` + `rope_ext(Q_normed, Q_roped)` into single kernel. Saves 2 launches per attention subgraph.
- **Savings:** 2 × 48 layers × 3 GPUs = 288 submissions = **~14ms/token**
- **Complexity:** Medium-High — RoPE has per-head rotation + frequency parameters
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### K-7. Fused Top-K Select (SOFTMAX + ARGSORT + GET_ROWS → 1) (Future)
- **Category:** MoE / Kernel Fusion
- **What it does:** Single `fused_top_k_select(logits, expert_ids, weights, k=8)` replaces 3 sequential kernels for MoE router.
- **Savings:** 2 × 48 MoE layers × 3 GPUs = 288 submissions = **~14ms/token**
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

### K-8. Gate + Up MUL_MAT_ID Shared Quantization (Future)
- **Category:** MoE / Kernel Fusion
- **What it does:** Gate and Up expert projections both use the same `ffn_norm` output. Quantize once, share Q8 result across both.
- **Savings:** 1 × 48 MoE layers × 3 GPUs = 144 submissions = **~7ms/token**
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

---

## Category 7: Load-Time Optimization

### L-1. L0 NEO Compiler Cache (JIT Cache)
- **Category:** Loading / JIT
- **What it does:** Compiled SYCL/L0 kernels cached to NVMe. Skips JIT recompilation on subsequent server starts.
- **Live path:** `/tmp/neo_compiler_cache/` (where L0 runtime looks)
- **Persistent paths:**
  - `/home/ryan/llm-stack/cache/neo_compiler_cache_immediate/`
  - `/home/ryan/llm-stack/cache/neo_compiler_cache_batched/`
- **Key insight:** Batched mode compiles DIFFERENT kernels than immediate mode — needs separate caches or cache is useless on mode switch.
- **Management:** `cache-manager.sh restore|save|status`
- **Impact:** 30B warm load: ~30s vs cold ~10-40 min (varies by mode/model)
- **Status:** ✅ IMPLEMENTED — arcllm-proxy saves/restores with mode-specific paths
- **Remaining issue:** Unhealthy startup may still save bad cache; stale `/tmp/neo_compiler_cache` can leak across modes

### L-2. KV Slot Save/Restore (Layer 2 Cache — Conversation State)
- **Category:** Serving / State Caching
- **What it does:** Saves/restores full KV cache state to NVMe. Enables instant conversation context restoration without re-processing prompts.
- **API:** `POST /slots/0?action=restore {"filename": "model.bin"}` / `POST /slots/0?action=save`
- **Path:** `/home/ryan/llm-stack/cache/slots/<model-name>.bin`
- **Size:** ~4.5GB for Qwen3-32B
- **Status:** ✅ IMPLEMENTED in arcllm-proxy
- **Known asymmetry:** Slot caching behavior is asymmetric across slots (Codex review flag)

### L-3. Bounce Buffer Removal for Arc GPUs (Load Speed)
- **Category:** Loading / Performance
- **What it does:** The Arc-unnecessary host bounce buffer in `ggml_backend_sycl_buffer_set_tensor()` copies mmap data through malloc'd host buffer before H2D transfer. On DG2/Arc, mmap addresses should be directly usable as DMA sources (no bounce needed — that's a PVC issue).
- **Impact:** Eliminates ~1,737 malloc+memcpy pairs during 30B load (~5.9 GiB redundant CPU→CPU copies). Expected 2-5× faster upload.
- **Source:** `ggml/src/ggml-sycl/ggml-sycl.cpp` (set_tensor path)
- **Codex risk flag:** Async DMA from mmap source may outlive unflushed source lifetime — needs hardening
- **Status:** ⚠️ PARTIALLY IMPLEMENTED — code landed but correctness risks flagged, needs hardening

### L-4. Parallel GPU Dispatch in Meta set_tensor (Load Speed)
- **Category:** Loading / Parallelism
- **What it does:** `meta_buffer_set_tensor` currently loops GPUs 0→1→2 sequentially. Could parallelize with OpenMP or `std::async` so all 3 GPUs upload in parallel.
- **Source:** `ggml/src/ggml-backend-meta.cpp:499` (meta buffer set_tensor)
- **Impact:** 1.5-2× upload speed improvement
- **Codex risk flag:** Parallel `set_tensor` thread safety not proven
- **Status:** ⚠️ PARTIALLY IMPLEMENTED — code landed but thread-safety needs verification

### L-5. Pre-Split Weight Snapshots (Future — Instant TP Load)
- **Category:** Loading / Architecture
- **What it does:** After first TP model load, dump each GPU's tensor shard to a contiguous binary blob on NVMe. Subsequent loads: check for pre-split files → mmap each shard → single bulk DMA per GPU. No GGUF parsing, no per-tensor splitting, no 1,737 individual memcpys.
- **Paths:** `/home/ryan/llm-stack/cache/tp-shards/<model-hash>/gpu{0,1,2}.bin` + `meta.json`
- **Estimated impact:** Load time ~10 min → ~2 seconds (3 bulk DMA transfers at PCIe 3.0 speeds: 2GB/GPU ÷ 13GB/s ≈ 0.15s/GPU)
- **Status:** 🔶 DESIGNED IN DETAIL / NOT IMPLEMENTED — highest priority cache task

### L-6. State File Caching (System Prompt Pre-computation)
- **Category:** Serving / State Caching
- **What it does:** Use `llama_state_save_file()` / `llama_state_load_file()` to serialize full context state (KV cache for attention + SSM state for Mamba layers) after system prompt processing. Load state file instead of re-processing on subsequent requests.
- **Hash key:** Hash of system prompt content
- **Path:** `/home/ryan/llm-stack/cache/state-files/<model-name>/<prompt-hash>.bin`
- **Size:** ~60-200MB (just context state, not weights)
- **Load speed:** ~60ms memcpy from NVMe vs 54s re-processing
- **Impact:** Makes Nemotron-120B and Qwen3.5 with large system prompts viable
- **Difference from L-2:** L-2 (KV slot cache) saves mid-conversation state. This pre-computes system prompt ONCE and reuses forever.
- **Status:** 🔶 DESIGNED / NOT IMPLEMENTED

---

## Category 8: Serving Infrastructure

### S-1. arcllm-proxy (Lazy Loading Reverse Proxy)
- **Category:** Serving
- **What it does:** Ollama-like reverse proxy wrapping llama-server. Manages model lifecycle (load on demand, swap on request), orchestrates all cache layers, handles DEVICE_LOST with GPU reset.
- **Paths:** Proxy on port 11435, llama-server on port 18400 (internal)
- **Source:** `/home/ryan/llm-stack/scripts/arcllm-proxy.py`
- **SYCL env defaults:**
  - `GGML_SYCL_DISABLE_GRAPH=1` (graphs currently disabled for 30B)
  - `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched mode)
  - `ZE_AFFINITY_MASK=0,1,2` (all 3 GPUs)
  - `GGML_SYCL_FUSED_MMQ=1`
- **GPU reset on DEVICE_LOST:** sysfs `echo 1 > /sys/class/drm/card*/device/reset` + clear shader caches
- **Status:** ✅ PRODUCTION — needs hardening (proxy cache lifecycle bugs flagged by Codex)

### S-2. Prompt Cache Reuse
- **Category:** Serving
- **What it does:** llama-server's built-in prompt prefix caching. Reuses KV cache segments across similar requests.
- **Flag:** `--cache-reuse 256`
- **Status:** ✅ PRODUCTION — enabled by default

---

## Category 9: Speculative Decoding

### SD-1. External Draft Model (BLOCKED)
- **Category:** Throughput / Speculative
- **What it does:** Small draft model (Qwen3-0.6B) generates N tokens ahead; large model (30B) verifies in parallel. Correct predictions accepted for free.
- **Flag:** `-md <draft_model_path>`
- **VRAM:** Comfortable — 30B MoE ~6.3GB/GPU + 0.6B ~0.2GB/GPU = ~6.5GB/16GB
- **Projected speedup:** 7-11 t/s at 60-80% acceptance rate (2.3-3.6× effective speedup)
- **Tested result:** ❌ **0% acceptance rate** — vocab mismatch
- **Root cause:** Qwen3-0.6B and Qwen3-30B-A3B-abliterated have incompatible tokenizers despite both reporting BPE/151936 size. `"target and draft vocabs are not compatible"` warning. Every draft token rejected after translation.
- **Overhead:** 577-754ms/token with spec decoding vs 330ms baseline — **SLOWER**
- **Fix needed:** Draft model with EXACT same tokenizer as target (same family, same fine-tune)
- **Status:** ❌ BLOCKED by tokenizer mismatch

### SD-2. N-Gram Self-Speculative Decoding (Untested)
- **Category:** Throughput / Speculative
- **What it does:** Self-speculative n-gram lookup with no draft model needed. Flags: `--lookup-cache-static` / `--lookup-cache-dynamic`. Multiple modes: `ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, `ngram_cache`.
- **Expected:** Lower acceptance rate than draft model, but zero overhead/VRAM
- **Status:** 🔶 NOT TESTED — still a valid alternative approach

---

## Category 10: Loading Path Analysis — What Was Investigated

### LA-1. Batched Mode Loading Stall — Root Cause Analysis
- **Category:** Investigation / Bug
- **What was thought:** Batched command lists were "dead" because 30B loading appeared to stall
- **Root cause found:** Per-tensor `.wait()` (Line 455, 994 in ggml-sycl.cpp) was forcing L0 to do batch-submit-flush-wait-reset per tensor. ~300+ batch-flush-wait cycles instead of ~5.
- **Lesson:** Never declare an optimization dead without root-causing. The actual inference speed was never tested independently from loading.
- **Source:** `outputs/wait-audit.txt`, `outputs/batched-load-analysis.txt`

### LA-2. L0 Command List Pointer Baking
- **Category:** Investigation / Hardware
- **What was found:** Regular L0 command lists have identical pointer-baking behavior to SYCL graphs. `zeCommandListAppendLaunchKernel` snapshots kernel arguments into the command buffer at record time. Going to L0 directly gives zero advantage for the MoE pointer instability problem.
- **Source:** `outputs/l0-cmdlist-analysis.txt`

---

## Category 11: Architecture Investigations (Failed / Not Viable)

### A-1. Mega-Graph (All 97 Subgraphs in One Graph) — NOT FEASIBLE
- **Category:** Graph / Architecture
- **Why not feasible:** AllReduce requires host staging (GPU→host, sum, host→GPU). SYCL graphs cannot record operations that cross graph execution boundary to host memory in this way. The AllReduce boundaries are hard synchronization points that require host CPU involvement.
- **Source:** `outputs/mega-graph-feasibility.txt`
- **Status:** ❌ NOT FEASIBLE

### A-2. Meta Backend Pipeline Batching — NOT FEASIBLE
- **Category:** Architecture
- **Why not feasible:** Every subgraph boundary IS an AllReduce (TP dependency). Subgraph i+1 computation always depends on the AllReduce result from subgraph i. No independent work exists to overlap. (Exception: MoE delayed AllReduce already handles the one feasible case.)
- **Source:** `outputs/meta-pipeline-feasibility.txt`
- **Status:** ❌ NOT FEASIBLE

### A-3. Subgraph Reduction 97 → 50-64 via Meta Backend Merge — NOT ACHIEVABLE
- **Category:** Architecture
- **Why not achievable:** Both AllReduces per layer are mathematically required (RMSNorm is non-linear; can't propagate PARTIAL through it). The only path to ~49 subgraphs is Expert Parallelism (eliminates 48 MoE AllReduces). Naive subgraph merging in meta backend alone cannot reduce count.
- **Source:** `outputs/subgraph-reduction-analysis.txt`
- **Status:** ❌ NOT ACHIEVABLE without EP

### A-4. AllReduce Pipeline Overlap — NOT APPLICABLE
- **Category:** Communication
- **Why not applicable:** Downstream computation always depends on AllReduce result (TP critical path). No work can overlap with the AllReduce within the TP computation sequence.
- **Source:** `outputs/allreduce-analysis.txt`
- **Status:** ❌ NOT APPLICABLE

---

## Token Time Budget Summary (30B MoE, 344ms/token)

| Component | Time | % |
|-----------|------|---|
| Kernel launch overhead (MoE: 169ms + Attn: 29ms) | 198ms | 57.5% |
| AllReduce overhead (96 calls × 6 waits each) | 48ms | 14.0% |
| MUL_MAT_ID host sync (D2H blocking, 432 waits) | 22ms | 6.4% |
| MoE expert pool alloc/quantize | 22ms | 6.4% |
| Flash Attention execution | 10ms | 2.9% |
| Meta backend dispatch | 8ms | 2.3% |
| Output projection + framework | 6ms | 1.7% |
| Element-wise / RoPE / residual | 6ms | 1.7% |
| GPU matmul compute (ALL expert math) | **<2ms** | **0.6%** |
| Unattributed | 22ms | 6.4% |

**Key finding: GPU compute is 0.6% of token time. 99.4% is overhead.**

GPU utilization during decode: **0%** (confirmed via intel_gpu_top and bench_gpu_monitor.py)

---

## Combined Optimization Estimates (Stacked)

| Config | Estimated t/s | Key changes |
|--------|--------------|-------------|
| Current (30B MoE TP batched no-graph) | 3.0 t/s | Baseline |
| + EP (96-expert model) | 3.95 t/s | -48 MoE AllReduces, local experts |
| + Fused MoE kernel (Phase 3) | ~5.7 t/s | 24 kernels/layer → 1 |
| + Expert aggregation fusion | ~6.5 t/s | 8 ADD → 1 fused reduce |
| + Attention graph replay | ~7.0 t/s | Eliminate attention launch overhead |
| + All kernel fusions (8 total) | ~8-10 t/s | ~132ms total kernel savings |
| + Speculative decoding (matching draft) | ~15-20 t/s | 2-3× effective multiplier |
| **Target** | **15+ t/s** | |
| **Physics ceiling** | **222 t/s** | Memory bandwidth bound (1.46GB active reads) |

---

## Environment Variables Reference

| Variable | Values | Effect |
|----------|--------|--------|
| `GGML_SYCL_DISABLE_GRAPH` | 0/1 | Enable/disable SYCL graph recording |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | 0/1 | 0=batched (faster inference), 1=immediate |
| `ZE_AFFINITY_MASK` | `0,1,2` | Use all 3 Arc A770 GPUs |
| `GGML_SYCL_FUSED_MMQ` | 1 | Fused dequant+matmul |
| `GGML_SYCL_DEBUG` | 1 | Enable graph hit/miss debug logging |

---

## Source File Quick Reference

| File | What it contains |
|------|-----------------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Main SYCL backend: graph_compute, mul_mat, AllReduce, MUL_MAT_ID, fused dispatch |
| `ggml/src/ggml-sycl/common.hpp` | Context structs, graph cache entries, pinned buffer pool |
| `ggml/src/ggml-sycl/mmvq.cpp` | Fused matrix multiply quantized (MMVQ) |
| `ggml/src/ggml-sycl/mmq.cpp` | Matrix multiply quantized |
| `ggml/src/ggml-sycl/fused-moe-mmvq.hpp/.cpp` | Fused multi-expert MMVQ kernel (new) |
| `ggml/src/ggml-backend-meta.cpp` | Meta backend: TP subgraph dispatch, AllReduce, EP dispatch |
| `src/llama-model.cpp` | Tensor split axis assignment per tensor name |
| `/home/ryan/llm-stack/scripts/arcllm-proxy.py` | Serving proxy: model lifecycle, cache orchestration |
| `/home/ryan/llm-stack/scripts/cache-manager.sh` | L0 compiler cache save/restore/status |
| `/home/ryan/llm-stack/scripts/bench_gpu_monitor.py` | GPU utilization/temp/RAM monitor |

---

## Status Legend
- ✅ IMPLEMENTED / WORKING
- ⚠️ IMPLEMENTED / NEEDS HARDENING OR HAS KNOWN ISSUES
- 🔶 DESIGNED / NOT IMPLEMENTED
- 🔬 RESEARCH ONLY / UNCERTAIN
- ❌ FAILED / NOT VIABLE / BLOCKED
