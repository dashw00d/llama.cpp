# Git Archaeology: llama.cpp Worktree Timeline
**Generated:** 2026-03-19  
**Repo:** `/home/ryan/llm-stack/llama.cpp-eptp` (primary), plus worktrees  
**Focus:** Every custom technique tried, in chronological order, with status

---

## Worktree Map

| Directory | Active Branch | Purpose |
|---|---|---|
| `llama.cpp-eptp/` | `ep-tp-combined` | **Primary dev worktree** — all custom work lives here |
| `llama.cpp-stable/` | `stable-baseline` | Clean baseline from `ac12736a2` (all optims, no EP) |
| `llama.cpp/` | `master` | Tracks upstream origin/master |
| `llama.cpp-tp/` | `tensor-parallelism-upstream` | TP upstream ref |
| `llama.cpp-expert/` | `expert-parallelism` → now same as `master` | Legacy EP branch |

All branches share the same underlying git repo (linked worktrees).

---

## Branch Genealogy

```
origin/master (upstream llama.cpp)
    └── tensor-parallelism-upstream (PR #19378 meta backend TP)
            └── stable-baseline (TP base + our optimizations from ac12736a2)
                    ├── master (same as stable-baseline HEAD)
                    ├── ep-tp-combined (HEAD = 5a590cbf7) ← ACTIVE
                    └── expert-parallelism (same as master)
```

---

## Technique Timeline (Oldest → Newest)

### Phase 0: Upstream TP Foundation
**Branch:** `tensor-parallelism-upstream`  
**Key commit:** `ae0334ffa` — "delay AllReduce for Moe for less I/O"  
**Status:** ACTIVE (merged into all downstream branches)

The upstream PR #19378 added a meta backend that shards tensors across GPUs.  
Commits in this phase (oldest-first):
- `245dac6c0` — `ggml: backend-agnostic tensor parallelism` (the core PR)
- `8ca0d423f` — support for GPT-OSS, Qwen3 MoE
- `743151afa` — support for 4/8 GPUs  
- `1b70f14cb` — unconditional peer access
- `e9f261a8c` — re-use buffers + ggml contexts
- `e8a9d8419` — fix output pattern
- `c732203a3` — NCCL support
- `60beaebe2` — GGML: HIP: add RCCL support
- `755c3ff4a` — Remove shfl and AllReduce from backend interface
- `620d7d461` — 2D tensor set/get support
- `74a6d922` — fix view_offs scaling
- `b04befbb1` — support for tensor dims % n_devs != 0
- `05fc6b3f0` — better granularity estimate
- `0e8eba810` — static memory allocations, fix usage count
- `fa381dadc` — fix tensor granularity
- `aae658413` — more even memory distribution
- `28e4790ed` — **use BF16 for AllReduce** (replaces F32 → less bandwidth)
- `3d9a86bc0` — Fix device mismatch during scatter of AllReduce (#11)
- `08400041d` — **Enable previous AllReduce implementation** — "better in both perf and stability" (#12)
- `ae0334ffa` — **delay AllReduce for MoE for less I/O** (reduces sync round-trips)
- `fbe54506b` — partial Qwen3 Next support
- `10f101ab2` — Fix Qwen3 30B crash (granularity bug for 768-dim tensors)
- `ca1c0facc` — Fix crashes due to KV cache serialization (#9)

---

### Phase 1: Baseline on Intel Arc / SYCL
**Branch:** `stable-baseline` (split from ep-tp-combined after `ac12736a2`)  
**Status:** ACTIVE — this is the "known good" reference point

Pure TP on Intel Arc, no custom graph hacks. Measured baselines:
- Legacy row-split: **0.6 t/s** (host stall bound)
- TP naive: **1.93 t/s** (per-copy malloc)
- TP + staging pool: **2.24 t/s** (reused pinned buffers)
- TP + direct AllReduce: **2.51 t/s** (host-staged sum)

---

### Phase 2: Mega-Optimization Commit (2026-03-18/19)
**Branch:** `ep-tp-combined`  
**Commit:** `ac12736a2` — "All optimizations from 2026-03-18/19 session"  
**Status:** ACTIVE — all optimizations below are permanently baked in  
**Author:** Ryan `<letmegoblind@gmail.com>`  
**Date:** Thu Mar 19 02:27:32 2026

**Files changed:** 19 files, +1,975 / -308 lines  
Key files: `ggml-sycl.cpp` (+1254 lines), `common.hpp` (+92), `ggml-backend-meta.cpp` (+66), `mmvq.cpp` (+61), `mmq.cpp` (+60), `sycl-gemm.hpp` (+210 new)

**Combined result:** 30B MoE: 2.4 → 2.9 t/s (+20%), load 180s → 25s. 0.6B TP: 13 → 18.5 t/s (+42%)

Sub-techniques bundled in this commit:

#### A. SYCL Graph Recording — 6 Bug Fixes
**Status:** ACTIVE  
**What:** Fixed graph cache to record and replay SYCL command graphs per-subgraph, per-device, keyed by tensor shape. Eliminates per-kernel-launch overhead (86µs per launch × 4,365 = 375ms / token → ~8ms on cache hit).  
**Bugs fixed:** Pointer invalidation, multi-device check removed, segmented dispatch, stability tracking  
**Files:** `ggml-sycl/ggml-sycl.cpp` (graph_compute, graph_compute_impl), `common.hpp` (graph_cache_entry struct)  
**Limitations still present:** MUL_MAT_ID (MoE expert dispatch) cannot be graphed — it does host memcpy inside graph recording. Workaround: split subgraphs at MUL_MAT_ID boundaries.

#### B. `.wait()` Removal / Bounce Buffer Bypass
**Status:** ACTIVE  
**What:** Audited all `queue.wait()` calls and `.dpct::async_dpct_memcpy()` calls. Removed host stalls that blocked pipeline. Bypassed intermediate "bounce buffers" to go direct device→device.  
**Files:** `ggml-sycl/ggml-sycl.cpp`  
**Why:** Per-tensor `queue.wait()` was the root cause of "batched mode is broken" — it wasn't batched mode, it was the sync calls.

#### C. MoE Dispatch Phase 1 — Pooled Pinned Buffer + Event Wait
**Status:** ACTIVE  
**What:** Instead of per-expert malloc for routing tables, pre-allocate a pooled pinned buffer. Use SYCL events instead of blocking wait for expert dispatch synchronization.  
**Files:** `ggml-sycl/ggml-sycl.cpp` (MUL_MAT_ID path)

#### D. MoE Decode Fast Path — Pre-quantize + Direct MMVQ Dispatch
**Status:** ACTIVE  
**What:** For decode (batch_size=1), pre-quantize src1 to Q8 once and reuse across all expert matmuls. Dispatch directly to MMVQ kernel instead of going through full ggml_compute_forward path.  
**Files:** `ggml-sycl/ggml-sycl.cpp`, `mmvq.cpp`, `mmvq.hpp`

#### E. Batched Load — Async Flush Bookkeeping + Device-Gated mmap DMA
**Status:** ACTIVE  
**What:** Model load now fires async DMA transfers to each GPU while mmap pages come in. Device-gated means CPU only stalls waiting for a device when that device's buffer is about to be used.  
**Files:** `ggml-sycl/ggml-sycl.cpp`  
**Result:** Load time 180s → 25s (7x faster)

#### F. Mode-Specific Compiler Cache + Proxy Auto-Wiring
**Status:** ACTIVE  
**What:** Separate SYCL/L0 compiler cache directories for different env modes (graph enabled vs disabled). arcllm-proxy updated to auto-wire the right cache path.  
**Files:** proxy script, cache path config

#### G. Concat Pipeline + Padding Zeroing + Dead Code Removal
**Status:** ACTIVE  
**What:** Optimized CONCAT op dispatch, zeroed padding bytes to avoid NaN propagation from uninitialized memory, removed dead code paths.  
**Files:** `ggml-sycl/concat.cpp`, `ggml-sycl/cpy.cpp`, others  

#### H. sycl-gemm.hpp (New File)
**Status:** ACTIVE (but may be unused depending on config)  
**What:** New SYCL GEMM implementation wrapper (+210 lines). Provides alternative matrix multiply path.  
**Files:** `ggml-sycl/sycl-gemm.hpp` (new)

---

### Phase 3: Expert Parallelism (EP) Implementation
**Branch:** `ep-tp-combined`  
**Commit:** `8ba0c7f9e` — "Expert Parallelism implementation for 96-expert REAM model"  
**Date:** Thu Mar 19 18:14:27 2026  
**Status:** ACTIVE (with bug fix in next commit)  
**Author:** Ryan  

**Result:** 250ms/token, **3.95 t/s** (+17% over TP, +65% over original baseline)  
**Files changed:** 15 files, +1,912 / -166 lines

New files created:
- `ggml-sycl/fused-add-rmsnorm.cpp/.hpp` — Fused AddRMSNorm kernel
- `ggml-sycl/fused-expert-agg.cpp/.hpp` — Fused expert aggregation kernel  
- `ggml-sycl/fused-moe-mmvq.cpp/.hpp` — Fused MoE matrix-multiply kernel
- `ggml-sycl/fused-topk-select.cpp/.hpp` — Fused top-K expert selection kernel

**Three-phase EP implementation:**

**Phase 1: SPLIT_AXIS_2 tensor split**  
Split expert weight tensors along axis 2 (expert dimension), one shard per GPU. Each GPU owns a subset of experts.

**Phase 2: Meta backend EP dispatch**  
`ggml-backend-meta.cpp` updated (+90 lines) to dispatch MUL_MAT_ID ops using EP routing — each GPU only computes the experts it owns.

**Phase 3: SYCL EP masking**  
`ggml-sycl.cpp` updated (+836 lines) with EP mask logic. After expert selection, each GPU applies a routing mask so only tokens assigned to that GPU's experts get computed.

**Subgraph fix:** Deferred AllReduce for gate/up MUL_MAT_ID reduced subgraph count from 193 → 97.

**Known issue at time of commit:** np>1 concurrency bug (garbled output when using multiple parallel slots)

---

### Phase 4: EP Concurrency Bug Fix
**Branch:** `ep-tp-combined` (HEAD)  
**Commit:** `5a590cbf7` — "Fix EP np>1 concurrency: invalidate Q8 cache between expert iterations in batch path"  
**Date:** Thu Mar 19 18:19:01 2026  
**Status:** ACTIVE  
**Author:** Ryan  
**Files:** `ggml-sycl/ggml-sycl.cpp` (+12 lines)

**Root cause:** `src1_contiguous` pointer was reused across experts. Q8 cache returned stale quantized data from previous expert. Fix: invalidate cache before each matmul in the batch path.

---

### Phase 5: Row-Split Multi-GPU (Experimental — Nemotron)
**Branch:** `master` / `expert-parallelism` (these are the same HEAD)  
**Commits:** `daf5a6fca` → `8b6d3cb0d` → `74233037c` → `e0ebf6ba0`  
**Status:** EXPERIMENTAL — Qwen3 works, Nemotron garbled

This was a parallel experimental track running on the `master`/`expert-parallelism` branches (NOT merged to `ep-tp-combined`). Aimed at Nemotron Nano 30B (SSM/Mamba architecture).

#### Row-Split Bug Fix #1: 5 Bugs Fixed
**Commit:** `daf5a6fca` — "SYCL row-split: fix 5 bugs enabling multi-GPU row-split on Intel Arc A770"  
Co-authored with Claude Opus 4.6

**5 bugs fixed:**
1. **binbcast split pointer:** MUL/ADD ops used `tensor->data` (invalid for split buffers) → fixed to resolve via `data_device[i]`
2. **1D tensor replication:** norm weights got 0 rows on some devices → replicate small tensors across all devices
3. **MMVQ Q8_1 ds visibility:** half2 scale not visible to subsequent kernels on Intel Arc → force DMMV for split tensors
4. **Cross-device merge:** `dpct::async_dpct_memcpy` silently fails on Intel Arc without P2P → replaced with per-column `dev2dev_memcpy`
5. **GEMM pool allocation:** dequantize buffer allocated on wrong device pool → use `ctx.pool(id)`

Also: SSM_SCAN SYCL port (Mamba-1/2), Nemotron-H assertion fix, universal local re-quantization for off-device Q8_1.

**Result:** Qwen3-0.6B row-split correct on 1/2/3 GPUs. Nemotron loads but garbled.

#### Row-Split Bug Fix #2: SSM Pointers + Broader Replication
**Commit:** `8b6d3cb0d`  
Co-authored with Claude Opus 4.6

- SSM_CONV and SSM_SCAN: resolve split buffer pointers for weight tensors
- Broader replication threshold: tensors < 256KB replicated across all devices
- Disable MMVQ and MMQ entirely for split tensors
- Add graph tracing instrumentation

**Result:** Nemotron produces real words instead of dots/commas. Still garbled.

#### Row-Split Bug Fix #3: Full Tensor Replication
**Commit:** `74233037c` — "Fix tensor replication: replicate full data even when device has partial shard"  
Co-authored with Claude Opus 4.6

**Root cause:** Replication only triggered for `nrows_split==0`, but e.g. `ssm_conv1d.weight` (6144 rows) got split with each device getting ~2048. SSM_CONV kernel on device 0 needs ALL rows. Now: when `needs_replication` (< 256KB), ALL devices get full tensor.

**Result:** More real words but still garbled ("The question't same time't own...").

#### Row-Split Bug Fix #4: Recurrent State + Cleanup
**Commit:** `e0ebf6ba0` — "SYCL row-split: clear recurrent state after reservation, clean debug"  
Co-authored with Claude Opus 4.6

- Clear memory after `sched_reserve()` to prevent SSM state corruption from reservation pass running Mamba2 with dummy data
- Disable MMVQ/MMQ for ALL multi-GPU configs (not just split tensors)
- SSM_CONV inputs now match between row-split and layer-split

**Status:** Qwen3 row-split works. Nemotron Nano row-split still diverges in block outputs.

---

### Phase 6: Benchmark Suite
**Branch:** `stable-baseline`  
**Commit:** `09ce74dd2` — "Add benchmark suite: run-bench.sh, run-suite.sh, README"  
**Date:** Thu Mar 19 20:27:36 2026  
**Status:** ACTIVE (tooling)  
**Author:** Ryan  

New files:
- `bench/run-bench.sh` — single benchmark run with garble detection, per-slot timing, JSON results
- `bench/run-suite.sh` — runs multiple configs and aggregates
- `bench/README.md`

Built on `stable-baseline` (ac12736a2 + no EP), intended as the clean measurement base.

---

## Optimization Inventory Summary

| Technique | Commit | Branch | Status | Measured Gain |
|---|---|---|---|---|
| BF16 AllReduce | `28e4790ed` | tensor-parallelism-upstream | ACTIVE | Reduces bandwidth vs F32 |
| Delayed AllReduce for MoE | `ae0334ffa` | tensor-parallelism-upstream | ACTIVE | Fewer sync round-trips |
| SYCL Graph Recording (6 bug fixes) | `ac12736a2` | ep-tp-combined | ACTIVE | 2.4→2.9 t/s (+20%) |
| `.wait()` removal / bounce buffer bypass | `ac12736a2` | ep-tp-combined | ACTIVE | Baked into above |
| MoE pooled pinned buffer + event wait | `ac12736a2` | ep-tp-combined | ACTIVE | Baked into above |
| MoE decode fast path (pre-quantize Q8) | `ac12736a2` | ep-tp-combined | ACTIVE | Baked into above |
| Async batched model load (180s→25s) | `ac12736a2` | ep-tp-combined | ACTIVE | 7x load speedup |
| Mode-specific compiler cache | `ac12736a2` | ep-tp-combined | ACTIVE | Avoids recompile |
| Expert Parallelism (3-phase) | `8ba0c7f9e` | ep-tp-combined | ACTIVE | 3.38→3.95 t/s (+17%) |
| 4 new fused SYCL kernels (EP) | `8ba0c7f9e` | ep-tp-combined | ACTIVE | Part of EP |
| Subgraph count reduction (193→97) | `8ba0c7f9e` | ep-tp-combined | ACTIVE | Part of EP |
| EP Q8 cache invalidation (np>1 fix) | `5a590cbf7` | ep-tp-combined | ACTIVE | Fixes garbled output |
| Row-split 5 bug fixes (Qwen3/Nemotron) | `daf5a6fca` | master/expert-parallelism | EXPERIMENTAL | Qwen3 works, Nemotron WIP |
| SSM/Mamba row-split support | `8b6d3cb0d`+`74233037c`+`e0ebf6ba0` | master/expert-parallelism | EXPERIMENTAL | Nemotron partially works |

---

## Abandoned / Not Pursued Techniques

These are mentioned in `OPTIMIZATION_PLAN.md` and `ORCHESTRATOR_BRIEF.md` but NOT yet implemented:

| Technique | Why Not Done | Priority |
|---|---|---|
| Option 4b: Full GPU-side MUL_MAT_ID dispatch | Hard — requires rewriting expert routing kernel | Later |
| Reduce subgraph count further (50-64 target) | Not yet coded, would require meta backend changes | Medium |
| Kernel fusion (RMSNorm+matmul, SiLU+mul) | Only matters after launch overhead solved | Low |
| Batched command lists (env var test) | `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` — easy test, never formally measured | Easy quick win |

---

## Key Measurements (from ORCHESTRATOR_BRIEF.md)

```
Config                           | t/s     | Notes
---------------------------------|---------|----------------------------------
Legacy row-split                 | 0.6     | Host stall bound
TP naive                         | 1.93    | Per-copy malloc
TP + staging pool                | 2.24    | Reused pinned buffers
TP + direct AllReduce            | 2.51    | Host-staged sum
After ac12736a2 (all Phase 2)   | 2.9     | 30B MoE
Expert Parallelism (8ba0c7f9e)  | 3.95    | 30B MoE (+17% over TP)
0.6B TP after Phase 2            | 18.5    | Up from 13 t/s
0.6B graph replay (est)          | ~125    | 8ms/token before crash
Target                           | 15+     | Graph replay + batched + fewer subgraphs
```

---

## Critical Files to Understand

| File | Role |
|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Main SYCL backend — graph_compute, mul_mat, AllReduce, EP masking |
| `ggml/src/ggml-sycl/common.hpp` | Context structs, graph_cache_entry struct |
| `ggml/src/ggml-backend-meta.cpp` | Meta backend — TP subgraph dispatch, AllReduce, EP routing |
| `ggml/src/ggml-sycl/mmvq.cpp` | Fused matrix-multiply vector quantized |
| `ggml/src/ggml-sycl/fused-moe-mmvq.cpp` | NEW: Fused MoE MMVQ kernel (EP Phase 3) |
| `ggml/src/ggml-sycl/fused-topk-select.cpp` | NEW: Top-K expert selection kernel |
| `ggml/src/ggml-sycl/fused-expert-agg.cpp` | NEW: Expert aggregation kernel |
| `ggml/src/ggml-sycl/fused-add-rmsnorm.cpp` | NEW: Fused AddRMSNorm kernel |
| `OPTIMIZATION_PLAN.md` | Full optimization analysis and implementation guide |
| `ORCHESTRATOR_BRIEF.md` | Task pipeline overview, agent roles, current measurements |
| `bench/run-bench.sh` | Benchmark with garble detection |

---

## Notes on Branch Strategy

- **`ep-tp-combined`** is the only branch with EP. It also has all Phase 2 optimizations.
- **`stable-baseline`** = `master` = reference without EP. Used for clean benchmarking.
- **`master`/`expert-parallelism`** = row-split experimental track for Nemotron/Mamba. NOT merged to ep-tp-combined (different GPU splitting strategy).
- The row-split work (Phases 3-4 on master) is a completely different sharding approach from TP (which splits tensor columns). They're not combined yet.

---

## What's Next (Based on OPTIMIZATION_PLAN.md)

1. **Batched command lists test** — just set `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` and measure. Easy.
2. **Graph recording for MoE** — split subgraphs at MUL_MAT_ID boundaries (Option 4a)
3. **Reduce subgraph count** — target 50-64 from current 97
4. **Nemotron Nano row-split** — fix remaining numerical divergence in block outputs
5. **Combine EP + graph recording** — the big unlock for 15+ t/s
