# Active Techniques: stable-baseline Branch Code Analysis

**Branch:** `stable-baseline` (`/home/ryan/llm-stack/llama.cpp-stable`)  
**Tip commit:** `ac12736a2` — "All optimizations from 2026-03-18/19 session"  
**Analysis date:** 2026-03-19

---

## 1. Compile-Time Macros (gating whole feature blocks)

| Macro | Default in CMakeLists | Effect |
|---|---|---|
| `GGML_SYCL_GRAPH` | **ON** (`ggml/CMakeLists.txt:248`) | Enables entire graph recording/replay path and async mem alloc |
| `GGML_SYCL_F16` | OFF (`ggml/CMakeLists.txt:247`) | 16-bit floats for SYCL calculations |
| `GGML_SYCL_FORCE_MMQ` | not in CMakeLists (manual define) | Forces MMQ path for all quantized matmul |
| `GGML_SYCL_DNNL` | not in CMakeLists (manual define) | Enables oneDNN integration |
| `SYCL_FLASH_ATTN` | controlled externally | Enables Flash Attention kernel; without it, FA is forced off at init |
| `SYCL_USE_XMX` | not set | Would gate XMX-accelerated MMQ batch size limit |
| `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` | runtime hardware detection | Enables async malloc/free for graph-safe reorder |

**Code locations:**
- `ggml/src/ggml-sycl/ggml-sycl.cpp:241-259` — compile-flag logging at startup
- `ggml/src/ggml-sycl/ggml-sycl.cpp:34` — `#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`

---

## 2. Runtime Environment Variables

All read once in `ggml_check_sycl()` (called on first backend use). Code: `ggml-sycl.cpp:220–302`

### Core Variables

| Variable | Default | Global | Effect |
|---|---|---|---|
| `GGML_SYCL_DEBUG` | `0` | `g_ggml_sycl_debug` | Enable verbose per-op debug logging (macro `GGML_SYCL_DEBUG(...)`) |
| `GGML_SYCL_DISABLE_GRAPH` | **`1`** (disabled by default!) | `g_ggml_sycl_disable_graph` | `0` = enable graph recording/replay; `1` = pure direct dispatch |
| `GGML_SYCL_DISABLE_OPT` | `0` | `g_ggml_sycl_disable_optimize` | `1` disables weight reordering (`should_reorder_tensor` returns false) |
| `GGML_SYCL_DISABLE_DNN` | `0` | `g_ggml_sycl_disable_dnn` | `1` disables oneDNN path (only relevant if compiled with `GGML_SYCL_DNNL`) |
| `GGML_SYCL_PRIORITIZE_DMMV` | `0` | `g_ggml_sycl_prioritize_dmmv` | `1` forces DMMV over MMVQ even when reorder is available |
| `GGML_SYCL_BOUNCE_BUFFER` | `0` | `g_ggml_sycl_bounce_buffer` | `1` re-enables PVC bounce-buffer path; disables Arc's direct mmap DMA |
| `GGML_SYCL_DIRECT_MMAP_DMA` | `0` | `g_ggml_sycl_direct_mmap_dma` | Force-enable direct DMA even on non-Arc GPUs (auto-on for Arc) |
| `GGML_SYCL_ENABLE_FLASH_ATTN` | `1` | `g_ggml_sycl_enable_flash_attention` | `0` disable Flash Attention; requires `SYCL_FLASH_ATTN` compile flag or forced off |
| `GGML_SYCL_PROFILE_EXPERTS` | `0` | `profile_experts` (static local) | `1` dump expert routing CSV to `/tmp/expert_routing.csv` |

**Derived at init:**
- `g_ggml_sycl_use_async_mem_op` — set to `!g_ggml_sycl_disable_graph` IFF compiled with `GGML_SYCL_GRAPH && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` AND all devices support `ext_oneapi_async_memory_alloc`. Falls back to `0` if any device lacks support. (`ggml-sycl.cpp:296-304`)

### External SYCL/Level Zero Variables (not read by this code directly)

| Variable | Effect |
|---|---|
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | Intel Level Zero driver setting: `0` = batched cmdlists (lower overhead per kernel), `1` = immediate (lower latency, higher overhead) |
| `ZE_AFFINITY_MASK` | Controls which GPUs are exposed to SYCL (e.g., `0,1,2` = all 3) |

---

## 3. Graph Recording / Replay System

**File:** `ggml/src/ggml-sycl/ggml-sycl.cpp:4755–5037`  
**Guard:** `#ifdef GGML_SYCL_GRAPH` (compile-time) + `!g_ggml_sycl_disable_graph` (runtime)

### Architecture

The graph path implements **segmented dispatch** with a **two-level cache**:

```
ggml_backend_sycl_graph_compute()         ← entry point
  └── if GGML_SYCL_GRAPH && !disable_graph:
        ├── instrument_graph_compute()    ← debug logging (first 3 calls/device → /tmp/graph_instrument.log)
        ├── device capability check (cached thread_local):
        │     limited_graph = ext_oneapi_limited_graph  ← Arc A770: YES (limited, non-updatable)
        │     can_update    = ext_oneapi_graph           ← Arc A770: NO (full updatable graph not supported)
        └── segmented graph dispatch:
              iterate cgraph->nodes:
                ├── NOOP nodes (RESHAPE, TRANSPOSE, VIEW, PERMUTE, NONE) → skip, don't break segment
                ├── NON-GRAPHABLE (CONCAT, MUL_MAT_ID, or MUL_MAT without async_mem) → direct dispatch + flush segment
                └── GRAPHABLE → accumulate into segment
              → replay_or_record_segment() for each closed segment
```

### `replay_or_record_segment()` (`ggml-sycl.cpp:4847`)

Two-level key design:
- **`topo_key`** = hash of (segment_index, node count, op codes, tensor types) — stable across KV cache growth
- **`shape_hash`** = hash of all `ne[]` dimensions — exact shape fingerprint

Cache lookup and action:

| Condition | Action |
|---|---|
| `ptr_miss_count >= 2` | **Unstable segment** — direct dispatch forever, skip graph entirely |
| Cache hit + non-updatable + pointers match | **Replay** `ext_oneapi_graph()` directly |
| Cache hit + non-updatable + pointers mismatch | Increment `ptr_miss_count`, re-record once; at count≥2 → unstable |
| Cache hit + updatable + shape matches | **Update** existing executable graph via `exec->update()` |
| Cache hit + updatable + shape mismatch | Re-record, replace cache entry |
| Cache miss | **Record**: `begin_recording()` → dispatch all nodes → `end_recording()` → `finalize()` → cache |
| Segment < 3 compute nodes | Skip graph overhead, dispatch directly |

**Arc A770 specific:** `limited_graph` support only → `can_update = false` → all cached entries are non-updatable → pointer-stability tracking is used instead of shape-based updates.

### Graph cache struct (`common.hpp:493`)
```cpp
struct graph_cache_entry {
    unique_ptr<executable_graph> exec;
    bool updatable;
    uint64_t shape_hash;
    vector<void*> recorded_data_ptrs;  // for pointer-stability validation
    int ptr_miss_count;                // 0→1: re-record, ≥2: direct dispatch
};
unordered_map<uint64_t, graph_cache_entry> graph_cache;
bool graph_recording = false;  // used by some ops to adjust behavior during recording
```

### Interaction with MUL_MAT reordering

`MUL_MAT` with `g_ggml_sycl_use_async_mem_op = true` → **graphable** (`is_node_graph_compatible` returns `g_ggml_sycl_use_async_mem_op`)  
`MUL_MAT` without async mem → **non-graphable** → always direct dispatch

---

## 4. Weight Reordering (MMVQ Optimization)

**Files:** `ggml-sycl.cpp:3507–3764`, `mmvq.cpp`

### What it does
Reorders quantized weight blocks in GPU memory at upload time for coalesced access patterns during matrix-vector multiplication. Activated once per tensor (marked `extra->optimized_feature.reorder`).

### Supported types
- `ggml_sycl_supports_reorder_dmmv()` → Q4_0, Q4_K, Q6_K
- `ggml_sycl_supports_reorder_mmvq()` → Q4_0, Q4_K, Q6_K  
- `ggml_sycl_supports_reorder_mul_mat_sycl()` → Q4_0, Q4_K, Q6_K

### Reorder kernels (`ggml-sycl.cpp:3586–3723`)
- `reorder_qw_q4_0()` — reorders Q4_0 weight blocks
- `reorder_qw_q4_k()` — reorders Q4_K blocks
- `reorder_qw_q6_k()` — reorders Q6_K blocks
- All use async `stream->parallel_for()` with `wait_and_throw()` only if `!g_ggml_sycl_use_async_mem_op`

### `should_reorder_tensor()` (`ggml-sycl.cpp:3728`)
Returns `!g_ggml_sycl_disable_optimize` AND tensor not already reordered AND type supported.

### `opt_for_reorder()` (`ggml-sycl.cpp:3737`)
Called at mul_mat dispatch time. Checks algo (DMMV/MMVQ/mul_mat_sycl) and type support. If eligible, calls `reorder_qw()` → marks tensor as reordered. This is a one-time cost per weight tensor.

### Interaction with graph recording
- During graph recording (`ctx.graph_recording = true`), `reorder_qw` uses async path (no `.wait()`) so the reorder op is captured in the graph
- `quantize_and_reorder_q8_1_soa` quantizer used for activation quantization when reorder active

---

## 5. MoE Decode Fast Path

**File:** `ggml-sycl.cpp:3866–3950` (function `ggml_sycl_mul_mat_id_decode_fast_path`)

### Activation condition
- Only for decode (`ne12 == 1`)
- Uses MMVQ-capable types
- Contiguous src1 and dst
- Non-split buffer

### What it does vs. generic path
Generic `MUL_MAT_ID` path: per-expert loop → full `ggml_sycl_mul_mat()` dispatch including type switch, device queries, capability checks every expert.

Fast path:
1. Resolve reorder state and kernel function pointer **once** before the expert loop
2. Quantize `src1` to q8_1 **once** — shared across all expert dispatches
3. Loop over expert IDs → raw kernel submissions back-to-back (no re-entry into dispatch machinery)

**Code location:** `ggml-sycl.cpp:4071` — `if (ne12 == 1 && ggml_sycl_mul_mat_id_decode_fast_path(...)) return;`

### Pooled pinned IDs buffer
- `ctx.get_ids_pinned(ids_nbytes)` — pre-allocated pinned host buffer for expert ID tensor
- Grows on demand, reused across calls
- Avoids per-call `malloc_host` overhead
- **Code:** `common.hpp:337–388`, `ggml-sycl.cpp:4027`

---

## 6. AllReduce (Tensor Parallelism)

**File:** `ggml-sycl.cpp:5849–5950` (SYCL implementation)  
**File:** `ggml-backend-meta.cpp:1091–1205` (meta backend dispatch)

### SYCL AllReduce (`ggml_backend_sycl_allreduce_tensor`)

Host-staged implementation (registered as `ggml_backend_allreduce_tensor` proc address):

1. **Step 1:** All GPUs → host in parallel (each GPU's partial result to pinned staging buffer, parallel `memcpy`)
2. **Wait:** `stream->wait()` for all GPU→host copies
3. **Step 2:** CPU sum in-place into `host_bufs[0]` — simple float loop, auto-vectorizable (no aliasing)
4. **Step 3:** Host → all GPUs in parallel
5. **Wait:** `stream->wait()` for all host→GPU copies

**Advantage over generic fallback:** 1 round of I/O (N GPUs→host→N GPUs) vs. N-1 pairwise exchange rounds of the `allreduce_fallback`.

**Timing:** Self-profiling — times first 5 calls + every 200th. Logs `[AR] #N ne=X Xms`.

**Staging buffer:** `ctx0->get_staging(nbytes * n_backends)` — pinned host memory, grows as needed, one allocation for all GPUs.

### Meta backend dispatch (`ggml-backend-meta.cpp:1187–1205`)

```
for each subgraph:
    compute all backends async
    if n_backends > 1 and not last subgraph:
        try ggml_backend_allreduce_tensor (SYCL fast path above)
        if fails → allreduce_fallback (pairwise P2P exchange)
```

**Delayed AllReduce for MoE** (`ggml-backend-meta.cpp:917`):  
The meta backend analyzes graph structure to detect MoE patterns (ADD_ID → MUL → VIEW chains) and delays the AllReduce to reduce I/O. Code: `get_i_delayed()` lambda.

**Meta timing:** Logs `[META] #N subgraphs=X total=Xms` every 5 calls + every 20th.

---

## 7. Direct mmap DMA (Async Load Optimization)

**File:** `ggml-sycl.cpp:515–597`

### Behavior
For Intel Arc GPUs (`is_intel_arc` device flag, auto-detected at init), `ggml_backend_sycl_buffer_set_tensor()` bypasses the bounce-buffer copy:

- **Bounce buffer path (conservative):** `malloc(size) → memcpy → async DMA → record pending host buf`
- **Direct DMA path (Arc):** Skip malloc/memcpy, issue DMA directly from caller's mmap/file-backed memory

**Control:**
- Auto-enabled for `is_intel_arc` devices
- Can force-enable: `GGML_SYCL_DIRECT_MMAP_DMA=1`
- Can disable: `GGML_SYCL_BOUNCE_BUFFER=1` (overrides Arc auto-detection)
- Windows: always uses `.wait()` path (no async DMA)

**Removed optimization:** `queues_wait_and_throw()` was called before each upload, draining the entire device queue. Removed — in-order queue semantics already guarantee sequential execution. See comment at `ggml-sycl.cpp:515`.

### Async flush bookkeeping
Pending host buffers are tracked to ensure they're freed only after DMA completes (for the conservative path). The async path records `nullptr` as a sentinel.

---

## 8. oneDNN Integration (GGML_SYCL_DNNL)

**File:** `ggml-sycl.cpp:2432–2489`, `3271–3353`

Guard: `#if GGML_SYCL_DNNL` (compile-time) + `!g_ggml_sycl_disable_dnn` (runtime)

Used for:
- Batched mul_mat with strided non-contiguous tensors (oneDNN handles strides natively)
- Avoids `ggml_get_to_fp16_nc_sycl` overhead for strided data

**Interaction with graph recording:** `ggml-sycl.cpp:2422–2480` — during `ctx.graph_recording`, DNN path may use recording-compatible ops.

---

## 9. Flash Attention

**File:** `ggml-sycl.cpp:4577`, `fattn-tile.cpp`, `fattn-common.hpp`

Guard: `SYCL_FLASH_ATTN` compile flag (if not defined, `g_ggml_sycl_enable_flash_attention = 0` at init)  
Runtime: `GGML_SYCL_ENABLE_FLASH_ATTN` (default 1, only effective if compiled in)

Supports: head sizes 40, 64, 72, 80, ... (tiled dispatch in `fattn-tile.cpp`)

---

## 10. MatMul Dispatch Hierarchy

**File:** `ggml-sycl.cpp:3790–3861`

For each `MUL_MAT` node, dispatch order (first match wins):

1. **F16 permuted single-batch** → `ggml_sycl_mul_mat_vec_p021()` (KQ path)
2. **F16 non-contiguous single-batch** → `ggml_sycl_mul_mat_vec_nc()` (KQV path)
3. **F16 multi-batch** → `ggml_sycl_mul_mat_batched_sycl()` (KQ+KQV batched)
4. **DMMV** (dequantize mul mat vec) — if `can_use_dequantize_mul_mat_vec` AND (not prioritizing reorder OR type doesn't support reorder MMVQ)
5. **MMVQ** (quantized mul mat vec) — if `can_use_mul_mat_vec_q`:
   - If reordered: `quantize_and_reorder_q8_1_soa` + `ggml_sycl_op_mul_mat_vec_q`
   - Otherwise: `quantize_q8_1` + `ggml_sycl_op_mul_mat_vec_q`
6. **MMQ** (quantized batched) — if `ggml_sycl_supports_mmq()` (Q4_0 excluded unless not IQ2_XXS)
7. **Generic SYCL** → `ggml_sycl_op_mul_mat_sycl` (GEMM via oneAPI)

**PRIORITIZE_DMMV interaction:** When `g_ggml_sycl_prioritize_dmmv=1`, `ggml_sycl_supports_mmq()` returns false for most types, and reorder path is skipped for DMMV.

---

## 11. Graph Instrumentation (Debug)

**File:** `ggml-sycl.cpp:5039–5193`

- `instrument_graph_compute()` — logs first 3 calls per device to `/tmp/graph_instrument.log`
- Logs: call number, device, node count, per-node op/type/shape/ptr
- Runs unconditionally at start of `ggml_backend_sycl_graph_compute()` (even when graphs disabled)
- Self-limiting: stops after 3 calls per device

---

## 12. Key Interactions & Gotchas

### Graph + MUL_MAT reorder
`MUL_MAT` is only graphable when `g_ggml_sycl_use_async_mem_op = true`, which requires:
- `GGML_SYCL_GRAPH` compiled in
- `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` hardware support on all devices
- `GGML_SYCL_DISABLE_GRAPH=0`

If any condition fails, MUL_MAT falls out of graph segments (direct dispatch). This means on some hardware configurations, graph segments will be very short.

### Arc A770: `limited_graph` vs `graph`
Arc A770 supports `ext_oneapi_limited_graph` but NOT `ext_oneapi_graph`.  
- `limited_graph` → `can_update = false` → non-updatable graphs
- Non-updatable: recorded pointer addresses are baked in. If pointers change (MoE realloc), must re-record.
- Pointer stability tracking prevents infinite re-record loops for volatile MoE segments.

### Graph + CONCAT/MUL_MAT_ID
Both ops contain `stream->wait()` calls incompatible with recording. They always break graph segments. For 30B MoE inference, `MUL_MAT_ID` calls will always be direct-dispatched.

### AllReduce staging buffer growth
`get_staging()` is a grow-only allocator (no shrink). First call allocates `nbytes * n_backends * 1.25` rounded up. Subsequent calls only reallocate if needed. Single allocation shared across all GPUs for a given backend[0] context.

### DMMV vs MMVQ selection
`g_ggml_sycl_prioritize_dmmv=1` → forces DMMV for types that support both. Default (0) → reorder MMVQ preferred when available.

---

## Summary: Default State (GGML_SYCL_GRAPH compiled, Arc A770)

| Technique | Active by default? | Env override |
|---|---|---|
| Graph recording | **NO** — default is `GGML_SYCL_DISABLE_GRAPH=1` | Set `GGML_SYCL_DISABLE_GRAPH=0` to enable |
| Weight reordering | **YES** | `GGML_SYCL_DISABLE_OPT=1` to disable |
| MoE decode fast path | **YES** (for decode batch=1) | None |
| Pooled pinned IDs buffer | **YES** | None |
| Direct mmap DMA | **YES** (auto on Arc) | `GGML_SYCL_BOUNCE_BUFFER=1` to disable |
| Removed per-tensor `.wait()` | **YES** (always) | None — hardcoded removed |
| SYCL AllReduce | **YES** (for TP) | Falls back to P2P if fails |
| Delayed AllReduce (MoE) | **YES** (meta backend) | None |
| Flash Attention | Depends on compile | `GGML_SYCL_ENABLE_FLASH_ATTN=0` to disable |
| oneDNN | Depends on compile | `GGML_SYCL_DISABLE_DNN=1` to disable |
| Async mem alloc | Depends on hw+compile | Inherits from graph disable |

**To enable graph recording on Arc A770:**
```bash
GGML_SYCL_DISABLE_GRAPH=0 ./bin/llama-server ...
```

**To enable graphs + batched cmdlists (lower kernel overhead):**
```bash
GGML_SYCL_DISABLE_GRAPH=0 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 ./bin/llama-server ...
```
