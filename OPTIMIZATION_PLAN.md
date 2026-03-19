# Optimization Plan: 2.5 t/s → 15+ t/s on 3x Arc A770

## Current State

```
Qwen3-30B-A3B MoE, --split-mode tensor, 3x Arc A770
Token generation: 2.5 t/s (400ms/token)
Bottleneck: 4,365 SYCL kernel launches × 86µs/launch = 375ms (94% of token time)
AllReduce: 30ms/token (7.5%) — already optimized
Actual compute: ~2-3ms (0.5%) — GPU barely working
```

The problem is entirely kernel launch overhead on Intel's SYCL/L0 stack (86µs vs CUDA's 5-10µs).

## Proven Data Point

```
0.6B model, --split-mode tensor, graphs enabled:
Request 1 (compile + run): 5143ms
Request 2 (graph replay):  8ms ← THIS IS THE TARGET
```

Graph replay eliminates kernel launch overhead. 8ms for a full forward pass means the GPU is doing actual work, not waiting on host dispatch. The crash on request 2 is a tensor pointer invalidation bug, not a fundamental limitation.

---

## Optimization A: Fix SYCL Graph Recording for TP (PRIMARY — 10-50x speedup)

### Why It Crashed

The graph records kernel launches with specific tensor data pointers baked in. On the second token:
- KV cache has grown (new K/V entries appended)
- Batch dimensions changed (prefill → decode)
- The meta backend may reallocate compute buffers
- Recorded graph still references old pointers → segfault in mul_mat

### The Fix: Shape-Keyed Graph Cache

Instead of recording one graph and updating it, maintain a cache of compiled graphs keyed by tensor shape signature. During decode (fixed batch_size=1), the shape is constant after the first decode token, so the cached graph replays indefinitely.

```
Shape key = hash(n_nodes, [node.op, node.ne[0..3], node.type] for each node)

graph_cache: map<shape_key, executable_graph>

On each graph_compute:
  1. Compute shape key for current cgraph
  2. If cache hit → replay cached graph (but update data pointers)
  3. If cache miss → record, finalize, store in cache, replay
```

### Implementation Steps

#### Step 1: Understand What Changes Between Tokens

Add instrumentation to `graph_compute` that logs, for the first 3 calls:
- Number of nodes
- Each node's op, ne[0..3], type
- Each node's data pointer

This tells us exactly what's stable (shapes) vs what changes (pointers) between tokens.

Files: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph_compute)

#### Step 2: Implement Graph Update for Stable Shapes

The SYCL spec's `command_graph::update()` is designed for exactly this case — same graph structure, different data pointers. The existing code already tries this (line 4517):

```cpp
sycl_ctx->exec_graph->update(model_sycl_graph);
```

But it crashes because either:
- a) The update doesn't propagate split tensor pointers correctly
- b) The graph structure actually changed (different node count between prefill/decode)
- c) The update is silently failing and replaying with stale pointers

**Test:** Add try/catch around the update with detailed error logging. If update fails, re-finalize (already handled in existing code). If update succeeds but replay crashes, the update is buggy — fall back to re-record + re-finalize.

Files: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph_compute, lines 4490-4534)

#### Step 3: Separate Prefill vs Decode Graphs

Prefill (batch_size > 1) and decode (batch_size = 1) have different graph structures. Cache two separate graphs:

```cpp
struct graph_cache_entry {
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec;
    int n_nodes;   // for quick shape validation
    int64_t ne1;   // batch dimension — key differentiator
};

// Per-device graph cache: prefill graph + decode graph
graph_cache_entry graph_prefill;
graph_cache_entry graph_decode;
```

On each compute:
- If ne1 == 1 (decode) → use/build decode graph
- If ne1 > 1 (prefill) → use/build prefill graph or skip graphing (prefill is compute-bound)

Files: `ggml/src/ggml-sycl/common.hpp` (add cache to context), `ggml-sycl.cpp` (graph_compute)

#### Step 4: Handle MUL_MAT_ID (MoE Expert Dispatch)

`MUL_MAT_ID` blocks graph recording because it does `stream->memcpy()` + `stream->wait()` to copy routing IDs from device to host. Two options:

**Option 4a: Split subgraphs at MUL_MAT_ID boundaries**

The meta backend already splits the model into 97 subgraphs. Within each subgraph, some nodes are graphable (matmul, norm, add) and some aren't (MUL_MAT_ID). Split each subgraph into:
- Graphable prefix (attention ops) → record and replay
- Non-graphable MUL_MAT_ID → dispatch normally
- Graphable suffix (remaining ops) → record and replay

This requires modifying `graph_compute_impl` to iterate nodes and batch graphable runs:

```cpp
for (int i = 0; i < cgraph->n_nodes; i++) {
    if (is_graphable(node)) {
        graphable_run.push_back(node);
    } else {
        // Flush graphable run
        if (!graphable_run.empty()) {
            replay_or_record_graph(graphable_run);
            graphable_run.clear();
        }
        // Dispatch non-graphable node directly
        ggml_sycl_compute_forward(*sycl_ctx, node);
    }
}
// Flush remaining
if (!graphable_run.empty()) {
    replay_or_record_graph(graphable_run);
}
```

**Option 4b: Make MUL_MAT_ID graph-compatible**

Remove the host memcpy of routing IDs. Instead, keep the IDs on the device and use a GPU-side dispatch kernel that routes tokens to experts. This is what CUDA does — the expert dispatch happens entirely on the GPU.

This is harder but eliminates the MUL_MAT_ID graph incompatibility entirely.

**Recommendation:** Start with 4a (split at boundaries), pursue 4b later.

Files: `ggml/src/ggml-sycl/ggml-sycl.cpp` (graph_compute_impl, mul_mat_id)

#### Step 5: Per-Subgraph Graph Caching

The meta backend calls `graph_compute` 97 times per token (once per subgraph per device). Each of these is a separate graph. The cache needs to handle 97 × 3 = 291 graph entries:

```cpp
// Key: (device_id, subgraph_index, batch_size)
// Value: compiled executable graph
std::unordered_map<uint64_t, graph_cache_entry> graph_cache;
```

The subgraph index can be derived from a hash of the node ops, or simply by using a call counter that resets each token.

#### Expected Results

```
Before: 97 subgraphs × ~15 kernels × 86µs = 375ms/token → 2.5 t/s
After:  97 subgraphs × 1 graph replay × ~15µs = 1.5ms
        + 97 × AllReduce × 0.31ms = 30ms
        + actual compute ~3ms
        ≈ ~35ms → ~28 t/s theoretical

Conservative (with overhead): 60-100ms → 10-16 t/s
```

---

## Optimization B: Reduce Subgraph Count (Delayed AllReduce)

### Current: 97 subgraphs for 48 layers (~2 per layer)

Each subgraph boundary = one AllReduce + one sync point. The meta backend inserts AllReduce after every weight matmul that is split across devices.

### Opportunity

Some AllReduces can be deferred or eliminated:
- **Fuse attention subgraph:** Q/K/V projections + attention + output projection can be one subgraph with one AllReduce at the end (instead of 2-3 separate AllReduces)
- **Delayed MoE AllReduce:** PR #19378's commit `ae0334ffa` already delays FFN AllReduce until after expert outputs are summed. Verify this is working.
- **Skip redundant norms:** RMSNorm on already-reduced tensors doesn't need a separate AllReduce

### Target: 97 → ~50-64 subgraphs

```
Before: 97 subgraphs × 4.0ms = 388ms
After:  64 subgraphs × (lower ms with graphs)
Combined with A: 64 × ~0.5ms = 32ms → 31 t/s
```

### Implementation

This is primarily in the meta backend's split logic (`ggml-backend-meta.cpp`), specifically the `get_tensor_config()` function that determines split states and therefore AllReduce boundaries.

Files: `ggml/src/ggml-backend-meta.cpp` (split state logic, subgraph boundaries)

---

## Optimization C: Kernel Fusion Within Subgraphs

### Current: ~15 kernel launches per subgraph

Typical breakdown per transformer layer slice:
- RMSNorm (1 kernel)
- Q projection matmul (1 kernel)
- K projection matmul (1 kernel)
- V projection matmul (1 kernel)
- RoPE (1 kernel)
- Attention score matmul (1 kernel)
- Softmax (1 kernel)
- Attention value matmul (1 kernel)
- Output projection matmul (1 kernel)
- Residual add (1 kernel)
- RMSNorm (1 kernel)
- Gate projection matmul (1 kernel)
- Up projection matmul (1 kernel)
- SiLU + elementwise mul (1-2 kernels)
- Down projection matmul (1 kernel)

### Fusable combinations:
- RMSNorm + first matmul → 1 kernel (saves 1 launch)
- SiLU + mul → 1 kernel (saves 1 launch)
- Residual add + RMSNorm → 1 kernel (saves 1 launch)
- Q/K/V projections → 1 batched matmul (saves 2 launches)

### Target: 15 → 8-10 kernels per subgraph

```
Without graphs: 2,900 launches × 86µs = 249ms → 4.0 t/s
With graphs: negligible impact (graph handles fusion internally)
```

This is most valuable WITHOUT graph recording. With graphs, the launch overhead is already eliminated, so fusion only helps actual compute efficiency (minor).

### Priority: LOW (do after A and B)

Files: `ggml/src/ggml-sycl/` (new fused kernel files)

---

## Optimization D: Batched Command Lists

### What

`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` groups kernel submissions into batches before sending to the GPU. Instead of 15 separate host→GPU round-trips per subgraph, the L0 runtime batches them.

### Expected Impact

```
Typical batched overhead: ~20-30µs per kernel (vs 86µs immediate)
4,365 launches × 30µs = 131ms → 7.6 t/s
```

### Risk

Batched mode previously caused deadlocks with cross-queue `depends_on()` (event-based sync). But TP doesn't use events — it uses simple `queue.memcpy()` + `queue.wait()`. Should be safe.

### Implementation

Just set the env var. No code changes. But need to verify the AllReduce's `queue.wait()` flushes batched command lists correctly.

### Priority: MEDIUM — easy to test, moderate impact, good fallback if graphs don't work for MoE

---

## Execution Order

```
Priority  | Optimization              | Expected t/s | Effort    | Risk
──────────┼───────────────────────────┼──────────────┼───────────┼──────
1         | A: Graph recording fix    | 10-28 t/s    | 3-5 days  | Medium
          |   Step 1: Instrument      |              | 2 hours   |
          |   Step 2: Fix update      |              | 4 hours   |
          |   Step 3: Prefill/decode  |              | 2 hours   |
          |   Step 4a: MUL_MAT_ID    |              | 8 hours   |
          |   Step 5: Cache           |              | 4 hours   |
2         | D: Batched cmdlists       | 6-8 t/s      | 1 hour    | Low
3         | B: Reduce subgraphs       | +30-50%      | 2-3 days  | Medium
4         | C: Kernel fusion          | +20-30%      | 5+ days   | High
```

**Start with A (Step 1-2) to validate graph replay works for TP.**
**Test D in parallel — it's one env var change.**
**Stack B after A is working.**
**C is last resort — only if A+B+D aren't enough.**

## Combined Ceiling

```
A (graphs) + B (fewer subgraphs) + D (batched as fallback):
  ~50 subgraphs × 0.5ms graph replay = 25ms
  + 50 × 0.31ms AllReduce = 15ms
  + ~3ms compute
  = ~43ms → 23 t/s

With aggressive B (32 subgraphs):
  ~32 × 0.5ms = 16ms + 32 × 0.31ms = 10ms + 3ms = 29ms → 34 t/s
```

Target range: **15-30 t/s** on 3x Arc A770 with Qwen3-30B-A3B MoE.
