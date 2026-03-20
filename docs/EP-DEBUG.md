# EP Output Corruption — Debug Guide

**Status:** UNSOLVED as of 2026-03-19 21:37 CDT
**Branch:** `ep-tp-combined` in `/home/ryan/llm-stack/llama.cpp-eptp/`
**Symptom:** Expert Parallelism produces garbled output at np=1. Speed is correct (250ms, 3.95 t/s). TP on the same model produces clean output.

---

## Quick Repro

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh

# WORKS — TP on stable build, clean output, 3.5 t/s
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  /home/ryan/llm-stack/llama.cpp-stable/build-sycl/bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 --no-warmup

# BROKEN — EP on eptp build, garbled output, 3.95 t/s
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  /home/ryan/llm-stack/llama.cpp-eptp/build-sycl/bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 --no-warmup

# Test with:
curl -s -X POST http://127.0.0.1:18404/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20}'
```

---

## What EP Changes (3 files)

### 1. `src/llama-model.cpp` — Phase 1
Expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) split on AXIS_2 (expert dimension) instead of AXIS_0/1 (weight dimension).

With 96 experts, 3 GPUs → each GPU gets 32 experts as a contiguous [hidden, intermediate, 32, 1] tensor.

### 2. `ggml/src/ggml-backend-meta.cpp` — Phase 2
- Separate `handle_mul_mat_id` that returns PARTIAL when src0 has SPLIT_AXIS_2
- Computes `expert_offset` per GPU: `sum(ne[0..j-1])`
- Writes to op_params: `[1]=expert_offset, [2]=n_local_experts`
- Subgraph boundary deferral for gate/up MUL_MAT_ID (only AllReduce after down projection)

### 3. `ggml/src/ggml-sycl/ggml-sycl.cpp` — Phase 3
- Reads `op_params[1]` as expert_offset, `op_params[2] > 0` as EP flag
- Pre-zeros dst with `memset(0)` when EP flag set
- Filters experts: skip if `expert_id < offset || expert_id >= offset + n_local`
- Remaps: `local_index = expert_id - expert_offset`
- All 3 MUL_MAT_ID paths modified: fused, decode fast, batch

---

## Investigated Root Causes

### Theory 1: Tensor Rotation (DEBUNKED)
**Hypothesis:** Rotation scrambles expert-to-GPU mapping per layer.

**Debunked by Opus analysis:** With 96 experts / 3 GPUs, all `ne[j]=32` (equal). Rotation shuffles identical values — it's a no-op. `set_tensor` always distributes data sequentially: GPU0=experts 0-31, GPU1=32-63, GPU2=64-95, regardless of rotation. The fix (`rotation=0`) was applied but is a no-op for equal splits.

**Still applies IF:** Model has `n_expert % n_devices != 0` (unequal splits).

### Theory 2: op_params Slot Mismatch (LATENT BUG — works by coincidence)
Phase 2 writes `[3]=EP flag`, Phase 3 reads `[2]>0` as EP flag. Works because `n_local_experts=32 > 0` is truthy. Not the corruption cause but should be cleaned up.

### Theory 3: ne02 vs n_local_experts (TOP SUSPECT)
Phase 3 reads `n_local_experts` from `ne02` (tensor shape dimension 2), NOT from `op_params[2]`. If `ne02` on the simple (per-GPU) tensor is 32 (correct), this works. If `ne02` is still 96 (global), then expert filtering passes all experts but local indexing overflows the 32-expert buffer.

**Opus analysis says code is correct**, but EP output IS garbled (confirmed side-by-side: TP=clean, EP=garbage on same model+prompt). Something in the actual runtime data flow is wrong even if the code reads correctly.

**THIS IS THE TOP SUSPECT.** Check:
```bash
# Add debug print to ggml_sycl_mul_mat_id:
# printf("EP: ne02=%ld expert_offset=%d n_local=%ld\n", ne02, expert_offset, n_local_experts);
```

### Theory 4: Fused Path Buffer Guard
`ggml_sycl_mul_mat_id_fused()` has guard `!ggml_backend_buffer_is_sycl_split(src0->buffer)`. With EP + SPLIT_AXIS_2, src0 is still in a split buffer → fused path is SKIPPED → falls through to per-expert dispatch. The per-expert dispatch may have bugs in EP mode.

### Theory 5: AllReduce on Wrong Tensor
With the deferred AllReduce (gate/up skip, only down gets AllReduce), the AllReduce happens on the MUL_MAT_ID output. But in EP, the output has zeros for non-owned experts. AllReduce SUM should combine correctly (0 + real = real). Verify this is actually happening.

### Theory 6: Pre-zeroing Race
`stream->memset(dst, 0)` is async. The expert matmul kernel that follows writes to specific output positions. If the memset and matmul overlap (different stream or OOO execution), zeros could overwrite real results. Should be safe on SYCL in-order queue, but verify.

---

## Debug Strategy

### Step 1: Add printf debugging
In `ggml_sycl_mul_mat_id()` (ggml-sycl.cpp), after reading op_params:
```cpp
if (expert_offset > 0 || ep_flag) {
    fprintf(stderr, "EP DEBUG: ne02=%ld expert_offset=%d ep_flag=%d n_local=%ld\n",
            ne02, expert_offset, ep_flag, n_local_experts);
}
```

### Step 2: Single-layer test
If possible, run with only 1 MoE layer (or disable EP on all but layer 0) to isolate whether the bug is layer-dependent.

### Step 3: Compare expert activations
For a given input, log which experts are activated on each GPU under TP vs EP. They should be the same globally, just dispatched to different GPUs.

### Step 4: Check the AllReduce output
After AllReduce, dump the first few floats of the output tensor. Under TP, all values should be nonzero. Under EP, same after AllReduce. If EP post-AllReduce has zeros where TP doesn't, the combination is wrong.

---

## Git State

```bash
cd /home/ryan/llm-stack/llama.cpp-eptp
git log --oneline -5
# 5a590cbf7 Fix EP np>1 concurrency: invalidate Q8 cache between expert iterations in batch path
# 8ba0c7f9e Expert Parallelism implementation for 96-expert REAM model
# ac12736a2 All optimizations from 2026-03-18/19 session
# ae0334ffa delay AllReduce for Moe for less I/O
# 08400041d Enable the previous allreduce implementation

# Plus uncommitted: rotation=0 fix + op_params cleanup (DID NOT FIX corruption)
git diff --stat
```

## Key Files

| File | What to look at |
|------|----------------|
| `src/llama-model.cpp:128-138` | EP tensor split assignment (AXIS_2, rotation) |
| `ggml/src/ggml-backend-meta.cpp:922-936` | expert_offset calculation + op_params write |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4194-4210` | EP params reading + pre-zeroing |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:3929` | Fused path EP flag |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4050` | Decode fast path EP flag |

## Full Analysis

See `/home/ryan/.openclaw/workspace/outputs/oq1-ep-corruption-clues.md` (350 lines, 5 bugs identified).
