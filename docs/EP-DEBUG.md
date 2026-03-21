# EP Output Corruption — Debug Guide

**Status:** CLARIFIED as of 2026-03-20
**Finding:** Garbling is NOT EP-specific — `Qwen3-30B-A3B-REAM-heretic-i1` produces garbled output on **both** stable and eptp builds. Same model + prompt works correctly on `Qwen3-30B-A3B-abliterated`.

**Action:** Use `Qwen3-30B-A3B-abliterated` for MoE testing. The REAM-heretic-i1 model has a quantization issue unrelated to EP implementation.

---

## Quick Repro

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh

# WORKS — abliterated MoE, clean output, ~28 t/s np=16
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  /home/ryan/llm-stack/llama.cpp-stable/build-sycl/bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode layer -ngl 99 -np 16 -c 512 --port 18410 --no-warmup

# BROKEN — REAM-heretic-i1 produces garbled output on same build
# Do NOT use this model for MoE testing
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  /home/ryan/llm-stack/llama.cpp-stable/build-sycl/bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/Qwen3-30B-A3B-REAM-heretic-i1-Q4_K_M.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 --no-warmup

# Test with:
curl -s -X POST http://127.0.0.1:18410/v1/chat/completions \
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

## Historical Notes (Pre-2026-03-20)

The investigation below was conducted using `Qwen3-30B-A3B-REAM-heretic-i1` which was later found to be broken independent of EP. The garbling issue is a model/quantization problem, not an EP implementation bug.

### Theory 1: Tensor Rotation
With 96 experts / 3 GPUs, all `ne[j]=32` (equal). Rotation shuffles identical values — it's a no-op. `set_tensor` distributes sequentially: GPU0=experts 0-31, GPU1=32-63, GPU2=64-95.

### Theory 2: op_params Slot Mismatch
Phase 2 writes `[3]=EP flag`, Phase 3 reads `[2]>0` as EP flag. Works because `n_local_experts=32 > 0` is truthy.

### Theory 3: ne02 vs n_local_experts
Investigated but inconclusive due to model issue.

### Theory 4: Fused Path Buffer Guard
`ggml_sycl_mul_mat_id_fused()` guard may cause fallback to per-expert dispatch.

### Theory 5: AllReduce on Wrong Tensor
Deferred AllReduce concern — may not be relevant given model issue.

### Theory 6: Pre-zeroing Race
SYCL in-order queue should prevent overlap, but worth verifying.

---

## Next Steps

1. Test EP with `Qwen3-30B-A3B-abliterated` model to confirm EP works correctly on a non-broken MoE
2. If EP works on abliterated model, the REAM model issue is likely a quantization problem in the heretic-i1 merge

## Key Files

| File | What to look at |
|------|----------------|
| `src/llama-model.cpp:128-138` | EP tensor split assignment (AXIS_2, rotation) |
| `ggml/src/ggml-backend-meta.cpp:922-936` | expert_offset calculation + op_params write |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4194-4210` | EP params reading + pre-zeroing |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:3929` | Fused path EP flag |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4050` | Decode fast path EP flag |
