# Design: Deferred Attention PARTIAL — Combined AllReduce (Option B)

**Status:** Design / pre-implementation  
**Task:** #27 in optimization sequence (seq:5, subgraph reduction track)  
**Target:** Reduce TP subgraphs from 97 → ~49 per token by collapsing the wo AllReduce and EP-down AllReduce into a single combined boundary.  
**Technique name:** Megatron-style Deferred Attention Residual Fusion  

---

## 1. Background and Motivation

### Current graph structure per transformer layer (97 subgraphs total for 30B MoE)

```
[subgraph A]  inpSA → attn_norm → Q/K/V projections → RoPE → Flash Attention
[AllReduce]   wo MUL_MAT (SPLIT_AXIS_0) → PARTIAL → AllReduce → MIRRORED
[subgraph B]  residual_add (wo_out + inpSA) → ffn_norm → gate/up projections
[AllReduce]   EP down_exps MUL_MAT_ID (SPLIT_AXIS_2) → PARTIAL → AllReduce → MIRRORED
[subgraph C]  residual_add (moe_out + ffn_inp)
```

Each attention layer contributes ~2 AllReduce boundaries (1 for wo, 1 for MoE down). With 28 layers in Qwen3-30B-A3B MoE, that's 56 AllReduce boundaries producing 97 total subgraphs (the rest are driven by internal EP MUL_MAT_ID boundaries).

### Goal: 49 subgraphs

Defer the wo AllReduce so that `wo(PARTIAL)` and `MoE_down(PARTIAL)` can both be resolved in a **single combined AllReduce** at the end of the FFN block. This halves the number of attention-side AllReduce calls, saving ~24 AllReduce round-trips per token.

### Key mathematical validity

- **ADD(PARTIAL_A, PARTIAL_B) → PARTIAL** is valid **if and only if** both tensors are PARTIAL on the **same axis** (each GPU holds a partial sum for the same element positions). In TP mode, both wo and down projections split on axis 0, so this holds.
- **RMSNorm(PARTIAL) is NOT valid** — RMSNorm requires globally-synchronized values (it divides by the full-vector RMS). `ffn_norm` must therefore operate on the pre-residual `inpSA` (which is MIRRORED), not on the wo output.
- The combined residual is numerically equivalent:
  ```
  out = AllReduce(wo_partial + down_partial) + inpSA
      = (AllReduce(wo_partial) + inpSA) + AllReduce(down_partial)   [by linearity]
  ```
  The intermediate `ffn_inp = wo_out + inpSA` from the current design disappears as a fused-in value.

---

## 2. Interface Change: What `build_attn` Must Return

### Current interface (llama-graph.cpp)

```cpp
// build_attn returns: MIRRORED post-residual tensor
// (wo output has already been AllReduced inside build_attn or at the meta backend boundary)
ggml_tensor * cur = build_attn(inp_attn, wo, wo_b, Qcur, Kcur, Vcur, ...);
// cur is MIRRORED (globally synchronized)
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);  // residual: MIRRORED + MIRRORED
```

### New interface (after this change)

`build_attn` must return **two tensors** instead of one:

```cpp
struct llm_attn_result {
    ggml_tensor * wo_partial;   // PARTIAL (not yet AllReduced); shape: [n_embd, n_tokens]
    ggml_tensor * inpSA;        // MIRRORED (the pre-norm input, unchanged); shape: [n_embd, n_tokens]
};

llm_attn_result attn_result = build_attn_partial(inp_attn, wo, wo_b, Qcur, Kcur, Vcur, ...);
```

Or, if the existing `build_attn` signature is kept, the caller at the model layer (e.g., `qwen3moe.cpp`) simply uses the wo output **before** any residual add:

```cpp
// New caller pattern:
ggml_tensor * wo_partial = build_attn(inp_attn, wo, wo_b, Qcur, Kcur, Vcur, ...);
// wo_partial is PARTIAL — do NOT add inpSA here
// inpSA is kept alive separately
```

**What changes inside `build_attn`:** Nothing needs to change in `build_attn` itself — the meta backend already produces PARTIAL output from the wo `MUL_MAT (SPLIT_AXIS_0)`. The change is that the **caller stops adding `inpSA` immediately** and instead passes `wo_partial` directly into the MoE block.

**Critically:** The existing `build_attn` already returns the tensor before any residual add — the residual is done in the model file (e.g., `qwen3moe.cpp` line: `ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA)`). So `build_attn` itself needs **no signature change**; only the caller logic changes.

---

## 3. How `ffn_norm` Is Restructured

### Current flow

```
ffn_inp = wo_out(MIRRORED) + inpSA(MIRRORED)   → MIRRORED
cur     = RMSNorm(ffn_inp)                       → MIRRORED  ✓
```

### New flow

The `wo(PARTIAL)` output cannot be normalized. Instead, `ffn_norm` operates directly on `inpSA` (the pre-attn-residual input, which is always MIRRORED):

```
cur = RMSNorm(inpSA)    → MIRRORED  ✓
```

This produces the same result **in the combined output** because:
- Current: `ffn_norm(wo_out + inpSA)` = RMSNorm of the post-attn residual
- New: `ffn_norm(inpSA)` = RMSNorm of the pre-attn input

**These are NOT the same operation.** This is a **mathematical change** to the model computation.

> ⚠️ **STOP — Read this carefully before implementing.**
>
> `ffn_norm(wo_out + inpSA)` ≠ `ffn_norm(inpSA)` in general.
>
> This design is only valid if the model architecture allows normalizing `inpSA` instead of the post-attn-residual for the FFN input. This is **NOT the standard Qwen3-MoE** computation.
>
> **Two valid paths:**
>
> **Path A (mathematically correct, requires architectural support):** Keep the residual before ffn_norm but defer only the AllReduce. This means the model must compute `ffn_norm(AllReduce(wo_partial) + inpSA)` at the top of the MoE block, not at the bottom. The AllReduce fires between attn and MoE, but MoE and EP-down share a single boundary. This saves 0 AllReduces on the attn side but saves the EP boundary by merging it with a later sync.
>
> **Path B (simplified combined approach):** Use a parallel-residual / parallel-FFN structure where ffn_norm fires on `inpSA` independently of the attn output, and both residuals are summed at the end. Some architectures (e.g., GPT-NeoX with parallel attention) do this. Qwen3-MoE does NOT currently use this structure.
>
> **Implementers for tasks 28-30 must decide:** Either (A) keep the math correct and only fuse the EP AllReduce with the next boundary, or (B) add a parallel-residual mode to the model graph and verify numerics against a reference. Option B requires architectural validation against reference outputs.

For the remainder of this document, we document **what would be required** for Path B (the aggressive fusion), clearly flagged as requiring numeric validation.

---

## 4. How the MoE Block Receives Both Inputs

In the deferred-PARTIAL design (Path B), the MoE block receives:

| Tensor | State | Shape | Purpose |
|--------|-------|-------|---------|
| `wo_partial` | PARTIAL (axis 0) | [n_embd, n_tokens] | Will be combined with MoE output at the final residual |
| `inpSA` | MIRRORED | [n_embd, n_tokens] | Feed to ffn_norm for MoE input normalization |

```cpp
// Inside per-layer loop (new flow):
ggml_tensor * inpSA = inpL;

// 1. Attention (returns PARTIAL wo output, no residual add)
ggml_tensor * wo_partial = build_attn(inp_attn, wo, wo_b, Qcur, Kcur, Vcur, ...);
// wo_partial is PARTIAL — meta backend will NOT AllReduce here

// 2. FFN normalization — on inpSA (MIRRORED), not on wo_partial
ggml_tensor * ffn_cur = build_norm(inpSA, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

// 3. MoE computation — ffn_cur feeds gate, up, down expert projections
ggml_tensor * down_partial = build_moe_ffn(ffn_cur, ...);
// down_partial is PARTIAL (axis 0 for TP, from EP down MUL_MAT_ID)

// 4. Combined residual: ADD two PARTIAL tensors → still PARTIAL
// Both are partial on axis 0 → this ADD is valid and stays in the same subgraph
ggml_tensor * combined_partial = ggml_add(ctx0, wo_partial, down_partial);
// combined_partial is PARTIAL → triggers combined AllReduce

// 5. Add inpSA (MIRRORED) → final output is MIRRORED
// This add is the AllReduce boundary output + residual in one shot
ggml_tensor * cur = ggml_add(ctx0, combined_partial, inpSA);
// combined_partial triggers AllReduce → result is MIRRORED
// then ggml_add with inpSA (MIRRORED) → MIRRORED
```

---

## 5. Where the Combined Residual Add Fires

The combined residual fires **after** the EP down AllReduce boundary, replacing both the old `ffn_inp = wo_out + inpSA` and the old `cur = moe_out + ffn_inp` residuals.

### Subgraph structure (new, ~49 subgraphs for 28 layers):

```
[subgraph A]  inpSA → attn_norm → Q/K/V → RoPE → FlashAttn → wo(PARTIAL)
              inpSA → ffn_norm → gate_inp → expert routing → up/gate projections
              → EP down_exps MUL_MAT_ID → down_partial (PARTIAL)
              → wo_partial + down_partial → combined_partial (PARTIAL)
[AllReduce]   combined_partial → AllReduce → MIRRORED
[subgraph B]  MIRRORED + inpSA → output (MIRRORED) → next layer's attn_norm
```

**Subgraph count calculation:**
- 28 layers × 1 AllReduce boundary = 28 boundaries
- 28 + 1 (final norm + lm_head subgraph) = **29 subgraphs** (theoretical minimum)
- Realistic estimate with EP internal boundaries (gate routing, per-expert indexing): **~49 subgraphs**
- Current: 97 subgraphs

The wo AllReduce boundaries (28 total) are completely eliminated. Each layer drops from ~3.5 subgraphs to ~1.75.

---

## 6. Meta-Backend Changes Required

### 6.1 PARTIAL + PARTIAL → PARTIAL for ADD

**Current behavior:**  
`handle_generic` in `ggml_backend_meta_get_split_state` is called for `GGML_OP_ADD`. It requires all sources to have the **same** split state. If both `src[0]` and `src[1]` are `GGML_BACKEND_SPLIT_AXIS_PARTIAL`, `split_states_equal()` returns true only if both have the same `ne[]` array.

**Potential assertion failure:**  
Line 1022 (approximate) in `ggml-backend-meta.cpp`:
```cpp
if (node->op == GGML_OP_ADD_ID) {
    GGML_ASSERT(ggml_backend_meta_get_split_state(node->src[1], false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL);
}
```
This assertion guards `ADD_ID` (used in MoE aggregation), not plain `ADD`. However, the dispatch loop at line ~1071:
```cpp
if (!defer_ep_allreduce && split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
    // triggers AllReduce boundary
```
...will fire on the first PARTIAL tensor (wo_partial) unless we teach it to defer.

**Required change 1: Allow PARTIAL + PARTIAL ADD without splitting the graph.**

In `ggml-backend-meta.cpp`, in the `get_i_delayed()` lambda (or equivalent lookahead), add logic to detect the pattern:

```
node: GGML_OP_MUL_MAT (wo, SPLIT_AXIS_0) → wo_partial (PARTIAL)
...  : [several MoE ops, all non-boundary]
next: GGML_OP_ADD (wo_partial, down_partial) → combined_partial
```

When the meta backend sees `wo_partial` as a PARTIAL node, it should look ahead and confirm:
- The ADD that consumes `wo_partial` also has a PARTIAL src[1] (`down_partial`)
- The ADD's output is itself consumed by another ADD with a MIRRORED tensor (the `+ inpSA` residual)

If this pattern is found, defer the AllReduce past the ADD, treating `combined_partial` as still PARTIAL.

**Required change 2: `split_state` inference for ADD(PARTIAL, PARTIAL)**

In `handle_generic` (called for `GGML_OP_ADD`), the current logic already handles the case where both inputs have identical split states — it returns that split state. Specifically, if both `src[0]` and `src[1]` are `GGML_BACKEND_SPLIT_AXIS_PARTIAL` with the **same `ne[]`**, `split_states_equal()` returns true and `handle_generic` returns `PARTIAL`.

**This may already work correctly** as long as:
- Both tensors are PARTIAL on axis 0 (same axis, matching chunk sizes across GPUs)
- The `ne[]` arrays are identical (they will be, since both tensors have shape [n_embd, n_tokens])

**Verify:** Run with `GGML_META_DEBUG_SPLITS=1` after implementing the graph changes. If the combined ADD is correctly inferred as PARTIAL (not triggering an ASSERT for UNKNOWN), no meta backend change is needed for this case.

**Required change 3: AllReduce deferral past the PARTIAL+PARTIAL ADD**

The dispatch loop needs to recognize that a PARTIAL node followed by `ADD(PARTIAL, PARTIAL)` should NOT trigger an AllReduce at the first PARTIAL. Only the combined ADD result should trigger the boundary.

Modify `get_i_delayed()` or add a parallel `defer_attn_allreduce` flag:

```cpp
// New: detect deferred attn AllReduce pattern
// Pattern: wo_partial (PARTIAL from MUL_MAT axis-0) → ... → ADD(PARTIAL, PARTIAL) → ADD(MIRRORED)
bool defer_attn_allreduce = false;
if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
        node->op == GGML_OP_MUL_MAT &&
        /* src[0] is split on axis 0 (wo weight) */ ) {
    // Look ahead for the PARTIAL+PARTIAL ADD pattern
    for (int k = i + 1; k < cgraph->n_nodes && k < i + 200; k++) {
        ggml_tensor * future = cgraph->nodes[k];
        if (future->op == GGML_OP_ADD) {
            const auto ss0 = ggml_backend_meta_get_split_state(future->src[0], false);
            const auto ss1 = ggml_backend_meta_get_split_state(future->src[1], false);
            if (ss0.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                ss1.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                defer_attn_allreduce = true;
                break;
            }
        }
        // Stop if we hit a regular AllReduce boundary (non-EP MUL_MAT PARTIAL)
        if (future->op == GGML_OP_MUL_MAT) {
            const auto ss = ggml_backend_meta_get_split_state(future, false);
            if (ss.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) break;
        }
    }
}
const bool new_subgraph = ... && !defer_attn_allreduce && ...;
```

### 6.2 Summary of Meta-Backend Changes

| Change | File | Risk | Required? |
|--------|------|------|-----------|
| Verify PARTIAL+PARTIAL ADD propagates correctly | ggml-backend-meta.cpp | Low — may already work | Verify with debug flag |
| Add `defer_attn_allreduce` lookahead in dispatch loop | ggml-backend-meta.cpp | Medium — affects subgraph boundary placement | Yes |
| Remove ASSERT guarding ADD_ID for PARTIAL src[1] if hit | ggml-backend-meta.cpp | Low | Only if ASSERT fires |
| Handle wo-PARTIAL deferral through EP MoE ops in `get_i_delayed` | ggml-backend-meta.cpp | Medium — lookahead window | Yes |

---

## 7. Pseudocode: New Graph Construction Sequence

The following pseudocode replaces the per-layer body in `qwen3moe.cpp` (and similar MoE model files):

```
CURRENT (qwen3moe.cpp):
────────────────────────
for each layer il:
    inpSA = inpL

    # Step 1: Attention norm
    cur = RMSNorm(inpL, attn_norm)

    # Step 2: Q/K/V projections + RoPE
    Qcur, Kcur, Vcur = project_qkv(cur)
    Qcur, Kcur = rope(Qcur, Kcur)

    # Step 3: Attention + wo projection
    cur = build_attn(inp_attn, wo, wo_b, Qcur, Kcur, Vcur)
    # cur is PARTIAL here, but meta backend AllReduces it at subgraph boundary
    # [AllReduce #1: wo]

    # Step 4: Residual
    ffn_inp = cur + inpSA         # MIRRORED + MIRRORED → MIRRORED

    # Step 5: FFN norm
    cur = RMSNorm(ffn_inp, ffn_norm)    # MIRRORED → MIRRORED

    # Step 6: MoE
    moe_out = build_moe_ffn(cur, ...)
    # [AllReduce #2: EP down]

    # Step 7: Residual
    cur = moe_out + ffn_inp       # MIRRORED + MIRRORED → MIRRORED

    inpL = cur
```

```
NEW (deferred PARTIAL fusion):
──────────────────────────────
for each layer il:
    inpSA = inpL     # MIRRORED

    # Step 1: Attention norm (unchanged)
    attn_normed = RMSNorm(inpL, attn_norm)   # MIRRORED → MIRRORED

    # Step 2: Q/K/V projections + RoPE (unchanged)
    Qcur, Kcur, Vcur = project_qkv(attn_normed)
    Qcur, Kcur = rope(Qcur, Kcur)

    # Step 3: Attention + wo projection → PARTIAL (no residual add!)
    wo_partial = build_attn(inp_attn, wo, wo_b, Qcur, Kcur, Vcur)
    # wo_partial is PARTIAL (axis 0)
    # NO AllReduce here — meta backend must defer past this node

    # Step 4: FFN norm — on inpSA (MIRRORED), NOT on wo output
    # ⚠️ Math change: ffn_norm(inpSA) ≠ ffn_norm(wo_out + inpSA)
    #    Requires parallel-residual architecture validation (see Section 3)
    ffn_normed = RMSNorm(inpSA, ffn_norm)    # MIRRORED → MIRRORED

    # Step 5: MoE computation — receives normalized inpSA
    down_partial = build_moe_ffn(ffn_normed, ...)
    # down_partial is PARTIAL (axis 0 for TP, or axis 0 from EP aggregation)
    # NO AllReduce yet — EP deferral already handles this (get_i_delayed)

    # Step 6: Combined residual add — PARTIAL + PARTIAL → PARTIAL
    # Both tensors are PARTIAL on axis 0 → valid element-wise ADD
    combined_partial = wo_partial + down_partial
    # combined_partial is PARTIAL → triggers the ONE combined AllReduce
    # [AllReduce: wo + MoE combined]

    # Step 7: Final residual with inpSA — fires after AllReduce
    cur = combined_partial + inpSA
    # combined_partial AllReduces → MIRRORED; then + inpSA (MIRRORED) → MIRRORED

    inpL = cur
```

### Execution trace for meta backend (new):

```
node[i+0]:  MUL_MAT(wo, attn_out)    → PARTIAL   ← defer_attn_allreduce=true
node[i+1]:  ... attn norm, Q/K/V ...
node[i+2]:  MUL_MAT(ffn_norm_w, inpSA) → MIRRORED
node[i+3]:  MUL_MAT_ID(up_exps, ...)   → PARTIAL  ← defer_ep_allreduce=true (existing)
node[i+4]:  SILU + MUL                 → PARTIAL
node[i+5]:  MUL_MAT_ID(down_exps, ...) → PARTIAL  ← defer_ep_allreduce=false (last EP op)
node[i+6]:  ADD(wo_partial, down_partial) → PARTIAL  ← new: PARTIAL+PARTIAL
  → AllReduce fires here (combined boundary)                 ─────────────────
node[i+7]:  ADD(mirrored_allreduce_out, inpSA) → MIRRORED
node[i+8]:  ... next layer attn_norm ...
```

---

## 8. Correctness Constraints Checklist

| Constraint | Status | Notes |
|-----------|--------|-------|
| RMSNorm must not receive PARTIAL input | ✓ | `ffn_norm` operates on `inpSA` (MIRRORED) in new design |
| ADD(PARTIAL_axis0, PARTIAL_axis0) is valid | ✓ | Same axis, same chunk boundaries → element-wise ADD is safe |
| ADD(PARTIAL, MIRRORED) triggers AllReduce in meta backend | ✓ | Existing behavior: PARTIAL + MIRRORED is UNKNOWN → boundary fires |
| `ffn_norm(inpSA)` vs `ffn_norm(wo_out + inpSA)` math equivalence | ❌ OPEN | These are NOT equivalent. Requires parallel-residual architectural change OR limit to Path A (attn AllReduce preserved, only EP boundary merged) |
| EP deferral (`get_i_delayed`) still applies within MoE block | ✓ | Existing logic handles UP/GATE/DOWN EP deferral; no change needed |
| wo PARTIAL deferral across MoE block in dispatch loop | 🔲 TODO | New lookahead logic required (Section 6.1 change 3) |
| PARTIAL+PARTIAL ADD inferred as PARTIAL by `handle_generic` | ⚠️ Verify | `split_states_equal` should return true; confirm with debug flag |

---

## 9. Implementation Plan for Tasks 28–30

**Task 28 — Meta backend: PARTIAL+PARTIAL ADD + attn deferral lookahead**
- File: `ggml/src/ggml-backend-meta.cpp`
- Add `defer_attn_allreduce` lookahead in the dispatch loop (see Section 6.1, change 3)
- Validate PARTIAL+PARTIAL → PARTIAL inference with `GGML_META_DEBUG_SPLITS=1`
- Add any needed handling if `handle_generic` ASSERT fires
- Tests: Use 0.6B dense model first (no MoE), verify subgraph count drops

**Task 29 — Model graph: Change qwen3moe.cpp caller to deferred PARTIAL pattern**
- File: `src/models/qwen3moe.cpp`
- Remove the `ffn_inp = ggml_add(ctx0, cur, inpSA)` line immediately after `build_attn`
- Change `ffn_norm` to operate on `inpSA` instead of `ffn_inp`
- Add `combined_partial = ggml_add(ctx0, wo_partial, down_partial)` after MoE
- Change final residual to `cur = ggml_add(ctx0, combined_partial, inpSA)`
- ⚠️ Must decide: Path A (preserve attn AllReduce, only fuse EP) or Path B (full fusion with parallel-residual; requires numeric validation)
- Tests: Compare output logits against current implementation (same prompt, same seed); perplexity must be within float rounding tolerance OR documented as architectural change

**Task 30 — Validate and benchmark**
- Count subgraphs with `GGML_META_DEBUG_SPLITS=1`: target ~49
- Benchmark with 30B REAM model: expect +5-15% throughput from AllReduce reduction
- Verify no output corruption vs TP baseline (`bench/tests/30-ep-vs-tp-ream.sh` equivalent)
- Document actual subgraph count and timing delta in `docs/TECHNIQUES.md`

---

## 10. Open Questions for Implementer

1. **Path A vs Path B decision:** Is the numeric difference from `ffn_norm(inpSA)` vs `ffn_norm(wo_out + inpSA)` acceptable (Path B), or do we require exact math equivalence (Path A, fewer savings)?

2. **PARTIAL+PARTIAL ADD and meta backend:** Does `handle_generic` correctly handle this case, or does the ASSERT at line ~1404 fire? Run with `GGML_META_DEBUG_SPLITS=1` early to detect.

3. **Graph recording compatibility:** The SYCL graph recording (`GGML_SYCL_DISABLE_GRAPH=0`) caches subgraphs. Changing the subgraph structure invalidates the graph cache. This is expected; the new stable structure will be cached after the first run.

4. **wo_partial deferral window:** The lookahead window for `defer_attn_allreduce` must cover the entire MoE block (~60-80 nodes). The existing EP lookahead uses `k < i + 60`. The new attn lookahead may need `k < i + 200` to span the full attn→MoE distance.

---

*Written by: Orion (subagent task #27)*  
*Date: 2026-03-20*  
*For implementation: tasks #28 (meta backend), #29 (model graph), #30 (validation)*
