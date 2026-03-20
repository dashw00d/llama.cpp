#include "models.h"

llm_build_qwen3moe::llm_build_qwen3moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        // Deferred PARTIAL fusion (Option B / Path B):
        // Keep wo_partial as PARTIAL — do NOT add inpSA here. The meta backend lookahead
        // (defer_attn_allreduce) will suppress the AllReduce boundary at wo_partial and
        // defer it past the MoE block where both PARTIAL tensors are combined.
        ggml_tensor * wo_partial = cur;  // PARTIAL (axis 0 from wo MUL_MAT SPLIT_AXIS_0)

        if (il == n_layer - 1 && inp_out_ids) {
            wo_partial = ggml_get_rows(ctx0, wo_partial, inp_out_ids);
            inpSA      = ggml_get_rows(ctx0,      inpSA, inp_out_ids);
        }

        // MoE branch: ffn_norm operates on inpSA (MIRRORED) — not on the wo output.
        // Note: this is a mathematical change (Path B parallel-residual structure).
        // ffn_norm(inpSA) != ffn_norm(wo_out + inpSA); numeric validation required (task 31).
        cur = build_norm(inpSA,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
        cb(moe_out, "ffn_moe_out", il);

        // Combined PARTIAL+PARTIAL ADD: both wo_partial and moe_out are PARTIAL on axis 0.
        // The meta backend defer_attn_allreduce lookahead defers the wo boundary to here.
        // AllReduce fires at this ADD (combined boundary) — halves AllReduce count per layer.
        ggml_tensor * combined_partial = ggml_add(ctx0, wo_partial, moe_out);
        cb(combined_partial, "combined_partial", il);

        // Final residual: combined_partial triggers AllReduce → MIRRORED, then + inpSA.
        cur = ggml_add(ctx0, combined_partial, inpSA);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
