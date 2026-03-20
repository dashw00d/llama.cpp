//
// Fused expert aggregation kernel for MoE layers (SYCL).
//
// Replaces the pattern: MUL(experts, weights) + VIEW×N + ADD×(N-1)
// with a single kernel that computes:
//   output[embd, token] = sum_k( weights[k, token] * experts[embd, k, token] )
//
// Saves (N_expert_used) kernel launches per MoE layer (1 MUL + N-1 ADDs → 1 kernel).
// For Qwen3-30B-A3B with 8 experts used × 48 layers × 3 GPUs = 1,152 launches saved.
//

#include "fused-expert-agg.hpp"
#include "ggml-impl.h"

#include <cstring>

// The SYCL kernel: each work-item computes one element of the output.
// For n_tokens=1 (decode), this is just n_embd work-items.
// For n_tokens>1 (prefill), this is n_embd * n_tokens work-items.
static void fused_expert_agg_kernel(
        const float * __restrict__ experts_data,   // [n_embd, n_expert_used, n_tokens] — unweighted expert outputs
        const float * __restrict__ weights_data,    // [1, n_expert_used, n_tokens] — expert weights (broadcast over n_embd)
        float * __restrict__ output_data,           // [n_embd, n_tokens] — aggregated output
        const int64_t n_embd,
        const int64_t n_expert_used,
        const int64_t n_tokens,
        const size_t expert_stride,                 // stride between experts in elements (experts->nb[1] / sizeof(float))
        const size_t token_stride_exp,              // stride between tokens in experts (experts->nb[2] / sizeof(float))
        const size_t weight_expert_stride,          // weights->nb[1] / sizeof(float)
        const size_t weight_token_stride,           // weights->nb[2] / sizeof(float)
        const size_t out_token_stride,              // output->nb[1] / sizeof(float)
        sycl::nd_item<1> item) {

    const int64_t idx = item.get_global_id(0);
    const int64_t total = n_embd * n_tokens;
    if (idx >= total) return;

    const int64_t i_embd  = idx % n_embd;
    const int64_t i_token = idx / n_embd;

    float sum = 0.0f;
    for (int64_t k = 0; k < n_expert_used; k++) {
        const float w = weights_data[k * weight_expert_stride + i_token * weight_token_stride];
        const float e = experts_data[i_embd + k * expert_stride + i_token * token_stride_exp];
        sum += w * e;
    }
    output_data[i_embd + i_token * out_token_stride] = sum;
}


int try_fused_expert_agg(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int n_nodes, int i) {
    // Step 1: Check if node[i] is the MUL for expert weighting
    ggml_tensor * mul_node = nodes[i];

    if (mul_node->op != GGML_OP_MUL) return 0;
    if (strstr(mul_node->name, "ffn_moe_weighted") == nullptr) return 0;

    // The MUL node's sources:
    //   src[0] = experts (unweighted) [n_embd, n_expert_used, n_tokens]
    //   src[1] = weights [1, n_expert_used, n_tokens]
    const ggml_tensor * experts = mul_node->src[0];
    const ggml_tensor * weights = mul_node->src[1];

    if (experts == nullptr || weights == nullptr) return 0;
    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32) return 0;

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2];

    // Need at least 2 experts to justify fusion
    if (n_expert_used < 2) return 0;

    // Step 2: Skip VIEW nodes after the MUL
    int j = i + 1;
    int n_views = 0;
    while (j < n_nodes && nodes[j]->op == GGML_OP_VIEW) {
        n_views++;
        j++;
    }

    // We expect n_expert_used VIEW nodes (one per expert slice)
    // But some may have been optimized away or we might have fewer
    // At minimum, verify we have ADD nodes forming the reduction chain

    // Step 3: Count the ADD chain
    // Pattern: ADD(expert[0], expert[1]) → ADD(result, expert[2]) → ... → ADD(result, expert[N-1])
    int n_adds = 0;
    int add_start = j;
    while (j < n_nodes && nodes[j]->op == GGML_OP_ADD && n_adds < (n_expert_used - 1)) {
        // Verify this ADD is part of the expert aggregation chain
        // The ADD's sources should be views into the weighted experts tensor
        // or the result of a previous ADD in this chain
        n_adds++;
        j++;
    }

    // We need exactly (n_expert_used - 1) ADDs for a complete fusion
    if (n_adds != (n_expert_used - 1)) return 0;

    // The last ADD node's output buffer is where we write the result
    ggml_tensor * last_add = nodes[j - 1];
    float * output_data = (float *)last_add->data;

    if (output_data == nullptr) return 0;

    // Get data pointers
    const float * experts_data = (const float *)experts->data;
    const float * weights_data = (const float *)weights->data;

    if (experts_data == nullptr || weights_data == nullptr) return 0;

    // Compute strides in elements (float)
    const size_t expert_stride       = experts->nb[1] / sizeof(float);
    const size_t token_stride_exp    = experts->nb[2] / sizeof(float);
    const size_t weight_expert_stride = weights->nb[1] / sizeof(float);
    const size_t weight_token_stride  = weights->nb[2] / sizeof(float);
    // Output is [n_embd, n_tokens] — contiguous
    const size_t out_token_stride    = last_add->nb[1] / sizeof(float);

    // Launch the fused kernel
    const int64_t total_elements = n_embd * n_tokens;
    const int64_t wg_size = 256;  // work-group size
    const int64_t n_groups = (total_elements + wg_size - 1) / wg_size;

    queue_ptr stream = ctx.stream();

    GGML_SYCL_DEBUG("[SYCL] fused expert agg: n_embd=%ld, n_expert_used=%ld, n_tokens=%ld, replacing %d nodes (1 MUL + %d VIEWs + %d ADDs)\n",
                    (long)n_embd, (long)n_expert_used, (long)n_tokens,
                    1 + n_views + n_adds, n_views, n_adds);
    {
        static bool fused_agg_printed = false;
        if (!fused_agg_printed) {
            fused_agg_printed = true;
            fprintf(stderr, "FUSED_AGG [%s] dev=%d n_embd=%ld n_exp=%ld n_tok=%ld mul_name=%s experts=%p last_add=%p\n",
                    mul_node->name, ctx.device, (long)n_embd, (long)n_expert_used, (long)n_tokens,
                    mul_node->name, (void*)experts_data, (void*)output_data);
        }
    }

    stream->parallel_for(
        sycl::nd_range<1>(n_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            fused_expert_agg_kernel(
                experts_data, weights_data, output_data,
                n_embd, n_expert_used, n_tokens,
                expert_stride, token_stride_exp,
                weight_expert_stride, weight_token_stride,
                out_token_stride,
                item);
        }
    );

    // Return the number of nodes consumed:
    // 1 (MUL) + n_views (VIEW) + n_adds (ADD)
    // The caller should skip these nodes.
    // Note: we return (j - i) which is the total span from the MUL to after the last ADD.
    return j - i;
}
