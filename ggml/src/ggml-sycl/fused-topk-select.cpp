//
// Fused softmax + top-k selection for MoE expert routing.
//
// Replaces: SOFT_MAX(logits) → ARGSORT(probs) → GET_ROWS(probs, top_k_ids)
// with a single kernel that computes softmax and selects top-k in one pass.
//
// For small n_expert (≤128) and small k (≤8), this is trivially parallelizable.
//

#include "fused-topk-select.hpp"
#include "ggml-impl.h"

#include <cstring>

int try_fused_topk_select(ggml_backend_sycl_context & ctx,
                          ggml_tensor ** nodes, int n_nodes, int i) {
    // Node i must be SOFT_MAX
    if (nodes[i]->op != GGML_OP_SOFT_MAX) return 0;
    ggml_tensor * softmax_node = nodes[i];

    // Softmax input must be f32, output f32, no mask
    if (softmax_node->type != GGML_TYPE_F32) return 0;
    if (softmax_node->src[0]->type != GGML_TYPE_F32) return 0;
    if (softmax_node->src[1] != nullptr) return 0;  // mask not supported in fused kernel

    // Look ahead for ARGSORT (skipping RESHAPE/VIEW no-ops)
    int j = i + 1;
    ggml_tensor * argsort_node = nullptr;
    while (j < n_nodes) {
        if (nodes[j]->op == GGML_OP_RESHAPE || nodes[j]->op == GGML_OP_VIEW ||
            nodes[j]->op == GGML_OP_PERMUTE || nodes[j]->op == GGML_OP_NONE) {
            j++;
            continue;
        }
        if (nodes[j]->op == GGML_OP_ARGSORT) {
            argsort_node = nodes[j];
            break;
        }
        return 0;  // unexpected op
    }
    if (!argsort_node) return 0;

    // ARGSORT must consume the softmax output (possibly through reshape)
    // Check that argsort's src[0] traces back to softmax_node
    ggml_tensor * argsort_src = argsort_node->src[0];
    while (argsort_src && (argsort_src->op == GGML_OP_RESHAPE || argsort_src->op == GGML_OP_VIEW)) {
        argsort_src = argsort_src->src[0];
    }
    if (argsort_src != softmax_node) return 0;

    // ARGSORT must be descending
    enum ggml_sort_order order = (enum ggml_sort_order)argsort_node->op_params[0];
    if (order != GGML_SORT_ORDER_DESC) return 0;

    // Look ahead for GET_ROWS (skipping VIEW no-ops)
    int k_idx = j + 1;
    ggml_tensor * getrows_node = nullptr;
    while (k_idx < n_nodes) {
        if (nodes[k_idx]->op == GGML_OP_RESHAPE || nodes[k_idx]->op == GGML_OP_VIEW ||
            nodes[k_idx]->op == GGML_OP_PERMUTE || nodes[k_idx]->op == GGML_OP_NONE) {
            k_idx++;
            continue;
        }
        if (nodes[k_idx]->op == GGML_OP_GET_ROWS) {
            getrows_node = nodes[k_idx];
            break;
        }
        return 0;
    }
    if (!getrows_node) return 0;

    // GET_ROWS src[1] must be indices from argsort (possibly through view)
    // GET_ROWS src[0] must be the softmax probabilities (possibly through reshape)
    ggml_tensor * getrows_src0 = getrows_node->src[0];  // probs data
    ggml_tensor * getrows_src1 = getrows_node->src[1];  // indices

    // Trace src1 back through views to argsort
    ggml_tensor * idx_src = getrows_src1;
    while (idx_src && (idx_src->op == GGML_OP_RESHAPE || idx_src->op == GGML_OP_VIEW)) {
        idx_src = idx_src->src[0];
    }
    if (idx_src != argsort_node) return 0;

    // Trace src0 back through reshapes to softmax
    ggml_tensor * probs_src = getrows_src0;
    while (probs_src && (probs_src->op == GGML_OP_RESHAPE || probs_src->op == GGML_OP_VIEW)) {
        probs_src = probs_src->src[0];
    }
    if (probs_src != softmax_node) return 0;

    // Extract dimensions
    const int64_t n_expert = softmax_node->src[0]->ne[0];  // number of experts
    const int64_t nrows = softmax_node->src[0]->ne[1];      // number of tokens (usually 1 for decode)

    // GET_ROWS output ne[1] gives us k (top-k count)
    const int64_t top_k = getrows_node->ne[1];

    // Sanity checks
    if (n_expert > 128 || n_expert < 2) return 0;  // kernel uses stack array[128]
    if (top_k > 32 || top_k < 1) return 0;
    if (nrows < 1) return 0;

    // Get softmax scale and max_bias
    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (const float *)softmax_node->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)softmax_node->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return 0;  // ALiBi not supported in fused kernel

    // Get data pointers
    const float * logits = (const float *)softmax_node->src[0]->data;
    float * softmax_out = (float *)softmax_node->data;
    int32_t * argsort_out = (int32_t *)argsort_node->data;
    float * getrows_out = (float *)getrows_node->data;

    if (!logits || !softmax_out || !argsort_out || !getrows_out) return 0;

    // We also need to write the softmax output (other ops might read it)
    // Actually, let's check: does anything else read the softmax output besides
    // the reshape→argsort and reshape→get_rows? If so, we still need it.
    // For safety, we'll write the full softmax output AND the argsort+getrows outputs.
    // This way all downstream consumers see correct data.

    queue_ptr stream = ctx.stream();

    // Launch: 1 work-item per row (n_expert is small, no parallelism needed within row)
    const int64_t n_wg = nrows;
    const int64_t wg_size = 1;

    stream->parallel_for(
        sycl::nd_range<1>(n_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            const int row = item.get_group(0);

            // --- Inline softmax + top-k ---
            const float * row_logits = logits + (int64_t)row * n_expert;
            float * row_softmax = softmax_out + (int64_t)row * n_expert;
            int32_t * row_sorted = argsort_out + (int64_t)row * n_expert;
            float * row_weights = getrows_out + (int64_t)row * top_k;

            // Find max
            float max_val = -INFINITY;
            for (int64_t e = 0; e < n_expert; e++) {
                float v = row_logits[e] * scale;
                if (v > max_val) max_val = v;
            }

            // Compute softmax
            float probs[128];
            float sum_exp = 0.0f;
            for (int64_t e = 0; e < n_expert; e++) {
                float ex = sycl::native::exp(row_logits[e] * scale - max_val);
                probs[e] = ex;
                sum_exp += ex;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int64_t e = 0; e < n_expert; e++) {
                probs[e] *= inv_sum;
                row_softmax[e] = probs[e];  // write softmax output
            }

            // Partial selection sort (descending) — only find top-k
            // The VIEW of argsort output only reads first k entries,
            // so we don't need to sort all n_expert indices.
            int32_t indices[128];
            for (int64_t e = 0; e < n_expert; e++) {
                indices[e] = (int32_t)e;
            }
            for (int64_t s = 0; s < top_k && s < n_expert; s++) {
                int64_t max_idx = s;
                float max_p = probs[indices[s]];
                for (int64_t l = s + 1; l < n_expert; l++) {
                    if (probs[indices[l]] > max_p) {
                        max_p = probs[indices[l]];
                        max_idx = l;
                    }
                }
                int32_t tmp = indices[s];
                indices[s] = indices[max_idx];
                indices[max_idx] = tmp;
            }

            // Write top-k sorted indices to argsort output buffer
            for (int64_t t = 0; t < top_k; t++) {
                row_sorted[t] = indices[t];
            }

            // Write top-k weights to GET_ROWS output buffer
            for (int64_t t = 0; t < top_k; t++) {
                row_weights[t] = probs[indices[t]];
            }
        }
    );

    GGML_SYCL_DEBUG("[SYCL] fused top-k select: n_expert=%ld, top_k=%ld, nrows=%ld, consumed %d nodes\n",
                    (long)n_expert, (long)top_k, (long)nrows, k_idx - i + 1);

    // Return number of nodes consumed (from SOFT_MAX to GET_ROWS inclusive,
    // including intermediate RESHAPE/VIEW no-ops)
    return k_idx - i + 1;
}
