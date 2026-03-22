// SYCL port of ggml-cuda/topk-moe.cu
// Fused softmax→top-k→get_rows for MoE expert selection.
// Key difference from CUDA: WARP_SIZE=16 on Arc A770 (vs 32 on NVIDIA).

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "topk-moe.hpp"
#include <cmath>
#include <cfloat>

struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax: each thread holds experts_per_thread values.
template <int experts_per_thread, bool use_limit>
static inline void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = sycl::max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = sycl::native::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum<WARP_SIZE>(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
static inline void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::native::exp(-vals[i])) : -INFINITY;
    }
}

// Main fused topk-moe kernel.
// Grid: ((n_rows+3)/4, 1, 1), Block: (WARP_SIZE, 4, 1)
// Each warp processes one row. 4 rows per block.
template <int n_experts, bool has_bias>
void topk_moe_sycl_kernel(const float *         logits,
                           float *               weights,
                           int32_t *             ids,
                           float *               bias,
                           const int             n_rows,
                           const int             n_expert_used,
                           const float           clamp_val,
                           const float           scale_val,
                           const topk_moe_config config,
                           sycl::nd_item<3>      item) {
    const int row = item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= n_rows) {
        return;
    }

    const int lane = item.get_local_id(2);

    logits  += n_experts * row;
    weights += n_expert_used * row;
    ids     += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } else {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // NaN sanitize
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // Bias for expert selection (e.g. suppress padded fake experts)
    float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + lane;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    // Iterative top-k via warp-cooperative argmax
    float wt_sum = 0.f;
    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = lane + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = dpct::permute_sub_group_by_xor(sg, max_val,   mask, WARP_SIZE);
                const float val_s  = dpct::permute_sub_group_by_xor(sg, max_val_s, mask, WARP_SIZE);
                const int   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = lane + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = dpct::permute_sub_group_by_xor(sg, max_val,    mask, WARP_SIZE);
                const int   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum              = sycl::max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Launch wrapper with template switch on n_experts.
template<bool has_bias>
static void launch_topk_moe_sycl(ggml_backend_sycl_context & ctx,
                                  const float *               logits,
                                  float *                     weights,
                                  int32_t *                   ids,
                                  float *                     bias,
                                  const int                   n_rows,
                                  const int                   n_expert,
                                  const int                   n_expert_used,
                                  const float                 clamp_val,
                                  const float                 scale_val,
                                  const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");

    const int rows_per_block = 4;
    sycl::range<3> grid(
        (n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    sycl::range<3> block(1, rows_per_block, WARP_SIZE);

    dpct::queue_ptr stream = ctx.stream();

    auto launch = [&]<int NE>() {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    topk_moe_sycl_kernel<NE, has_bias>(
                        logits, weights, ids, bias, n_rows, n_expert_used,
                        clamp_val, scale_val, config, item);
                });
        });
    };

    switch (n_expert) {
        case 1:   launch.template operator()<1>();   break;
        case 2:   launch.template operator()<2>();   break;
        case 4:   launch.template operator()<4>();   break;
        case 8:   launch.template operator()<8>();   break;
        case 16:  launch.template operator()<16>();  break;
        case 32:  launch.template operator()<32>();  break;
        case 64:  launch.template operator()<64>();  break;
        case 128: launch.template operator()<128>(); break;
        case 256: launch.template operator()<256>(); break;
        default:
            GGML_ABORT("topk-moe: unsupported n_expert %d", n_expert);
            break;
    }
}

void ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                            const ggml_tensor *             logits,
                            ggml_tensor *                   weights,
                            ggml_tensor *                   ids,
                            const ggml_tensor *             clamp,
                            const ggml_tensor *             scale,
                            const ggml_tensor *             bias,
                            const ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_sycl<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_sycl<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op,
                                    const ggml_tensor * weights,
                                    const ggml_tensor * logits,
                                    const ggml_tensor * ids) {
    const int n_expert = ids->nb[1] / ids->nb[0];
    // Must be power of 2 and <= 256 (Arc WARP_SIZE=16, 512 experts = 32 regs/thread, risky)
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 256) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax = gating_op;
        float scale    = 1.0f;
        float max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }
        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);
        if (op != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
    }

    return true;
}
