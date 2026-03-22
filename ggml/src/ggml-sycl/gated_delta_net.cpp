#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>

// Direct 1:1 port of the CUDA kernel.
// One thread per column. Each thread holds the full state column in registers.
// No warp reductions — each thread independently computes its column's contribution,
// then a block-level reduction sums across threads for the attention score.

template <int S_v, bool KDA>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           n_seqs,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          int64_t           rq1,
                          int64_t           rq3,
                          float             scale,
                          sycl::nd_item<3>  item_ct1,
                          float *           shared_attn) {
    const int64_t h_idx    = item_ct1.get_group(0);
    const int64_t sequence = item_ct1.get_group(1);
    const int     col      = item_ct1.get_local_id(2);  // each thread owns one column

    const int64_t iq1 = h_idx / rq1;
    const int64_t iq3 = sequence / rq3;

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    // Load state column into registers
    float s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i * S_v + col];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                kv_col += s[i] * k_t[i];
            }

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                s[i] = g_val * s[i] + k_t[i] * delta_col;
                attn_col += s[i] * q_t[i];
            }

            attn_data[col] = attn_col * scale;
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                kv_col += sycl::native::exp(g_t[i]) * s[i] * k_t[i];
            }

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                s[i] = sycl::native::exp(g_t[i]) * s[i] + k_t[i] * delta_col;
                attn_col += s[i] * q_t[i];
            }

            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        state[i * S_v + col] = s[i];
    }
}

template <bool KDA>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         rq1,
                                   int64_t         rq3,
                                   float           scale,
                                   dpct::queue_ptr stream) {
    // Grid: one block per (head, sequence). Block: S_v threads (one per column).
    sycl::range<3> grid_dims(H, n_seqs, 1);
    sycl::range<3> block_dims(1, 1, S_v);

    auto launch = [&]<int sv>() {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shared_attn(sycl::range<1>(sv), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    gated_delta_net_sycl<sv, KDA>(
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
                        H, n_tokens, n_seqs,
                        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                        rq1, rq3, scale, item_ct1,
                        shared_attn.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    };

    switch (S_v) {
        case 16:  launch.template operator()<16>();  break;
        case 32:  launch.template operator()<32>();  break;
        case 64:  launch.template operator()<64>();  break;
        case 128: launch.template operator()<128>(); break;
        default:
            GGML_ABORT("fatal error: unsupported S_v");
            break;
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == src_k->ne[1]);

    const int64_t rq1 = nev1 / neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, rq1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, rq1, rq3, scale, stream);
    }
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}
