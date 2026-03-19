//
// Fused ADD + RMSNorm kernel for SYCL.
//
// Replaces: dst_add[i] = a[i] + b[i]
//           dst_norm[i] = RMSNorm(dst_add[i])
// with a single kernel that computes both outputs in one pass.
//
// The ADD result is written to dst_add (needed as residual for next layer).
// The RMSNorm result is written to dst_norm (used as input to next matmul).
//

#include "fused-add-rmsnorm.hpp"
#include "ggml-impl.h"

#include <cstring>

// SYCL kernel: fused add + rms_norm
// Each work-group handles one row.
// Writes both sum (a+b) and normalized output.
static void fused_add_rmsnorm_kernel(
        const float * __restrict__ a,           // [ncols] per row
        const float * __restrict__ b,           // [ncols] per row
        float * __restrict__ dst_sum,           // [ncols] per row — a + b (residual)
        float * __restrict__ dst_norm,          // [ncols] per row — RMSNorm(a+b)
        const int ncols,
        const float eps,
        const sycl::nd_item<3> & item_ct1,
        float * s_sum) {

    const int row = item_ct1.get_group(2);
    const int tid = item_ct1.get_local_id(2);
    const int block_size = item_ct1.get_local_range(2);
    const int nwarps = block_size / WARP_SIZE;

    // Offset to current row
    const float * a_row = a + (int64_t)row * ncols;
    const float * b_row = b + (int64_t)row * ncols;
    float * sum_row = dst_sum + (int64_t)row * ncols;
    float * norm_row = dst_norm + (int64_t)row * ncols;

    // Pass 1: compute a+b, write to sum_row, accumulate sum-of-squares
    float ss = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float val = a_row[col] + b_row[col];
        sum_row[col] = val;
        ss += val * val;
    }

    // Reduce sum-of-squares across work-group
    ss = warp_reduce_sum(ss, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = ss;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = (nwarps + WARP_SIZE - 1) / WARP_SIZE;
        ss = 0.f;
        for (size_t i = 0; i < nreduce; i++) {
            ss += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        ss = warp_reduce_sum(ss, item_ct1);
    }

    // Pass 2: normalize and write to norm_row
    const float mean = ss / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        norm_row[col] = scale * sum_row[col];
    }
}


int try_fused_add_rmsnorm(ggml_backend_sycl_context & ctx,
                          ggml_tensor ** nodes, int n_nodes, int i) {
    // Node i must be ADD; look ahead for RMS_NORM (skipping no-ops)
    if (i + 1 >= n_nodes) return 0;

    ggml_tensor * add_node = nodes[i];
    if (add_node->op != GGML_OP_ADD) return 0;

    // Find the next compute node
    int j = i + 1;
    while (j < n_nodes) {
        ggml_tensor * n = nodes[j];
        if (ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
            n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE ||
            (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            j++;
            continue;
        }
        break;
    }
    if (j >= n_nodes) return 0;

    ggml_tensor * norm_node = nodes[j];
    if (norm_node->op != GGML_OP_RMS_NORM) return 0;

    // The RMS_NORM must consume the ADD's output
    if (norm_node->src[0] != add_node) return 0;

    // Both must be f32
    if (add_node->type != GGML_TYPE_F32) return 0;
    if (norm_node->type != GGML_TYPE_F32) return 0;
    if (add_node->src[0]->type != GGML_TYPE_F32) return 0;
    if (add_node->src[1]->type != GGML_TYPE_F32) return 0;

    // Both inputs to ADD must be contiguous
    if (!ggml_is_contiguous(add_node->src[0])) return 0;
    if (!ggml_is_contiguous(add_node->src[1])) return 0;
    if (!ggml_is_contiguous(add_node)) return 0;
    if (!ggml_is_contiguous(norm_node)) return 0;

    // ADD src0 and src1 must have same shape (element-wise add, not broadcast)
    const ggml_tensor * a = add_node->src[0];
    const ggml_tensor * b = add_node->src[1];
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] ||
        a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3]) return 0;

    // Strides must match for simple row-major layout
    const int64_t ncols = a->ne[0];
    const int64_t nrows = ggml_nrows(a);

    if (ncols == 0 || nrows == 0) return 0;

    // Get eps from RMS_NORM op_params
    float eps;
    memcpy(&eps, norm_node->op_params, sizeof(float));

    // Get data pointers
    const float * a_data = (const float *)a->data;
    const float * b_data = (const float *)b->data;
    float * sum_data = (float *)add_node->data;
    float * norm_data = (float *)norm_node->data;

    if (!a_data || !b_data || !sum_data || !norm_data) return 0;

    // Launch the fused kernel
    queue_ptr stream = ctx.stream();
    const int device = ctx.device;

    const sycl::range<3> global_dims(1, 1, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fused_add_rmsnorm_kernel(a_data, b_data, sum_data, norm_data,
                                            ncols, eps, item_ct1, nullptr);
                });
        });
    } else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc(
                sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    fused_add_rmsnorm_kernel(a_data, b_data, sum_data, norm_data,
                                            ncols, eps, item_ct1,
                                            s_sum_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }

    GGML_SYCL_DEBUG("[SYCL] fused ADD+RMSNorm: ncols=%ld, nrows=%ld, eps=%g\n",
                    (long)ncols, (long)nrows, eps);

    return j - i + 1;  // consumed nodes from ADD to RMS_NORM (inclusive, including no-ops)
}
