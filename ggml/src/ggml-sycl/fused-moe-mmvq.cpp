//
// Fused Multi-Expert MMVQ Kernel for MoE Decode Path
//
// Single kernel launch handles all active experts for a MoE layer's decode step.
// Grid dimension 0 = expert index, dimensions 1-2 = row blocks within expert.
// Each work-group computes one row of one expert's mat-vec product.
//
// Eliminates N individual kernel launches per MoE op (N = n_expert_used).
//

#include "fused-moe-mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

// ============================================================================
// Fused kernel: multi-expert mat-vec with quantized weights
// ============================================================================
//
// Grid: (n_experts, 1, block_num_y)  — or (n_experts, MMV_Y, padded) for reorder
//   - dim 0: expert index (0..n_experts-1)
//   - dim 2: row block within that expert
// Block: (1, GGML_SYCL_MMV_Y, WARP_SIZE)
//
// Each work-group computes one output row for one expert.
// Expert weight data is accessed via indirect pointers.

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot>
static void fused_moe_mul_mat_vec_q(
    const char * const * __restrict__ expert_ptrs,   // [n_experts] weight data pointers
    const void *         __restrict__ vy,             // shared q8_1 input (same for all experts)
    float * const *      __restrict__ dst_ptrs,       // [n_experts] output pointers
    const int ncols,
    const int nrows,
    const sycl::nd_item<3> & item_ct1)
{
    const int expert_idx = item_ct1.get_group(0);  // which expert in this dispatch
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const void * vx = expert_ptrs[expert_idx];
    float * dst = dst_ptrs[expert_idx];

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    static_assert(blocks_per_warp > 0);

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot(&x[ibx], &y[iby], iqs);
        }
    }

    // Sub-group reduction
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

// Reorder variant of the fused kernel
template <typename reorder_vec_dot_q_sycl>
static void fused_moe_mul_mat_vec_q_reorder(
    const char * const * __restrict__ expert_ptrs,
    const void *         __restrict__ vy,
    float * const *      __restrict__ dst_ptrs,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3> & nd_item)
{
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int expert_idx = nd_item.get_group(0);
    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_in_expert = nd_item.get_group(2);
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_in_expert * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const void * vx = expert_ptrs[expert_idx];
    float * dst = dst_ptrs[expert_idx];

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;
        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// ============================================================================
// Type-specific fused launch wrappers
// ============================================================================

static void fused_moe_q4_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                    VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_reorder_q4_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    const int block_num_y_padded = ceil_div(block_num_y, (int)num_subgroups) * num_subgroups;
    const sycl::range<3> global_size(n_experts, GGML_SYCL_MMV_Y, block_num_y_padded * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, nd_item);
            });
    });
}

static void fused_moe_q6_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                    VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_reorder_q6_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    const int block_num_y_padded = ceil_div(block_num_y, (int)num_subgroups) * num_subgroups;
    const sycl::range<3> global_size(n_experts, GGML_SYCL_MMV_Y, block_num_y_padded * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, nd_item);
            });
    });
}

static void fused_moe_q4_0_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK4_0, QI4_0, block_q4_0,
                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_reorder_q4_0_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    const int block_num_y_padded = ceil_div(block_num_y, (int)num_subgroups) * num_subgroups;
    const sycl::range<3> global_size(n_experts, GGML_SYCL_MMV_Y, block_num_y_padded * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, nd_item);
            });
    });
}

static void fused_moe_q8_0_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_q5_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                    VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_q3_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

static void fused_moe_q2_K_q8_1(
    const char * const * expert_ptrs, const void * vy, float * const * dst_ptrs,
    const int ncols, const int nrows, const int n_experts, dpct::queue_ptr stream)
{
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> grid(n_experts, 1, block_num_y);
    const sycl::range<3> block(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_moe_mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                    expert_ptrs, vy, dst_ptrs, ncols, nrows, item_ct1);
            });
    });
}

// ============================================================================
// Kernel resolver
// ============================================================================

fused_moe_kernel_fn_t get_fused_moe_kernel(ggml_type type, bool use_reorder) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return use_reorder ? fused_moe_reorder_q4_0_q8_1 : fused_moe_q4_0_q8_1;
        case GGML_TYPE_Q8_0:
            return fused_moe_q8_0_q8_1;
        case GGML_TYPE_Q2_K:
            return fused_moe_q2_K_q8_1;
        case GGML_TYPE_Q3_K:
            return fused_moe_q3_K_q8_1;
        case GGML_TYPE_Q4_K:
            return use_reorder ? fused_moe_reorder_q4_K_q8_1 : fused_moe_q4_K_q8_1;
        case GGML_TYPE_Q5_K:
            return fused_moe_q5_K_q8_1;
        case GGML_TYPE_Q6_K:
            return use_reorder ? fused_moe_reorder_q6_K_q8_1 : fused_moe_q6_K_q8_1;
        default:
            return nullptr;
    }
}
