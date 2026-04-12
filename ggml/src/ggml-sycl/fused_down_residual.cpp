// iter24: fused Q4K down-projection + residual ADD kernel.
// See fused_down_residual.hpp for the contract.
//
// Implementation pattern: cooperative-warp Q4K matmul (SG=16, one
// subgroup per output row, threads stride the K-dim blocks, sub-group
// reduce, thread 0 of each subgroup adds the residual and writes).
// Math is character-for-character the same as iter15's
// `run_q4k_scalar_cooperative` from mmvq.cpp -- the only addition is
// the inline residual read and add at the writeback.

#include "fused_down_residual.hpp"

#include <cstdint>
#include <sycl/sycl.hpp>

// We need block_q4_K from ggml-common.h. Bring in the same defines
// the rest of ggml-sycl uses.
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

void ggml_sycl_fused_down_residual_dispatch(
    sycl::queue & queue,
    const ggml_sycl_fused_down_residual_args & args) {

    if (args.vx == nullptr || args.x == nullptr || args.dst == nullptr) return;
    if (args.residual == nullptr) return;
    if (args.ncols_x <= 0 || args.n_rows_out <= 0 || args.n_cols <= 0) return;
    if ((args.ncols_x % 256) != 0) return;  // Q4K block size is 256

    constexpr int SG = 16;
    const block_q4_K * vx_blocks = static_cast<const block_q4_K *>(args.vx);
    const int n_blocks_per_row = args.ncols_x / 256;

    // Capture-by-value into the SYCL kernel.
    const int     nr           = args.n_rows_out;
    const int     nc           = args.n_cols;
    const std::size_t x_stride       = args.x_col_stride;
    const std::size_t y_stride       = args.dst_col_stride;
    const std::size_t r_stride       = args.residual_col_stride;
    const float * x_base       = args.x;
    const float * r_base       = args.residual;
    float *       y_base       = args.dst;
    const block_q4_K * blocks  = vx_blocks;

    queue.submit([=](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<std::size_t>(nr) * SG),
                sycl::range<1>(SG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int row = static_cast<int>(it.get_group(0));
                const int lid = static_cast<int>(it.get_local_linear_id());
                if (row >= nr) {
                    return;
                }

                auto fp16_to_fp32 = [](std::uint16_t h) -> float {
                    std::uint32_t s = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
                    std::uint32_t e = (h >> 10) & 0x1fU;
                    std::uint32_t m = h & 0x03ffU;
                    if (e == 0x1fU) return 0.0f;
                    if (e == 0) {
                        if (m == 0) return 0.0f;
                        float val = static_cast<float>(m) * (1.0f/1024.0f) * (1.0f/16384.0f);
                        return (h & 0x8000U) ? -val : val;
                    }
                    return sycl::bit_cast<float>(s | ((e + 112U) << 23) | (m << 13));
                };

                float acc[32] = {};  // up to ncols_y == 32 (iter22 widening)
                // Strided block loop: each of SG threads handles a
                // disjoint subset of the n_blocks_per_row blocks.
                for (int ib = lid; ib < n_blocks_per_row; ib += SG) {
                    const block_q4_K & block =
                        blocks[static_cast<std::size_t>(row) * n_blocks_per_row + ib];
                    const std::uint8_t * bytes =
                        reinterpret_cast<const std::uint8_t *>(&block);
                    const std::uint16_t d_raw  = bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8);
                    const std::uint16_t dm_raw = bytes[2] | (static_cast<std::uint16_t>(bytes[3]) << 8);
                    const float d_val    = fp16_to_fp32(d_raw);
                    const float neg_dmin = -fp16_to_fp32(dm_raw);
                    const std::uint8_t * sc = bytes + 4;

                    float scales[8];
                    float biases[8];
                    for (int i = 0; i < 4; ++i) {
                        const std::uint8_t a = sc[0 + i];
                        const std::uint8_t b = sc[4 + i];
                        const std::uint8_t c = sc[8 + i];
                        scales[i]     = d_val    * static_cast<float>(a & 0x3FU);
                        scales[i + 4] = d_val    * static_cast<float>((c & 0x0FU) | ((a >> 2) & 0x30U));
                        biases[i]     = neg_dmin * static_cast<float>(b & 0x3FU);
                        biases[i + 4] = neg_dmin * static_cast<float>((c >> 4) | ((b >> 2) & 0x30U));
                    }

                    const std::uint8_t * qs = block.qs;
                    for (int pair = 0; pair < 4; ++pair) {
                        const float s_lo = scales[pair * 2];
                        const float b_lo = biases[pair * 2];
                        const float s_hi = scales[pair * 2 + 1];
                        const float b_hi = biases[pair * 2 + 1];
                        const std::uint8_t * pair_bytes = qs + pair * 32;
                        for (int j = 0; j < 32; ++j) {
                            const std::uint8_t packed = pair_bytes[j];
                            const float w_lo = s_lo * static_cast<float>(packed & 0x0FU) + b_lo;
                            const float w_hi = s_hi * static_cast<float>(packed >> 4) + b_hi;
                            const int k_lo = ib * 256 + pair * 64 + j;
                            const int k_hi = ib * 256 + pair * 64 + 32 + j;
                            for (int c = 0; c < nc; ++c) {
                                acc[c] += w_lo * x_base[c * x_stride + k_lo];
                                acc[c] += w_hi * x_base[c * x_stride + k_hi];
                            }
                        }
                    }
                }

                // Subgroup reduction across the 16 threads sharing this row.
                for (int c = 0; c < nc; ++c) {
                    acc[c] = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
                }

                // Thread 0 of each subgroup writes dst[c, row] =
                // acc[c] + residual[c, row].
                if (lid == 0) {
                    for (int c = 0; c < nc; ++c) {
                        const float r = r_base[c * r_stride + row];
                        y_base[c * y_stride + row] = acc[c] + r;
                    }
                }
            });
    });
}
