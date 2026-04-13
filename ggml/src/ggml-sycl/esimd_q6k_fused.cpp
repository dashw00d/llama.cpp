// iter27: ESIMD Q6_K multi-column fused matmul kernel.
// Ported from gemma4-ipex/src/esimd/linear_forward_q6k_sycl.cpp.
// See esimd_q6k_fused.hpp for the contract.

#include "esimd_q6k_fused.hpp"

#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace {

namespace esimd_ns = sycl::ext::intel::esimd;

// Q6K block layout constants (ggml AoS format, 210 bytes per block)
constexpr int kQ6KBlockBytes = 210;
constexpr int kQ6KQlOffset   = 0;    // 128 bytes: 4-bit low quantized values
constexpr int kQ6KQhOffset   = 128;  // 64 bytes:  2-bit high quantized values
constexpr int kQ6KScOffset   = 192;  // 16 bytes:  int8 per-16-value scales
constexpr int kQ6KDOffset    = 208;  // 2 bytes:   fp16 delta

// Per-field sizes for SoA layout.
constexpr int kQ6KQlBytes    = 128;
constexpr int kQ6KQhBytes    = 64;
constexpr int kQ6KScaleBytes = 16;
constexpr int kQ6KDBytes     = 2;

inline float esimd_q6k_fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    if (exp == 0x1fU) return 0.0f;
    if (exp == 0) {
        // Subnormal fp16: value = (-1)^sign * 2^-14 * (mant/1024)
        if (mant == 0) return 0.0f;
        float val = static_cast<float>(mant) * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000U) ? -val : val;
    }
    return sycl::bit_cast<float>(sign | ((exp + 112U) << 23) | (mant << 13));
}

} // anonymous namespace

void ggml_sycl_esimd_q6k_fused_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_q6k_fused_args & args) {

    if (args.n_rows <= 0 || args.n_blocks_per_row <= 0 || args.n_cols <= 0) {
        return;
    }

    const std::size_t n_rows = static_cast<std::size_t>(args.n_rows);

    // Detect AoS vs SoA: if ql_base is null, use raw_blocks in AoS mode.
    const bool use_aos = (args.ql_base == nullptr && args.raw_blocks != nullptr);

    queue.submit([=](sycl::handler & cgh) {
        const auto raw_blocks = args.raw_blocks;
        const auto ql_base    = args.ql_base;
        const auto qh_base    = args.qh_base;
        const auto sc_base    = args.sc_base;
        const auto d_base     = args.d_base;
        const auto nbpr       = args.n_blocks_per_row;
        const auto x          = args.x;
        const auto y          = args.y;
        const auto nr         = args.n_rows;
        const auto nc         = args.n_cols;
        const auto x_stride   = args.x_col_stride;
        const auto y_stride   = args.y_col_stride;
        const auto res        = args.residual;
        const auto res_stride = args.residual_col_stride;
        const auto aos        = use_aos;

        cgh.parallel_for(
            sycl::range<1>(n_rows),
            [=](sycl::id<1> row_id) SYCL_ESIMD_KERNEL {
                const int row = static_cast<int>(row_id[0]);
                if (row >= nr) return;

                float acc[32] = {};  // iter22: widened from [8] for npl <= 32

                for (int ib = 0; ib < nbpr; ++ib) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(nbpr) +
                        static_cast<std::size_t>(ib);

                    const std::uint8_t * ql;
                    const std::uint8_t * qh;
                    const std::int8_t  * sc;
                    float d_val;

                    if (aos) {
                        // AoS: 210-byte blocks, ql@0, qh@128, sc@192, d@208
                        const std::uint8_t * blk = raw_blocks + block_idx * kQ6KBlockBytes;
                        ql = blk + kQ6KQlOffset;
                        qh = blk + kQ6KQhOffset;
                        sc = reinterpret_cast<const std::int8_t *>(blk + kQ6KScOffset);
                        const std::uint16_t d_raw = blk[kQ6KDOffset] |
                            (static_cast<std::uint16_t>(blk[kQ6KDOffset + 1]) << 8);
                        d_val = esimd_q6k_fp16_to_float(d_raw);
                    } else {
                        // SoA: each field stored contiguously for ALL blocks
                        ql = ql_base + block_idx * kQ6KQlBytes;
                        qh = qh_base + block_idx * kQ6KQhBytes;
                        sc = reinterpret_cast<const std::int8_t *>(
                            sc_base + block_idx * kQ6KScaleBytes);
                        const std::uint16_t d_raw =
                            d_base[block_idx * kQ6KDBytes] |
                            (static_cast<std::uint16_t>(d_base[block_idx * kQ6KDBytes + 1]) << 8);
                        d_val = esimd_q6k_fp16_to_float(d_raw);
                    }

                    // Process 2 halves × 2 segments = 4 iterations,
                    // 64 values each = 256 total per block.
                    for (int half = 0; half < 2; ++half) {
                        const std::uint8_t * ql_h = ql + half * 64;
                        const std::uint8_t * qh_h = qh + half * 32;

                        for (int seg = 0; seg < 2; ++seg) {
                            const int base = seg * 16;
                            const float s0 = d_val * static_cast<float>(sc[half * 8 + seg + 0]);
                            const float s2 = d_val * static_cast<float>(sc[half * 8 + seg + 2]);
                            const float s4 = d_val * static_cast<float>(sc[half * 8 + seg + 4]);
                            const float s6 = d_val * static_cast<float>(sc[half * 8 + seg + 6]);

                            esimd_ns::simd<std::uint8_t, 16> ql0_raw =
                                esimd_ns::block_load<std::uint8_t, 16>(
                                    const_cast<std::uint8_t *>(ql_h + base));
                            esimd_ns::simd<std::uint8_t, 16> ql1_raw =
                                esimd_ns::block_load<std::uint8_t, 16>(
                                    const_cast<std::uint8_t *>(ql_h + base + 32));
                            esimd_ns::simd<std::uint8_t, 16> qh0_raw =
                                esimd_ns::block_load<std::uint8_t, 16>(
                                    const_cast<std::uint8_t *>(qh_h + base));
                            esimd_ns::simd<std::uint32_t, 16> ql0_u32 = esimd_ns::convert<std::uint32_t>(ql0_raw);
                            esimd_ns::simd<std::uint32_t, 16> ql1_u32 = esimd_ns::convert<std::uint32_t>(ql1_raw);
                            esimd_ns::simd<std::uint32_t, 16> qh0_u32 = esimd_ns::convert<std::uint32_t>(qh0_raw);

                            // 6-bit extraction + dequant. Convert to float BEFORE
                            // subtracting 32 to avoid uint32 wraparound.
                            esimd_ns::simd<float, 16> w1 = (esimd_ns::convert<float>(
                                (ql0_u32 & 0x0FU) | ((qh0_u32 & 0x03U) << 4)) - 32.0f) * s0;
                            esimd_ns::simd<float, 16> w2 = (esimd_ns::convert<float>(
                                (ql1_u32 & 0x0FU) | (((qh0_u32 >> 2) & 0x03U) << 4)) - 32.0f) * s2;
                            esimd_ns::simd<float, 16> w3 = (esimd_ns::convert<float>(
                                (ql0_u32 >> 4) | (((qh0_u32 >> 4) & 0x03U) << 4)) - 32.0f) * s4;
                            esimd_ns::simd<float, 16> w4 = (esimd_ns::convert<float>(
                                (ql1_u32 >> 4) | ((qh0_u32 >> 6) << 4)) - 32.0f) * s6;

                            for (int c = 0; c < nc; ++c) {
                                const float * xc = x + c * x_stride;
                                const float * x_seg = xc + ib * 256 + half * 128 + base;

                                esimd_ns::simd<float, 16> x0 =
                                    esimd_ns::block_load<float, 16>(const_cast<float *>(x_seg));
                                esimd_ns::simd<float, 16> x1 =
                                    esimd_ns::block_load<float, 16>(const_cast<float *>(x_seg + 32));
                                esimd_ns::simd<float, 16> x2 =
                                    esimd_ns::block_load<float, 16>(const_cast<float *>(x_seg + 64));
                                esimd_ns::simd<float, 16> x3 =
                                    esimd_ns::block_load<float, 16>(const_cast<float *>(x_seg + 96));

                                esimd_ns::simd<float, 16> dot = w1 * x0 + w2 * x1 + w3 * x2 + w4 * x3;
                                acc[c] += esimd_ns::reduce<float>(dot, std::plus<>());
                            }
                        }
                    }
                }

                for (int c = 0; c < nc; ++c) {
                    float v = acc[c];
                    if (res) v += res[c * res_stride + row];
                    y[c * y_stride + row] = v;
                }
            });
    });
}
