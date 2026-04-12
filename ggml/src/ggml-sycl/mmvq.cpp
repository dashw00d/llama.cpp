#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sycl/ext/intel/esimd.hpp>
#include <unordered_map>
#include <vector>

namespace {

namespace esimd = sycl::ext::intel::esimd;

bool ggml_sycl_debug_q4k_cpu_aos_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_CPU_AOS") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_esimd_f32_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_ESIMD_F32") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_cpu_aos_should_run(int ncols_x, int nrows_x, int ncols_y) {
    static bool ran = false;
    if (ran || !ggml_sycl_debug_q4k_cpu_aos_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    ran = true;
    return true;
}

bool ggml_sycl_debug_q4k_esimd_f32_should_run(int device, int ncols_x, int nrows_x, int ncols_y) {
    static bool ran[GGML_SYCL_MAX_DEVICES] = {};
    if (!ggml_sycl_debug_q4k_esimd_f32_enabled()) {
        return false;
    }
    if (device >= 0 && device < GGML_SYCL_MAX_DEVICES && ran[device]) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    if (device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        ran[device] = true;
    }
    return true;
}

inline float ggml_sycl_debug_q4k_fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;

    if (exp == 0x1fU) {
        return 0.0f;
    }
    if (exp == 0) {
        if (mant == 0) {
            return 0.0f;
        }
        float val = static_cast<float>(mant) * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000U) ? -val : val;
    }

    return sycl::bit_cast<float>(sign | ((exp + 112U) << 23) | (mant << 13));
}

void ggml_sycl_debug_q4k_unpack_meta(
    const std::uint8_t * meta_ptr,
    float scales[8],
    float biases[8]) {
    const float scale_base = ggml_fp16_to_fp32(
        static_cast<ggml_fp16_t>(meta_ptr[0] | (meta_ptr[1] << 8)));
    const float neg_min_base = -ggml_fp16_to_fp32(
        static_cast<ggml_fp16_t>(meta_ptr[2] | (meta_ptr[3] << 8)));

    std::uint8_t raw_scales[8];
    std::uint8_t raw_mins[8];
    for (int i = 0; i < 4; ++i) {
        const std::uint8_t a = meta_ptr[4 + i];
        const std::uint8_t b = meta_ptr[8 + i];
        const std::uint8_t c = meta_ptr[12 + i];
        raw_scales[i]     = static_cast<std::uint8_t>(a & 0x3fU);
        raw_scales[i + 4] = static_cast<std::uint8_t>((c & 0x0fU) | ((a >> 2) & 0x30U));
        raw_mins[i]       = static_cast<std::uint8_t>(b & 0x3fU);
        raw_mins[i + 4]   = static_cast<std::uint8_t>((c >> 4) | ((b >> 2) & 0x30U));
    }

    for (int i = 0; i < 8; ++i) {
        scales[i] = scale_base * static_cast<float>(raw_scales[i]);
        biases[i] = neg_min_base * static_cast<float>(raw_mins[i]);
    }
}

float ggml_sycl_debug_q4k_dot_aos_q8_1(
    const block_q4_K & block,
    const float * x_q8_scales,
    const int8_t * x_q8_quants) {
    float scales[8];
    float biases[8];
    ggml_sycl_debug_q4k_unpack_meta(
        reinterpret_cast<const std::uint8_t *>(&block),
        scales,
        biases);

    float partial = 0.0f;
    for (int pair = 0; pair < 4; ++pair) {
        const int low_group = pair * 2;
        const int high_group = low_group + 1;
        const float d_low = x_q8_scales[low_group];
        const float d_high = x_q8_scales[high_group];
        const int8_t * x_low = x_q8_quants + low_group * QK8_1;
        const int8_t * x_high = x_q8_quants + high_group * QK8_1;
        for (int i = 0; i < 32; ++i) {
            const std::uint8_t packed = block.qs[pair * 32 + i];
            const float lo = static_cast<float>(packed & 0x0fU);
            const float hi = static_cast<float>(packed >> 4);
            partial += (scales[low_group] * lo + biases[low_group]) * (d_low * static_cast<float>(x_low[i]));
            partial += (scales[high_group] * hi + biases[high_group]) * (d_high * static_cast<float>(x_high[i]));
        }
    }
    return partial;
}

float ggml_sycl_debug_q4k_dot_aos_f32(
    const block_q4_K & block,
    const float * x_block) {
    float scales[8];
    float biases[8];
    ggml_sycl_debug_q4k_unpack_meta(
        reinterpret_cast<const std::uint8_t *>(&block),
        scales,
        biases);

    float partial = 0.0f;
    for (int pair = 0; pair < 4; ++pair) {
        const int low_group = pair * 2;
        const int high_group = low_group + 1;
        const int x_low = low_group * 32;
        const int x_high = high_group * 32;
        for (int i = 0; i < 32; ++i) {
            const std::uint8_t packed = block.qs[pair * 32 + i];
            const float lo = static_cast<float>(packed & 0x0fU);
            const float hi = static_cast<float>(packed >> 4);
            partial += (scales[low_group] * lo + biases[low_group]) * x_block[x_low + i];
            partial += (scales[high_group] * hi + biases[high_group]) * x_block[x_high + i];
        }
    }
    return partial;
}

void ggml_sycl_debug_quantize_q8_1_cpu(
    const float * x,
    float * scales_out,
    int8_t * quants_out,
    int n_blocks) {
    for (int ib = 0; ib < n_blocks; ++ib) {
        const float * x_block = x + static_cast<size_t>(ib) * QK8_1;
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; ++i) {
            amax = std::max(amax, std::fabs(x_block[i]));
        }

        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        scales_out[ib] = d;

        for (int i = 0; i < QK8_1; ++i) {
            quants_out[static_cast<size_t>(ib) * QK8_1 + i] =
                d == 0.0f ? 0 : static_cast<int8_t>(std::nearbyint(x_block[i] / d));
        }
    }
}

void ggml_sycl_debug_compare_q4k_cpu_aos(
    const void * vx,
    const float * x,
    const float * dst,
    int ncols_x,
    int nrows_x,
    int ncols_y,
    size_t dst_col_stride,
    dpct::queue_ptr stream) {
    if (!ggml_sycl_debug_q4k_cpu_aos_should_run(ncols_x, nrows_x, ncols_y)) {
        return;
    }

    const int n_blocks_per_row = ncols_x / QK_K;
    const int n_q8_blocks_per_col = ncols_x / QK8_1;
    const int rows_to_compare = std::min(nrows_x, 32);

    std::vector<block_q4_K> vx_host(static_cast<size_t>(nrows_x) * n_blocks_per_row);
    std::vector<float> x_host(static_cast<size_t>(ncols_y) * ncols_x);
    std::vector<float> x_q8_scales(static_cast<size_t>(ncols_y) * n_q8_blocks_per_col);
    std::vector<int8_t> x_q8_quants(static_cast<size_t>(ncols_y) * ncols_x);
    std::vector<float> dst_host(dst_col_stride * static_cast<size_t>(ncols_y));

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        vx_host.data(),
        vx,
        vx_host.size() * sizeof(block_q4_K)).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        x_host.data(),
        x,
        x_host.size() * sizeof(float)).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        dst_host.data(),
        dst,
        dst_host.size() * sizeof(float)).wait()));

    for (int c = 0; c < ncols_y; ++c) {
        ggml_sycl_debug_quantize_q8_1_cpu(
            x_host.data() + static_cast<size_t>(c) * ncols_x,
            x_q8_scales.data() + static_cast<size_t>(c) * n_q8_blocks_per_col,
            x_q8_quants.data() + static_cast<size_t>(c) * ncols_x,
            n_q8_blocks_per_col);
    }

    float max_abs_err = 0.0f;
    int max_row = -1;
    int max_col = -1;
    float max_ref = 0.0f;
    float max_stock = 0.0f;

    for (int c = 0; c < ncols_y; ++c) {
        const float * x_col_scales = x_q8_scales.data() + static_cast<size_t>(c) * n_q8_blocks_per_col;
        const int8_t * x_col_quants = x_q8_quants.data() + static_cast<size_t>(c) * ncols_x;
        for (int row = 0; row < rows_to_compare; ++row) {
            const block_q4_K * row_blocks =
                vx_host.data() + static_cast<size_t>(row) * n_blocks_per_row;
            float ref = 0.0f;
            for (int ib = 0; ib < n_blocks_per_row; ++ib) {
                ref += ggml_sycl_debug_q4k_dot_aos_q8_1(
                    row_blocks[ib],
                    x_col_scales + static_cast<size_t>(ib) * (QK_K / QK8_1),
                    x_col_quants + static_cast<size_t>(ib) * QK_K);
            }

            const float stock = dst_host[static_cast<size_t>(c) * dst_col_stride + row];
            const float abs_err = std::fabs(ref - stock);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_row = row;
                max_col = c;
                max_ref = ref;
                max_stock = stock;
            }
        }
    }

    GGML_LOG_INFO(
        "Q4_K CPU AoS compare: rows=%d cols=%d ncols_x=%d max_abs_err=%.6f row=%d col=%d ref=%.6f stock=%.6f\n",
        rows_to_compare,
        ncols_y,
        ncols_x,
        max_abs_err,
        max_row,
        max_col,
        max_ref,
        max_stock);
}

void ggml_sycl_debug_compare_q4k_esimd_f32(
    const void * vx,
    const float * x,
    const float * stock_dst,
    int device,
    int ncols_x,
    int row_low,
    int nrows_x,
    int ncols_y,
    size_t stock_dst_col_stride,
    dpct::queue_ptr stream) {
    if (!ggml_sycl_debug_q4k_esimd_f32_should_run(device, ncols_x, nrows_x, ncols_y)) {
        return;
    }

    const int n_blocks_per_row = ncols_x / QK_K;
    const int rows_to_compare = std::min(nrows_x, 32);
    const int total_blocks = rows_to_compare * n_blocks_per_row;
    const size_t esimd_dst_col_stride = stock_dst_col_stride;

    std::vector<block_q4_K> vx_host(static_cast<size_t>(total_blocks));
    std::vector<float> x_host(static_cast<size_t>(ncols_y) * ncols_x);
    std::vector<float> stock_host(stock_dst_col_stride * static_cast<size_t>(ncols_y));
    std::vector<float> ref_host(static_cast<size_t>(ncols_y) * rows_to_compare);
    std::vector<float> esimd_host(esimd_dst_col_stride * static_cast<size_t>(ncols_y), 0.0f);
    std::vector<std::uint8_t> payload_host(static_cast<size_t>(total_blocks) * 128);
    std::vector<std::uint8_t> meta_host(static_cast<size_t>(total_blocks) * 16);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        vx_host.data(),
        vx,
        vx_host.size() * sizeof(block_q4_K)).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        x_host.data(),
        x,
        x_host.size() * sizeof(float)).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        stock_host.data(),
        stock_dst,
        stock_host.size() * sizeof(float)).wait()));

    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const block_q4_K & block = vx_host[block_idx];
        std::memcpy(payload_host.data() + static_cast<size_t>(block_idx) * 128, block.qs, 128);
        std::memcpy(meta_host.data() + static_cast<size_t>(block_idx) * 16, &block, 16);
    }

    for (int c = 0; c < ncols_y; ++c) {
        const float * x_col = x_host.data() + static_cast<size_t>(c) * ncols_x;
        for (int row = 0; row < rows_to_compare; ++row) {
            const block_q4_K * row_blocks =
                vx_host.data() + static_cast<size_t>(row) * n_blocks_per_row;
            float ref = 0.0f;
            for (int ib = 0; ib < n_blocks_per_row; ++ib) {
                ref += ggml_sycl_debug_q4k_dot_aos_f32(
                    row_blocks[ib],
                    x_col + static_cast<size_t>(ib) * QK_K);
            }
            ref_host[static_cast<size_t>(c) * rows_to_compare + row] = ref;
        }
    }

    auto * payload_dev = static_cast<std::uint8_t *>(sycl::malloc_device(payload_host.size(), *stream));
    auto * meta_dev = static_cast<std::uint8_t *>(sycl::malloc_device(meta_host.size(), *stream));
    auto * esimd_dev = static_cast<float *>(sycl::malloc_device(esimd_host.size() * sizeof(float), *stream));
    GGML_ASSERT(payload_dev != nullptr);
    GGML_ASSERT(meta_dev != nullptr);
    GGML_ASSERT(esimd_dev != nullptr);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(payload_dev, payload_host.data(), payload_host.size()).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(meta_dev, meta_host.data(), meta_host.size()).wait()));

    stream->submit([&](sycl::handler & cgh) {
        const auto payload_base = payload_dev;
        const auto meta_base = meta_dev;
        const auto x_base = x;
        const auto y_base = esimd_dev;
        const auto nbpr = n_blocks_per_row;
        const auto nr = rows_to_compare;
        const auto nc = ncols_y;
        const auto x_stride = static_cast<size_t>(ncols_x);
        const auto y_stride = esimd_dst_col_stride;

        cgh.parallel_for(
            sycl::range<1>(static_cast<size_t>(rows_to_compare)),
            [=](sycl::id<1> row_id) SYCL_ESIMD_KERNEL {
                const int row = static_cast<int>(row_id[0]);
                if (row >= nr) {
                    return;
                }

                float acc[8] = {};

                for (int ib = 0; ib < nbpr; ++ib) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(nbpr) +
                        static_cast<std::size_t>(ib);
                    const std::uint8_t * payload = payload_base + block_idx * 128;
                    const std::uint8_t * meta_ptr = meta_base + block_idx * 16;

                    const float scale_base = ggml_sycl_debug_q4k_fp16_to_float(
                        static_cast<std::uint16_t>(meta_ptr[0] | (meta_ptr[1] << 8)));
                    const float neg_min_base = -ggml_sycl_debug_q4k_fp16_to_float(
                        static_cast<std::uint16_t>(meta_ptr[2] | (meta_ptr[3] << 8)));

                    float scales[8];
                    float biases[8];
                    for (int i = 0; i < 4; ++i) {
                        const std::uint8_t a = meta_ptr[4 + i];
                        const std::uint8_t b = meta_ptr[8 + i];
                        const std::uint8_t c = meta_ptr[12 + i];
                        const std::uint8_t scale0 = static_cast<std::uint8_t>(a & 0x3fU);
                        const std::uint8_t scale1 = static_cast<std::uint8_t>((c & 0x0fU) | ((a >> 2) & 0x30U));
                        const std::uint8_t min0 = static_cast<std::uint8_t>(b & 0x3fU);
                        const std::uint8_t min1 = static_cast<std::uint8_t>((c >> 4) | ((b >> 2) & 0x30U));
                        scales[i] = scale_base * static_cast<float>(scale0);
                        scales[i + 4] = scale_base * static_cast<float>(scale1);
                        biases[i] = neg_min_base * static_cast<float>(min0);
                        biases[i + 4] = neg_min_base * static_cast<float>(min1);
                    }

                    esimd::simd<std::uint32_t, 128> full_block =
                        esimd::convert<std::uint32_t>(
                            esimd::block_load<std::uint8_t, 128>(
                                const_cast<std::uint8_t *>(payload)));

                    for (int pair = 0; pair < 4; ++pair) {
                        const int low_group = pair * 2;
                        const int high_group = low_group + 1;
                        const float low_scale = scales[low_group];
                        const float high_scale = scales[high_group];
                        const float low_bias = biases[low_group];
                        const float high_bias = biases[high_group];

                        esimd::simd<std::uint32_t, 32> packed_u32 =
                            full_block.template select<32, 1>(pair * 32);

                        esimd::simd<float, 32> w_lo =
                            esimd::convert<float>(packed_u32 & 0x0fU) * low_scale + low_bias;
                        esimd::simd<float, 32> w_hi =
                            esimd::convert<float>(packed_u32 >> 4) * high_scale + high_bias;

                        const std::size_t x_lo_off = static_cast<std::size_t>(ib) * 256 + low_group * 32;
                        const std::size_t x_hi_off = static_cast<std::size_t>(ib) * 256 + high_group * 32;

                        esimd::simd<float, 16> wl0 = w_lo.template select<16, 1>(0);
                        esimd::simd<float, 16> wl1 = w_lo.template select<16, 1>(16);
                        esimd::simd<float, 16> wh0 = w_hi.template select<16, 1>(0);
                        esimd::simd<float, 16> wh1 = w_hi.template select<16, 1>(16);

                        for (int c = 0; c < nc; ++c) {
                            const float * xc = x_base + c * x_stride;

                            esimd::simd<float, 16> x_lo_0 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_lo_off));
                            esimd::simd<float, 16> x_lo_1 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_lo_off + 16));
                            esimd::simd<float, 16> x_hi_0 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_hi_off));
                            esimd::simd<float, 16> x_hi_1 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_hi_off + 16));

                            acc[c] += esimd::reduce<float>(wl0 * x_lo_0 + wl1 * x_lo_1, std::plus<>());
                            acc[c] += esimd::reduce<float>(wh0 * x_hi_0 + wh1 * x_hi_1, std::plus<>());
                        }
                    }
                }

                for (int c = 0; c < nc; ++c) {
                    y_base[c * y_stride + row] = acc[c];
                }
            });
    });
    stream->wait();

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        esimd_host.data(),
        esimd_dev,
        esimd_host.size() * sizeof(float)).wait()));

    float max_ref_err = 0.0f;
    int max_ref_row = -1;
    int max_ref_col = -1;
    float max_ref_expected = 0.0f;
    float max_ref_actual = 0.0f;

    float max_stock_err = 0.0f;
    int max_stock_row = -1;
    int max_stock_col = -1;
    float max_stock_expected = 0.0f;
    float max_stock_actual = 0.0f;

    for (int c = 0; c < ncols_y; ++c) {
        for (int row = 0; row < rows_to_compare; ++row) {
            const float esimd_val = esimd_host[static_cast<size_t>(c) * esimd_dst_col_stride + row];
            const float ref_val = ref_host[static_cast<size_t>(c) * rows_to_compare + row];
            const float stock_val = stock_host[static_cast<size_t>(c) * stock_dst_col_stride + row];

            const float ref_err = std::fabs(esimd_val - ref_val);
            if (ref_err > max_ref_err) {
                max_ref_err = ref_err;
                max_ref_row = row;
                max_ref_col = c;
                max_ref_expected = ref_val;
                max_ref_actual = esimd_val;
            }

            const float stock_err = std::fabs(esimd_val - stock_val);
            if (stock_err > max_stock_err) {
                max_stock_err = stock_err;
                max_stock_row = row;
                max_stock_col = c;
                max_stock_expected = stock_val;
                max_stock_actual = esimd_val;
            }
        }
    }

    GGML_LOG_INFO(
        "Q4_K ESIMD F32 compare: device=%d row_low=%d rows=%d cols=%d ncols_x=%d max_ref_err=%.6f row=%d col=%d ref=%.6f esimd=%.6f max_stock_err=%.6f stock_row=%d stock_col=%d stock=%.6f esimd_stock=%.6f\n",
        device,
        row_low,
        rows_to_compare,
        ncols_y,
        ncols_x,
        max_ref_err,
        max_ref_row,
        max_ref_col,
        max_ref_expected,
        max_ref_actual,
        max_stock_err,
        max_stock_row,
        max_stock_col,
        max_stock_expected,
        max_stock_actual);

    sycl::free(payload_dev, *stream);
    sycl::free(meta_dev, *stream);
    sycl::free(esimd_dev, *stream);
}

/* gemma4-ipex iter7: per-device cached split buffers for the live ESIMD Q4_K path.

   Iter 5's scratch compare and iter 6's uncached live path proved the
   cleanroom fused kernel is correct when fed its required payload[128]
   + meta[16] split contract. Iter 6's -npl 8 collapse (7.34 -> 0.55
   t/s) was pure per-call scratch overhead: host round-trip memcpy +
   malloc_device + free on every matching call.

   This iter7 rewrite removes that overhead by caching the split per
   (device, vx) pair. First call on a given weight tensor pays the
   full split + upload cost once; every subsequent call on the same
   weight reuses the cached payload_dev / meta_dev and only runs the
   kernel. Cache entries are invalidated when total_blocks changes
   for the same vx (graph-shape change) and freed in place.

   Still gated on GGML_SYCL_DEBUG_Q4K_ESIMD_LIVE with the narrow
   5376x8 shape gate -- this is a diagnostic, not the production
   Q4K dispatch.

   NOTE: PROVEN-KERNEL-ARCHITECTURE.md / PORTING-GUIDE.md wording that
   claims the kernel reads ggml AoS block_q4_K directly is WRONG. See
   include/ipex_cleanroom/esimd/linear_forward_q4k_fused_sycl.hpp
   (linear_forward_q4k_fused_args { payload[128/blk] + meta[16/blk] })
   and src/esimd/linear_forward_q4k_fused_sycl.cpp:115 -- the kernel
   indexes separate payload and meta contiguous buffers, not a 144-byte
   AoS stride. Iter 4's AoS-direct port failed because block_load<128>
   on a non-128-byte-aligned payload pointer inside a 144-byte AoS
   stride reads garbage. The split is not optional. */

bool ggml_sycl_debug_q4k_esimd_live_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_ESIMD_LIVE") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_esimd_live_should_run(int ncols_x, int nrows_x, int ncols_y) {
    if (!ggml_sycl_debug_q4k_esimd_live_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    return true;
}

struct ggml_sycl_q4k_esimd_cached_split {
    std::uint8_t * payload_dev;
    std::uint8_t * meta_dev;
    int total_blocks;
};

struct ggml_sycl_q4k_esimd_live_arena_device {
    std::unordered_map<const void *, ggml_sycl_q4k_esimd_cached_split> cache;
};

static ggml_sycl_q4k_esimd_live_arena_device g_q4k_esimd_live_arena[GGML_SYCL_MAX_DEVICES];

const ggml_sycl_q4k_esimd_cached_split * ggml_sycl_debug_q4k_esimd_live_cache_lookup(
    const void * vx,
    int total_blocks,
    int device,
    dpct::queue_ptr stream) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }

    auto & cache = g_q4k_esimd_live_arena[device].cache;
    auto it = cache.find(vx);

    if (it != cache.end() && it->second.total_blocks == total_blocks) {
        return &it->second;
    }

    // Cache miss OR invalidation because total_blocks changed.
    if (it != cache.end()) {
        GGML_LOG_INFO(
            "Q4_K ESIMD LIVE cache invalidation: device=%d vx=%p old_blocks=%d new_blocks=%d\n",
            device, vx, it->second.total_blocks, total_blocks);
        sycl::free(it->second.payload_dev, *stream);
        sycl::free(it->second.meta_dev, *stream);
        cache.erase(it);
    }

    std::vector<block_q4_K> vx_host(static_cast<size_t>(total_blocks));
    std::vector<std::uint8_t> payload_host(static_cast<size_t>(total_blocks) * 128);
    std::vector<std::uint8_t> meta_host(static_cast<size_t>(total_blocks) * 16);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        vx_host.data(),
        vx,
        vx_host.size() * sizeof(block_q4_K)).wait()));

    for (int i = 0; i < total_blocks; ++i) {
        const block_q4_K & block = vx_host[i];
        std::memcpy(payload_host.data() + static_cast<size_t>(i) * 128, block.qs, 128);
        std::memcpy(meta_host.data() + static_cast<size_t>(i) * 16, &block, 16);
    }

    auto * payload_dev = static_cast<std::uint8_t *>(sycl::malloc_device(payload_host.size(), *stream));
    auto * meta_dev = static_cast<std::uint8_t *>(sycl::malloc_device(meta_host.size(), *stream));
    GGML_ASSERT(payload_dev != nullptr);
    GGML_ASSERT(meta_dev != nullptr);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(payload_dev, payload_host.data(), payload_host.size()).wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(meta_dev, meta_host.data(), meta_host.size()).wait()));

    ggml_sycl_q4k_esimd_cached_split entry = { payload_dev, meta_dev, total_blocks };
    auto [inserted_it, ok] = cache.emplace(vx, entry);
    (void) ok;

    GGML_LOG_INFO(
        "Q4_K ESIMD LIVE cache fill: device=%d vx=%p total_blocks=%d payload_bytes=%zu meta_bytes=%zu cache_entries=%zu\n",
        device, vx, total_blocks,
        payload_host.size(), meta_host.size(),
        cache.size());

    return &inserted_it->second;
}

void ggml_sycl_debug_q4k_esimd_live(
    const void * vx,
    const float * x,
    float * dst,
    int device,
    int ncols_x,
    int row_low,
    int nrows_x,
    int ncols_y,
    size_t dst_col_stride,
    dpct::queue_ptr stream) {
    if (!ggml_sycl_debug_q4k_esimd_live_should_run(ncols_x, nrows_x, ncols_y)) {
        return;
    }

    {
        static bool announced[GGML_SYCL_MAX_DEVICES] = {};
        if (device >= 0 && device < GGML_SYCL_MAX_DEVICES && !announced[device]) {
            announced[device] = true;
            GGML_LOG_INFO(
                "Q4_K ESIMD LIVE (cached) active: device=%d ncols_x=%d nrows_x=%d ncols_y=%d dst_stride=%zu row_low=%d\n",
                device, ncols_x, nrows_x, ncols_y, dst_col_stride, row_low);
        }
    }

    const int n_blocks_per_row = ncols_x / QK_K;
    const int total_blocks = nrows_x * n_blocks_per_row;

    const auto * entry = ggml_sycl_debug_q4k_esimd_live_cache_lookup(vx, total_blocks, device, stream);
    if (entry == nullptr) {
        return;
    }

    stream->submit([&](sycl::handler & cgh) {
        const auto payload_base = entry->payload_dev;
        const auto meta_base = entry->meta_dev;
        const auto x_base = x;
        float * const y_base = dst;
        const auto nbpr = n_blocks_per_row;
        const auto nr = nrows_x;
        const auto nc = ncols_y;
        const auto x_stride = static_cast<size_t>(ncols_x);
        const auto y_stride = dst_col_stride;

        cgh.parallel_for(
            sycl::range<1>(static_cast<size_t>(nrows_x)),
            [=](sycl::id<1> row_id) SYCL_ESIMD_KERNEL {
                const int row = static_cast<int>(row_id[0]);
                if (row >= nr) {
                    return;
                }

                float acc[8] = {};

                for (int ib = 0; ib < nbpr; ++ib) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(nbpr) +
                        static_cast<std::size_t>(ib);
                    const std::uint8_t * payload = payload_base + block_idx * 128;
                    const std::uint8_t * meta_ptr = meta_base + block_idx * 16;

                    const float scale_base = ggml_sycl_debug_q4k_fp16_to_float(
                        static_cast<std::uint16_t>(meta_ptr[0] | (meta_ptr[1] << 8)));
                    const float neg_min_base = -ggml_sycl_debug_q4k_fp16_to_float(
                        static_cast<std::uint16_t>(meta_ptr[2] | (meta_ptr[3] << 8)));

                    float scales[8];
                    float biases[8];
                    for (int i = 0; i < 4; ++i) {
                        const std::uint8_t a = meta_ptr[4 + i];
                        const std::uint8_t b = meta_ptr[8 + i];
                        const std::uint8_t c = meta_ptr[12 + i];
                        const std::uint8_t scale0 = static_cast<std::uint8_t>(a & 0x3fU);
                        const std::uint8_t scale1 = static_cast<std::uint8_t>((c & 0x0fU) | ((a >> 2) & 0x30U));
                        const std::uint8_t min0 = static_cast<std::uint8_t>(b & 0x3fU);
                        const std::uint8_t min1 = static_cast<std::uint8_t>((c >> 4) | ((b >> 2) & 0x30U));
                        scales[i] = scale_base * static_cast<float>(scale0);
                        scales[i + 4] = scale_base * static_cast<float>(scale1);
                        biases[i] = neg_min_base * static_cast<float>(min0);
                        biases[i + 4] = neg_min_base * static_cast<float>(min1);
                    }

                    esimd::simd<std::uint32_t, 128> full_block =
                        esimd::convert<std::uint32_t>(
                            esimd::block_load<std::uint8_t, 128>(
                                const_cast<std::uint8_t *>(payload)));

                    for (int pair = 0; pair < 4; ++pair) {
                        const int low_group = pair * 2;
                        const int high_group = low_group + 1;
                        const float low_scale = scales[low_group];
                        const float high_scale = scales[high_group];
                        const float low_bias = biases[low_group];
                        const float high_bias = biases[high_group];

                        esimd::simd<std::uint32_t, 32> packed_u32 =
                            full_block.template select<32, 1>(pair * 32);

                        esimd::simd<float, 32> w_lo =
                            esimd::convert<float>(packed_u32 & 0x0fU) * low_scale + low_bias;
                        esimd::simd<float, 32> w_hi =
                            esimd::convert<float>(packed_u32 >> 4) * high_scale + high_bias;

                        const std::size_t x_lo_off = static_cast<std::size_t>(ib) * 256 + low_group * 32;
                        const std::size_t x_hi_off = static_cast<std::size_t>(ib) * 256 + high_group * 32;

                        esimd::simd<float, 16> wl0 = w_lo.template select<16, 1>(0);
                        esimd::simd<float, 16> wl1 = w_lo.template select<16, 1>(16);
                        esimd::simd<float, 16> wh0 = w_hi.template select<16, 1>(0);
                        esimd::simd<float, 16> wh1 = w_hi.template select<16, 1>(16);

                        for (int c = 0; c < nc; ++c) {
                            const float * xc = x_base + c * x_stride;

                            esimd::simd<float, 16> x_lo_0 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_lo_off));
                            esimd::simd<float, 16> x_lo_1 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_lo_off + 16));
                            esimd::simd<float, 16> x_hi_0 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_hi_off));
                            esimd::simd<float, 16> x_hi_1 =
                                esimd::block_load<float, 16>(const_cast<float *>(xc + x_hi_off + 16));

                            acc[c] += esimd::reduce<float>(wl0 * x_lo_0 + wl1 * x_lo_1, std::plus<>());
                            acc[c] += esimd::reduce<float>(wh0 * x_hi_0 + wh1 * x_hi_1, std::plus<>());
                        }
                    }
                }

                for (int c = 0; c < nc; ++c) {
                    y_base[c * y_stride + row] = acc[c];
                }
            });
    });
    // gemma4-ipex iter8: NO stream->wait() here. Stock dispatch runs the
    // stream asynchronously -- the in-order SYCL queue serializes subsequent
    // ops after this submit naturally. The host-side wait was one of the
    // two sources of the remaining 4x gap after iter7.

    // Cache owns payload_dev / meta_dev -- no per-call free.
}

} // namespace

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_nvfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}


static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}
static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                    reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                // gemma4-ipex iter8: skip stock MMVQ entirely when the ESIMD
                // live path will overwrite the result on the gated shape.
                // Avoids doing the same matmul twice on gated calls.
                if (ggml_sycl_debug_q4k_esimd_live_should_run(
                        static_cast<int>(ne00),
                        static_cast<int>(row_diff),
                        static_cast<int>(src1_ncols))) {
                    GGML_SYCL_DEBUG("Skipping stock Q4K MMVQ: ESIMD live path owns this shape\n");
                    break;
                }
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_MXFP4:
                mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_NVFP4:
                mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(src0->type));
        }
    }
    if (src0->type == GGML_TYPE_Q4_K) {
        ggml_sycl_debug_compare_q4k_cpu_aos(
            src0_dd_i,
            src1_ddf_i,
            dst_dd_i,
            static_cast<int>(ne00),
            static_cast<int>(row_diff),
            static_cast<int>(src1_ncols),
            static_cast<size_t>(dst->ne[0]),
            stream);
        ggml_sycl_debug_compare_q4k_esimd_f32(
            src0_dd_i,
            src1_ddf_i,
            dst_dd_i,
            id,
            static_cast<int>(ne00),
            static_cast<int>(row_low),
            static_cast<int>(row_diff),
            static_cast<int>(src1_ncols),
            static_cast<size_t>(dst->ne[0]),
            stream);
        ggml_sycl_debug_q4k_esimd_live(
            src0_dd_i,
            src1_ddf_i,
            dst_dd_i,
            id,
            static_cast<int>(ne00),
            static_cast<int>(row_low),
            static_cast<int>(row_diff),
            static_cast<int>(src1_ncols),
            static_cast<size_t>(dst->ne[0]),
            stream);
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}
