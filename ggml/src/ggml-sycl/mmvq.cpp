#include "mmvq.hpp"

#include "ggml.h"
#include "ggml-quants.h"
#include "common.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <unordered_map>
#include <vector>

namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace jm = sycl::ext::oneapi::experimental::matrix;

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

/* gemma4-ipex iter9: env-gated per-call timing for the live Q4_K path.

   Enabled by GGML_SYCL_DEBUG_Q4K_ESIMD_LIVE_TIMING=1 (on top of LIVE=1).
   When enabled:
     - ggml_sycl_debug_q4k_esimd_live wraps its ESIMD submit in a
       stream->wait() sandwich and records wall-clock time per call
     - ggml_sycl_debug_q4k_esimd_live_cache_lookup records wall-clock
       time per cache fill (miss path only)
     - a static destructor prints a per-device summary at program exit

   The wait() sandwich deliberately undoes iter8's async optimisation
   for the timing run, so per-call numbers are comparable to stock's
   serial dispatch model. The production (timing-off) path is
   unchanged and async. */

struct ggml_sycl_q4k_esimd_live_timing_stats {
    std::uint64_t kernel_count = 0;
    double        kernel_total_ns = 0.0;
    double        kernel_min_ns = 1e18;
    double        kernel_max_ns = 0.0;
    std::uint64_t fill_count = 0;
    double        fill_total_ns = 0.0;
    double        fill_min_ns = 1e18;
    double        fill_max_ns = 0.0;
};

static ggml_sycl_q4k_esimd_live_timing_stats g_q4k_esimd_live_timing[GGML_SYCL_MAX_DEVICES];

bool ggml_sycl_debug_q4k_esimd_live_timing_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_ESIMD_LIVE_TIMING") != nullptr;
    return enabled;
}

void ggml_sycl_debug_q4k_esimd_live_timing_record_kernel(int device, double ns) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    auto & s = g_q4k_esimd_live_timing[device];
    s.kernel_count++;
    s.kernel_total_ns += ns;
    if (ns < s.kernel_min_ns) s.kernel_min_ns = ns;
    if (ns > s.kernel_max_ns) s.kernel_max_ns = ns;
    if (s.kernel_count % 200 == 0) {
        GGML_LOG_INFO(
            "Q4_K ESIMD LIVE kernel timing: device=%d count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s\n",
            device,
            (unsigned long long) s.kernel_count,
            s.kernel_total_ns / (double) s.kernel_count / 1e6,
            s.kernel_min_ns / 1e6,
            s.kernel_max_ns / 1e6,
            s.kernel_total_ns / 1e9);
    }
}

void ggml_sycl_debug_q4k_esimd_live_timing_record_fill(int device, double ns) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    auto & s = g_q4k_esimd_live_timing[device];
    s.fill_count++;
    s.fill_total_ns += ns;
    if (ns < s.fill_min_ns) s.fill_min_ns = ns;
    if (ns > s.fill_max_ns) s.fill_max_ns = ns;
    GGML_LOG_INFO(
        "Q4_K ESIMD LIVE fill timing: device=%d count=%llu this=%.2f ms avg=%.2f min=%.2f max=%.2f total=%.2f s\n",
        device,
        (unsigned long long) s.fill_count,
        ns / 1e6,
        s.fill_total_ns / (double) s.fill_count / 1e6,
        s.fill_min_ns / 1e6,
        s.fill_max_ns / 1e6,
        s.fill_total_ns / 1e9);
}

struct ggml_sycl_q4k_esimd_live_timing_summary_printer {
    ~ggml_sycl_q4k_esimd_live_timing_summary_printer() {
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            const auto & s = g_q4k_esimd_live_timing[d];
            if (s.kernel_count == 0 && s.fill_count == 0) {
                continue;
            }
            GGML_LOG_INFO(
                "Q4_K ESIMD LIVE timing SUMMARY device=%d: "
                "kernel count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s | "
                "fill count=%llu avg=%.2f ms min=%.2f max=%.2f total=%.2f s\n",
                d,
                (unsigned long long) s.kernel_count,
                s.kernel_count ? s.kernel_total_ns / (double) s.kernel_count / 1e6 : 0.0,
                s.kernel_count ? s.kernel_min_ns / 1e6 : 0.0,
                s.kernel_max_ns / 1e6,
                s.kernel_total_ns / 1e9,
                (unsigned long long) s.fill_count,
                s.fill_count ? s.fill_total_ns / (double) s.fill_count / 1e6 : 0.0,
                s.fill_count ? s.fill_min_ns / 1e6 : 0.0,
                s.fill_max_ns / 1e6,
                s.fill_total_ns / 1e9);
        }
    }
};

static ggml_sycl_q4k_esimd_live_timing_summary_printer g_q4k_esimd_live_timing_summary_printer;

/* gemma4-ipex iter9: parallel timing for stock Q4_K MMVQ on the same shape.

   Enabled by GGML_SYCL_DEBUG_Q4K_STOCK_TIMING=1. When set, the Q4_K case in
   ggml_sycl_op_mul_mat_vec_q wraps the stock mul_mat_vec_q4_K_q8_1_sycl
   (or its reordered cousin) with the same steady_clock + wait() timer and
   records stats per device. Compare against the live kernel's own
   per-call numbers on the same gated shape. Run with LIVE unset so stock
   actually runs (the iter8 skip only fires when LIVE is set). */

struct ggml_sycl_q4k_stock_timing_stats {
    std::uint64_t count = 0;
    double        total_ns = 0.0;
    double        min_ns = 1e18;
    double        max_ns = 0.0;
};

static ggml_sycl_q4k_stock_timing_stats g_q4k_stock_timing[GGML_SYCL_MAX_DEVICES];

bool ggml_sycl_debug_q4k_stock_timing_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_STOCK_TIMING") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_stock_timing_should_run(int ncols_x, int nrows_x, int ncols_y) {
    if (!ggml_sycl_debug_q4k_stock_timing_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    return true;
}

void ggml_sycl_debug_q4k_stock_timing_record(int device, double ns) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    auto & s = g_q4k_stock_timing[device];
    s.count++;
    s.total_ns += ns;
    if (ns < s.min_ns) s.min_ns = ns;
    if (ns > s.max_ns) s.max_ns = ns;
    if (s.count % 200 == 0) {
        GGML_LOG_INFO(
            "Q4_K STOCK MMVQ timing: device=%d count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s\n",
            device,
            (unsigned long long) s.count,
            s.total_ns / (double) s.count / 1e6,
            s.min_ns / 1e6,
            s.max_ns / 1e6,
            s.total_ns / 1e9);
    }
}

struct ggml_sycl_q4k_stock_timing_summary_printer {
    ~ggml_sycl_q4k_stock_timing_summary_printer() {
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            const auto & s = g_q4k_stock_timing[d];
            if (s.count == 0) {
                continue;
            }
            GGML_LOG_INFO(
                "Q4_K STOCK MMVQ timing SUMMARY device=%d: count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s\n",
                d,
                (unsigned long long) s.count,
                s.total_ns / (double) s.count / 1e6,
                s.min_ns / 1e6,
                s.max_ns / 1e6,
                s.total_ns / 1e9);
        }
    }
};

static ggml_sycl_q4k_stock_timing_summary_printer g_q4k_stock_timing_summary_printer;

/* gemma4-ipex iter11: per-device timing for the XMX live path. Same structure
   as the iter9 ESIMD and stock timing paths. Enabled via
   GGML_SYCL_DEBUG_Q4K_XMX_LIVE_TIMING=1. Adds a stream->wait() around the
   GEMM submit so the chrono clock measures actual kernel completion. */

static ggml_sycl_q4k_stock_timing_stats g_q4k_xmx_live_timing[GGML_SYCL_MAX_DEVICES];

bool ggml_sycl_debug_q4k_xmx_live_timing_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_XMX_LIVE_TIMING") != nullptr;
    return enabled;
}

void ggml_sycl_debug_q4k_xmx_live_timing_record(int device, double ns) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    auto & s = g_q4k_xmx_live_timing[device];
    s.count++;
    s.total_ns += ns;
    if (ns < s.min_ns) s.min_ns = ns;
    if (ns > s.max_ns) s.max_ns = ns;
    if (s.count % 200 == 0) {
        GGML_LOG_INFO(
            "Q4_K XMX LIVE kernel timing: device=%d count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s\n",
            device,
            (unsigned long long) s.count,
            s.total_ns / (double) s.count / 1e6,
            s.min_ns / 1e6,
            s.max_ns / 1e6,
            s.total_ns / 1e9);
    }
}

struct ggml_sycl_q4k_xmx_live_timing_summary_printer {
    ~ggml_sycl_q4k_xmx_live_timing_summary_printer() {
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            const auto & s = g_q4k_xmx_live_timing[d];
            if (s.count == 0) {
                continue;
            }
            GGML_LOG_INFO(
                "Q4_K XMX LIVE kernel timing SUMMARY device=%d: count=%llu avg=%.3f ms min=%.3f max=%.3f total=%.2f s\n",
                d,
                (unsigned long long) s.count,
                s.total_ns / (double) s.count / 1e6,
                s.min_ns / 1e6,
                s.max_ns / 1e6,
                s.total_ns / 1e9);
        }
    }
};

static ggml_sycl_q4k_xmx_live_timing_summary_printer g_q4k_xmx_live_timing_summary_printer;

/* gemma4-ipex iter12: pre-dequanted fp16 weight cache for a pure-GEMM XMX path.

   The iter10 XMX kernel is bottlenecked by ~672 work-group barriers per
   subgroup per kernel (336 K-steps x 2 barriers each), which exists because
   B has to be dequanted into SLM in the inner loop. This cache eliminates
   the problem by storing the B matrix as pre-dequanted fp16 weights in a
   TRANSPOSED layout [K=ncols_x][N=nrows_x], so joint_matrix_load for tB
   can read directly from global memory in row_major with stride = nrows_x.
   No SLM for B, no per-K-step dequant, no inner-loop barriers.

   Cache is per-(device, vx) keyed on the weight pointer, same pattern as
   the iter7 split cache. Memory budget: fp16 is 2 bytes/element vs Q4_K's
   0.5625 bytes/element (128 payload + 16 meta per 256 elements), so the
   cache is roughly 3.5x the size of the iter7 cache. With 92 entries per
   device at ~36 MB avg in iter7 (total 3.3 GiB), this cache would be
   ~11.8 GiB per device which exceeds the 16 GiB A770 envelope when
   combined with weights (~5.8 GiB) and KV/compute buffers (~2 GiB).

   Mitigation: per-device cache SIZE CAP (default 3 GiB). Cache lookup
   returns null if adding the new entry would exceed the cap, and the
   live function falls back to the iter10 SLM-dequant path for tensors
   that don't fit. Mixed-mode operation: the N tensors that DO fit get
   the fast pure-GEMM path; the rest keep the slower SLM path. Aggregate
   throughput = weighted average of the two. */

struct ggml_sycl_q4k_xmx_fp16_cached {
    // Stored as uint16_t* on device to avoid sycl::half strict-aliasing
    // issues. The kernel reads the raw 16-bit pattern and converts to
    // float manually. Each element is still an fp16 bit pattern.
    std::uint16_t * fp16_dev;
    int             nrows_x;   // N (output rows for this device slice)
    int             ncols_x;   // K (reduction dim)
};

struct ggml_sycl_q4k_xmx_fp16_arena_device {
    std::unordered_map<const void *, ggml_sycl_q4k_xmx_fp16_cached> cache;
    std::size_t total_bytes = 0;
};

static ggml_sycl_q4k_xmx_fp16_arena_device g_q4k_xmx_fp16_arena[GGML_SYCL_MAX_DEVICES];

// Default cap: 3 GiB per device. Adjust via GGML_SYCL_DEBUG_Q4K_XMX_FP16_CAP_MB.
std::size_t ggml_sycl_debug_q4k_xmx_fp16_cap_bytes() {
    static const std::size_t cap = []() -> std::size_t {
        const char * env = std::getenv("GGML_SYCL_DEBUG_Q4K_XMX_FP16_CAP_MB");
        const std::size_t mb = (env != nullptr) ? static_cast<std::size_t>(std::atoll(env)) : 3072;
        return mb * 1024ULL * 1024ULL;
    }();
    return cap;
}

inline float ggml_sycl_debug_q4k_fp16_host(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    if (exp == 0x1fU) return 0.0f;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        float val = static_cast<float>(mant) * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000U) ? -val : val;
    }
    std::uint32_t out = sign | ((exp + 112U) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
}

// iter12-rev5: host-side fp32 -> fp16 converter that bypasses sycl::half
// entirely. The iter12 cache spent three revs producing garbage output
// on paper-correct data -- the common factor was `sycl::half(float)` in
// host code, which on this oneAPI 2025.0.4 docker image apparently
// silently produces wrong values despite non-zero bit patterns in the
// diagnostic dump. This converter manipulates IEEE 754 bits directly
// and matches ggml_fp32_to_fp16.
inline std::uint16_t ggml_sycl_debug_fp32_to_fp16_host(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign32 = (bits >> 16) & 0x8000U;
    const std::int32_t  exp32  = static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127;
    std::uint32_t       mant32 = bits & 0x007FFFFFU;
    if (exp32 == 128) {
        // inf / nan
        return static_cast<std::uint16_t>(sign32 | 0x7C00U | (mant32 ? 0x0200U : 0U));
    }
    const std::int32_t exp16 = exp32 + 15;
    if (exp16 >= 31) {
        // overflow -> inf
        return static_cast<std::uint16_t>(sign32 | 0x7C00U);
    }
    if (exp16 <= 0) {
        // subnormal or zero. shift the mantissa right (1 - exp16) bits.
        if (exp16 < -10) {
            return static_cast<std::uint16_t>(sign32);
        }
        mant32 |= 0x00800000U; // implicit leading 1
        const std::uint32_t shift = 14 - exp16; // 14 = 23 - 10 + 1
        const std::uint32_t mant16 = mant32 >> shift;
        return static_cast<std::uint16_t>(sign32 | mant16);
    }
    const std::uint32_t mant16 = mant32 >> 13;
    return static_cast<std::uint16_t>(
        sign32 |
        (static_cast<std::uint32_t>(exp16) << 10) |
        mant16);
}

const ggml_sycl_q4k_xmx_fp16_cached * ggml_sycl_debug_q4k_xmx_fp16_cache_lookup(
    const void * vx,
    int nrows_x,
    int ncols_x,
    int device,
    dpct::queue_ptr stream) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return nullptr;
    }

    auto & arena = g_q4k_xmx_fp16_arena[device];
    auto & cache = arena.cache;
    auto it = cache.find(vx);

    if (it != cache.end() &&
        it->second.nrows_x == nrows_x &&
        it->second.ncols_x == ncols_x) {
        return &it->second;
    }

    // Invalidation on shape change.
    if (it != cache.end()) {
        const std::size_t old_bytes =
            static_cast<std::size_t>(it->second.nrows_x) * it->second.ncols_x * sizeof(sycl::half);
        sycl::free(it->second.fp16_dev, *stream);
        arena.total_bytes -= old_bytes;
        cache.erase(it);
    }

    const std::size_t total_elements =
        static_cast<std::size_t>(nrows_x) * static_cast<std::size_t>(ncols_x);
    const std::size_t new_bytes = total_elements * sizeof(sycl::half);

    if (arena.total_bytes + new_bytes > ggml_sycl_debug_q4k_xmx_fp16_cap_bytes()) {
        // Over cap -- let the caller fall back to the iter10 SLM-dequant path.
        GGML_LOG_INFO(
            "Q4_K XMX FP16 cache over cap: device=%d would-be bytes=%zu total=%zu cap=%zu -- falling back for this tensor\n",
            device, new_bytes, arena.total_bytes,
            ggml_sycl_debug_q4k_xmx_fp16_cap_bytes());
        return nullptr;
    }

    const int n_blocks_per_row = ncols_x / QK_K;
    const int total_blocks = nrows_x * n_blocks_per_row;

    // Copy Q4_K blocks to host
    std::vector<block_q4_K> vx_host(static_cast<size_t>(total_blocks));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        vx_host.data(),
        vx,
        vx_host.size() * sizeof(block_q4_K)).wait()));

    // iter12-rev6: use ggml_fp32_to_fp16 (ggml's own reference converter)
    // directly. Store as raw uint16 so no sycl::half host code is involved.
    // iter13: diagnostic is captured RIGHT AFTER the first row is written,
    // so the fp32 values logged match the fp16 values logged.
    std::vector<std::uint16_t> fp16_host(total_elements);
    std::vector<float> row_fp32(static_cast<size_t>(ncols_x));
    for (int n = 0; n < nrows_x; ++n) {
        const block_q4_K * row_blocks =
            vx_host.data() + static_cast<size_t>(n) * n_blocks_per_row;
        dequantize_row_q4_K(row_blocks, row_fp32.data(), ncols_x);
        std::uint16_t * row_out = fp16_host.data() + static_cast<size_t>(n) * ncols_x;
        for (int k = 0; k < ncols_x; ++k) {
            row_out[k] = static_cast<std::uint16_t>(ggml_fp32_to_fp16(row_fp32[k]));
        }
        if (n == 0) {
            static bool dumped = false;
            if (!dumped) {
                dumped = true;
                GGML_LOG_INFO(
                    "Q4_K XMX FP16 cache row0 sample: fp32=%.6f %.6f %.6f %.6f  fp16=0x%04x 0x%04x 0x%04x 0x%04x\n",
                    row_fp32[0], row_fp32[1], row_fp32[2], row_fp32[3],
                    row_out[0], row_out[1], row_out[2], row_out[3]);
            }
        }
    }

    // Allocate device buffer and upload
    std::uint16_t * fp16_dev = sycl::malloc_device<std::uint16_t>(total_elements, *stream);
    if (fp16_dev == nullptr) {
        GGML_LOG_INFO(
            "Q4_K XMX FP16 cache malloc_device FAILED: device=%d bytes=%zu -- falling back\n",
            device, new_bytes);
        return nullptr;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
        fp16_dev,
        fp16_host.data(),
        new_bytes).wait()));

    // iter13 diagnostic: read back the first 4 halves from device memory
    // and compare to the host source we just uploaded.
    {
        static bool verified = false;
        if (!verified) {
            verified = true;
            std::uint16_t readback[4];
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
                readback,
                fp16_dev,
                sizeof(readback)).wait()));
            GGML_LOG_INFO(
                "Q4_K XMX FP16 cache READBACK: host[0..3]=0x%04x 0x%04x 0x%04x 0x%04x  dev[0..3]=0x%04x 0x%04x 0x%04x 0x%04x\n",
                fp16_host[0], fp16_host[1], fp16_host[2], fp16_host[3],
                readback[0], readback[1], readback[2], readback[3]);
        }
    }

    ggml_sycl_q4k_xmx_fp16_cached entry = { fp16_dev, nrows_x, ncols_x };
    auto [inserted_it, ok] = cache.emplace(vx, entry);
    (void) ok;
    arena.total_bytes += new_bytes;

    GGML_LOG_INFO(
        "Q4_K XMX FP16 cache fill: device=%d vx=%p nrows=%d ncols=%d bytes=%zu arena_total=%.2f MiB entries=%zu\n",
        device, vx, nrows_x, ncols_x, new_bytes,
        arena.total_bytes / (1024.0 * 1024.0),
        cache.size());

    return &inserted_it->second;
}

bool ggml_sycl_debug_q4k_xmx_fp16_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_XMX_FP16") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_xmx_fp16_should_run(int ncols_x, int nrows_x, int ncols_y) {
    if (!ggml_sycl_debug_q4k_xmx_fp16_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    return true;
}

/* gemma4-ipex iter10: per-device x_half scratch arena for the XMX live path.
   The x_half buffer is needed per call (activation x changes every step)
   but its MAX size is bounded by ne0 * TM = 5376 * 8 = 43008 halfs = 86 KB
   per device for the gated shape. Allocate once per device on first use,
   reuse across every subsequent call, free never. Eliminates the per-call
   malloc_device / free overhead that was hurting iter10-rev1. */
struct ggml_sycl_q4k_xmx_scratch_device {
    sycl::half * x_half = nullptr;
    std::size_t  capacity = 0;   // in halfs
};
static ggml_sycl_q4k_xmx_scratch_device g_q4k_xmx_scratch[GGML_SYCL_MAX_DEVICES];

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

    const auto fill_t0 = std::chrono::steady_clock::now();

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

    const auto fill_t1 = std::chrono::steady_clock::now();

    GGML_LOG_INFO(
        "Q4_K ESIMD LIVE cache fill: device=%d vx=%p total_blocks=%d payload_bytes=%zu meta_bytes=%zu cache_entries=%zu\n",
        device, vx, total_blocks,
        payload_host.size(), meta_host.size(),
        cache.size());

    if (ggml_sycl_debug_q4k_esimd_live_timing_enabled()) {
        ggml_sycl_debug_q4k_esimd_live_timing_record_fill(
            device,
            std::chrono::duration<double, std::nano>(fill_t1 - fill_t0).count());
    }

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

    const bool timing_enabled = ggml_sycl_debug_q4k_esimd_live_timing_enabled();
    const auto kern_t0 = timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

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
    // gemma4-ipex iter8: NO stream->wait() here in the production path.
    // Stock dispatch runs the stream asynchronously -- the in-order SYCL
    // queue serializes subsequent ops after this submit naturally. The
    // host-side wait was one of the two sources of the remaining 4x gap
    // after iter7.
    //
    // gemma4-ipex iter9: UNDER TIMING ENV, we force a stream->wait() so
    // the chrono clock measures actual kernel completion, not just the
    // submit/enqueue time. Production (timing-off) path is unchanged.
    if (timing_enabled) {
        stream->wait();
        const auto kern_t1 = std::chrono::steady_clock::now();
        ggml_sycl_debug_q4k_esimd_live_timing_record_kernel(
            device,
            std::chrono::duration<double, std::nano>(kern_t1 - kern_t0).count());
    }

    // Cache owns payload_dev / meta_dev -- no per-call free.
}

/* gemma4-ipex iter10: XMX-based Q4_K live path using sycl::joint_matrix.

   First real XMX user in llama.cpp's SYCL backend on Arc. Upstream's
   SYCL_USE_XMX macro is defined but it only switches mmq.cpp tile sizes
   -- zero calls to joint_matrix anywhere in ggml-sycl. This kernel is
   the first one to actually drive the A770 matrix engines for Q4_K.

   Kernel structure adapted from src/esimd/linear_forward_q4k_fused_sycl.cpp:197-398
   (dispatch_linear_forward_q4k_xmx in the cleanroom). Unlike the ESIMD
   per-row kernel, this uses:
     - nd_range with 8-thread subgroups (one subgroup per TN=8 output rows)
     - joint_matrix tiles TM=8 x TN=8 x TK=16 fp16 -> fp32 accumulator
     - SLM-buffered B tile dequanted from the iter7 cached payload/meta
     - A tile loaded from a pre-converted fp16 copy of the activation x
     - one joint_matrix_mad per K-step per subgroup

   Reuses the iter7 per-(device, vx) split cache for the payload/meta
   input -- no extra cache needed. The only new per-call allocations
   are (a) the scratch fp16 x_half buffer (small, ne0 * 8 halfs) and
   (b) the pre-convert kernel launch. Both are removable in entry 11
   if the base idea works. */

bool ggml_sycl_debug_q4k_xmx_live_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_XMX_LIVE") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_xmx_live_should_run(int ncols_x, int nrows_x, int ncols_y) {
    if (!ggml_sycl_debug_q4k_xmx_live_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    return true;
}

void ggml_sycl_debug_q4k_xmx_live(
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
    if (!ggml_sycl_debug_q4k_xmx_live_should_run(ncols_x, nrows_x, ncols_y)) {
        return;
    }

    {
        static bool announced[GGML_SYCL_MAX_DEVICES] = {};
        if (device >= 0 && device < GGML_SYCL_MAX_DEVICES && !announced[device]) {
            announced[device] = true;
            GGML_LOG_INFO(
                "Q4_K XMX LIVE active: device=%d ncols_x=%d nrows_x=%d ncols_y=%d dst_stride=%zu row_low=%d\n",
                device, ncols_x, nrows_x, ncols_y, dst_col_stride, row_low);
        }
    }

    const int n_blocks_per_row = ncols_x / QK_K;
    const int total_blocks = nrows_x * n_blocks_per_row;

    // Reuse the iter7 cache for the payload/meta split.
    const auto * entry = ggml_sycl_debug_q4k_esimd_live_cache_lookup(vx, total_blocks, device, stream);
    if (entry == nullptr) {
        return;
    }

    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int TK = 16;
    constexpr int kSgSize = 8;

    const int ne0 = ncols_x;  // K dimension
    const int n_tiles_n = (nrows_x + TN - 1) / TN;

    // Pre-convert activation x (fp32 [ncols_y][ncols_x]) to fp16 in
    // standard row-major [TM][ne0] layout so joint_matrix_load for tA
    // can read a TM x TK tile at ptr = x_half + k_start with
    // stride = ne0 (leading dimension = full K). col_major tA is not
    // supported on Arc Xe-HPG -- attempting it segfaults at runtime.
    //
    // iter10-rev2: reuse the x_half scratch buffer across calls via a
    // per-device arena. Allocate on first use, grow on larger shapes,
    // never free. Eliminates per-call malloc_device / free overhead.
    const std::size_t x_half_size = static_cast<std::size_t>(TM) * ne0;
    sycl::half * x_half = nullptr;
    if (device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        auto & arena = g_q4k_xmx_scratch[device];
        if (arena.x_half == nullptr || arena.capacity < x_half_size) {
            if (arena.x_half != nullptr) {
                sycl::free(arena.x_half, *stream);
            }
            arena.x_half = sycl::malloc_device<sycl::half>(x_half_size, *stream);
            arena.capacity = x_half_size;
            GGML_LOG_INFO(
                "Q4_K XMX LIVE scratch arena: device=%d allocated %zu halfs (%.2f MiB)\n",
                device, x_half_size,
                (double) x_half_size * sizeof(sycl::half) / (1024.0 * 1024.0));
        }
        x_half = arena.x_half;
    } else {
        x_half = sycl::malloc_device<sycl::half>(x_half_size, *stream);
    }
    GGML_ASSERT(x_half != nullptr);

    {
        const auto nc = ncols_y;
        const auto x_stride = static_cast<size_t>(ncols_x);
        const auto ne0_cap = ne0;
        const auto x_src = x;
        const auto x_h = x_half;
        stream->submit([&](sycl::handler & h) {
            h.parallel_for(
                sycl::range<1>(static_cast<std::size_t>(TM) * ne0),
                [=](sycl::id<1> idx) {
                    const std::size_t flat = idx[0];
                    const std::size_t m = flat / static_cast<std::size_t>(ne0_cap);
                    const std::size_t k = flat % static_cast<std::size_t>(ne0_cap);
                    const float v = (static_cast<int>(m) < nc)
                        ? x_src[m * x_stride + k] : 0.0f;
                    // row-major [TM][ne0]: element (m, k) at m*ne0+k
                    x_h[m * static_cast<std::size_t>(ne0_cap) + k] = sycl::half(v);
                });
        });
    }

    // iter11: optional per-call kernel timing. When
    // GGML_SYCL_DEBUG_Q4K_XMX_LIVE_TIMING=1, wrap the GEMM submit in
    // stream->wait() + steady_clock and record stats per device.
    const bool xmx_timing_enabled = ggml_sycl_debug_q4k_xmx_live_timing_enabled();
    const auto xmx_kern_t0 = xmx_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    // Main GEMM kernel: one subgroup per TN=8 output rows, joint_matrix
    // over K in steps of TK=16. Dequant B from cached payload/meta
    // into SLM cooperatively (TN threads, each dequants 1 row of TK
    // weights per K-step).
    {
        const auto payload_base = entry->payload_dev;
        const auto meta_base = entry->meta_dev;
        const auto nbpr = n_blocks_per_row;
        const auto y = dst;
        const auto nr = nrows_x;
        const auto nc = ncols_y;
        const auto y_stride = dst_col_stride;
        const auto x_h = x_half;
        const auto ne0_val = ne0;

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<sycl::half, 1> slm_b(
                sycl::range<1>(TK * TN), cgh);
            sycl::local_accessor<float, 1> slm_c(
                sycl::range<1>(TM * TN), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(static_cast<std::size_t>(n_tiles_n) * kSgSize),
                    sycl::range<1>(kSgSize)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSgSize)]] {
                    auto sg = it.get_sub_group();
                    const int tile_n = static_cast<int>(it.get_group(0));
                    const int n_base = tile_n * TN;
                    const int lid = static_cast<int>(it.get_local_linear_id());

                    // tA is [TM=8][TK=16] row-major. Loaded with stride=ne0
                    // (the leading dimension of x_half's [TM][ne0] layout).
                    // col_major tA is not supported on Arc -- would segfault.
                    jm::joint_matrix<sycl::sub_group, sycl::half,
                        jm::use::a, TM, TK, jm::layout::row_major> tA;
                    jm::joint_matrix<sycl::sub_group, sycl::half,
                        jm::use::b, TK, TN, jm::layout::row_major> tB;
                    jm::joint_matrix<sycl::sub_group, float,
                        jm::use::accumulator, TM, TN> tC;
                    jm::joint_matrix_fill(sg, tC, 0.0f);

                    for (int k_start = 0; k_start < ne0_val; k_start += TK) {
                        // Load A tile from pre-converted fp16 x. Layout is
                        // [TM][ne0] row-major, so row stride is ne0. Start
                        // at x_h + k_start so columns [k_start, k_start+TK)
                        // are picked up for all TM rows.
                        auto a_ptr = sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(x_h + k_start);
                        jm::joint_matrix_load(sg, tA, a_ptr,
                            static_cast<std::size_t>(ne0_val));

                        // Dequant TN rows of TK weights into SLM B tile.
                        // Each thread (lid 0..TN-1) dequants 1 row.
                        if (lid < TN) {
                            const int row = n_base + lid;
                            if (row < nr) {
                                for (int ki = 0; ki < TK; ++ki) {
                                    const int k = k_start + ki;
                                    const int ib = k / 256;
                                    const int k_in_block = k % 256;
                                    const std::size_t block_idx =
                                        static_cast<std::size_t>(row) * nbpr + ib;
                                    const std::uint8_t * payload =
                                        payload_base + block_idx * 128;
                                    const std::uint8_t * meta =
                                        meta_base + block_idx * 16;

                                    const int pair = k_in_block / 64;
                                    const int pair_pos = k_in_block % 64;
                                    const int group = pair * 2 + (pair_pos >= 32 ? 1 : 0);
                                    const int byte_idx = pair * 32 + (pair_pos % 32);
                                    const std::uint8_t packed = payload[byte_idx];
                                    const float nibble = (pair_pos >= 32)
                                        ? static_cast<float>(packed >> 4)
                                        : static_cast<float>(packed & 0x0FU);

                                    const std::uint16_t d_raw =
                                        meta[0] | (static_cast<std::uint16_t>(meta[1]) << 8);
                                    const std::uint16_t dm_raw =
                                        meta[2] | (static_cast<std::uint16_t>(meta[3]) << 8);
                                    auto fp16f = [](std::uint16_t h) -> float {
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
                                    const float d_val = fp16f(d_raw);
                                    const float neg_dmin = -fp16f(dm_raw);
                                    const std::uint8_t * sc = meta + 4;
                                    float scale, bias;
                                    if (group < 4) {
                                        scale = d_val * static_cast<float>(sc[group] & 0x3FU);
                                        bias = neg_dmin * static_cast<float>(sc[group + 4] & 0x3FU);
                                    } else {
                                        int i = group - 4;
                                        scale = d_val * static_cast<float>((sc[8 + i] & 0x0FU) | ((sc[i] >> 2) & 0x30U));
                                        bias = neg_dmin * static_cast<float>((sc[8 + i] >> 4) | ((sc[i + 4] >> 2) & 0x30U));
                                    }
                                    const float val = scale * nibble + bias;
                                    slm_b[ki * TN + lid] = sycl::half(val);
                                }
                            } else {
                                for (int ki = 0; ki < TK; ++ki) {
                                    slm_b[ki * TN + lid] = sycl::half(0.0f);
                                }
                            }
                        }

                        it.barrier(sycl::access::fence_space::local_space);

                        auto b_ptr = slm_b.template get_multi_ptr<
                            sycl::access::decorated::no>().get();
                        jm::joint_matrix_load(sg, tB,
                            sycl::address_space_cast<
                                sycl::access::address_space::local_space,
                                sycl::access::decorated::no>(b_ptr),
                            static_cast<std::size_t>(TN));

                        jm::joint_matrix_mad(sg, tC, tA, tB, tC);

                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    // Store accumulator to SLM, then scatter to dst
                    auto c_slm_ptr = slm_c.template get_multi_ptr<
                        sycl::access::decorated::no>().get();
                    jm::joint_matrix_store(sg, tC,
                        sycl::address_space_cast<
                            sycl::access::address_space::local_space,
                            sycl::access::decorated::no>(c_slm_ptr),
                        static_cast<std::size_t>(TN),
                        jm::layout::row_major);

                    it.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        for (int m = 0; m < nc; ++m) {
                            for (int n = 0; n < TN; ++n) {
                                const int row = n_base + n;
                                if (row < nr) {
                                    y[m * y_stride + row] = c_slm_ptr[m * TN + n];
                                }
                            }
                        }
                    }
                });
        });
    }

    // iter10-rev2: no stream->wait() and no per-call free in the production
    // path. The SYCL in-order queue serializes subsequent ops naturally,
    // and the x_half buffer is owned by the per-device arena.
    //
    // iter11: when timing is enabled, force a wait + record so the chrono
    // clock measures actual kernel completion.
    if (xmx_timing_enabled) {
        stream->wait();
        const auto xmx_kern_t1 = std::chrono::steady_clock::now();
        ggml_sycl_debug_q4k_xmx_live_timing_record(
            device,
            std::chrono::duration<double, std::nano>(xmx_kern_t1 - xmx_kern_t0).count());
    }
}

/* gemma4-ipex iter12: pure fp16 GEMM path using pre-dequanted cached weights.

   Same dispatch scaffolding as ggml_sycl_debug_q4k_xmx_live (iter10), but
   the B matrix is loaded directly from the cached transposed fp16 buffer
   instead of being dequanted on-the-fly into SLM. The kernel body has no
   SLM writes for B, no per-K-step dequant, and ZERO barriers inside the
   K-loop -- each K-step is just (load A, load B, mad).

   Falls back to the iter10 SLM-dequant path when the FP16 cache is over
   its size cap (so the live dispatch still produces correct output for
   tensors that don't fit the cap). */

/* gemma4-ipex iter13: scalar diagnostic kernel that uses the iter12 fp16
   cache but reads it via a pure one-thread-per-row GEMV -- no tiles, no
   SLM, no joint_matrix. The only purpose is to isolate "is the cache
   data correct?" from "does the joint_matrix path work with the cache?".

   Iter12 spent 7 revs trying kernel variants on top of the cache without
   first verifying the cache itself. This is the step iter12 skipped.

   If this kernel produces correct output under GGML_SYCL_DEBUG_Q4K_XMX_FP16_SCALAR=1,
   the cache data is correct and iter12's bug is in the joint_matrix
   usage. If it produces garbage, the cache fill / upload path is wrong. */

bool ggml_sycl_debug_q4k_xmx_fp16_scalar_enabled() {
    static const bool enabled = std::getenv("GGML_SYCL_DEBUG_Q4K_XMX_FP16_SCALAR") != nullptr;
    return enabled;
}

bool ggml_sycl_debug_q4k_xmx_fp16_scalar_should_run(int ncols_x, int nrows_x, int ncols_y) {
    if (!ggml_sycl_debug_q4k_xmx_fp16_scalar_enabled()) {
        return false;
    }
    if (ncols_y != 8 || ncols_x != 5376 || nrows_x <= 0) {
        return false;
    }
    return true;
}

void ggml_sycl_debug_q4k_xmx_fp16_scalar_live(
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
    if (!ggml_sycl_debug_q4k_xmx_fp16_scalar_should_run(ncols_x, nrows_x, ncols_y)) {
        return;
    }

    {
        static bool announced[GGML_SYCL_MAX_DEVICES] = {};
        if (device >= 0 && device < GGML_SYCL_MAX_DEVICES && !announced[device]) {
            announced[device] = true;
            GGML_LOG_INFO(
                "Q4_K XMX FP16 SCALAR diagnostic active: device=%d ncols_x=%d nrows_x=%d ncols_y=%d dst_stride=%zu row_low=%d\n",
                device, ncols_x, nrows_x, ncols_y, dst_col_stride, row_low);
        }
    }

    // iter13-rev2: bypass the iter12 fp16 cache entirely. Read vx as
    // block_q4_K * and dequant on the fly inside the kernel. If this
    // path produces correct output, the iter12 fp16 cache has a bug
    // somewhere I can't see through the readback. If it's still
    // garbage, my scalar kernel structure is broken.
    const block_q4_K * vx_blocks = static_cast<const block_q4_K *>(vx);
    const int n_blocks_per_row_local = ncols_x / QK_K;

    float * const y_base = dst;
    const auto nr = nrows_x;
    const auto nc = ncols_y;
    const auto ne0_val = ncols_x;
    const auto x_stride = static_cast<size_t>(ncols_x);
    const auto y_stride = dst_col_stride;
    const auto x_base = x;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::range<1>(static_cast<std::size_t>(nr)),
            [=](sycl::id<1> row_id) {
                const int row = static_cast<int>(row_id[0]);
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
                float acc[8] = {};
                // Dequant on the fly for this row, one Q4K block at a time.
                for (int ib = 0; ib < n_blocks_per_row_local; ++ib) {
                    const block_q4_K & block =
                        vx_blocks[static_cast<std::size_t>(row) * n_blocks_per_row_local + ib];
                    const std::uint8_t * bytes =
                        reinterpret_cast<const std::uint8_t *>(&block);
                    const std::uint16_t d_raw = bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8);
                    const std::uint16_t dm_raw = bytes[2] | (static_cast<std::uint16_t>(bytes[3]) << 8);
                    const float d_val = fp16_to_fp32(d_raw);
                    const float neg_dmin = -fp16_to_fp32(dm_raw);
                    const std::uint8_t * sc = bytes + 4;

                    // Unpack 8 groups
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
                for (int c = 0; c < nc; ++c) {
                    y_base[c * y_stride + row] = acc[c];
                }
            });
    });
}

void ggml_sycl_debug_q4k_xmx_fp16_live(
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
    if (!ggml_sycl_debug_q4k_xmx_fp16_should_run(ncols_x, nrows_x, ncols_y)) {
        return;
    }

    {
        static bool announced[GGML_SYCL_MAX_DEVICES] = {};
        if (device >= 0 && device < GGML_SYCL_MAX_DEVICES && !announced[device]) {
            announced[device] = true;
            GGML_LOG_INFO(
                "Q4_K XMX FP16 LIVE active: device=%d ncols_x=%d nrows_x=%d ncols_y=%d dst_stride=%zu row_low=%d\n",
                device, ncols_x, nrows_x, ncols_y, dst_col_stride, row_low);
        }
    }

    const ggml_sycl_q4k_xmx_fp16_cached * fp16_entry =
        ggml_sycl_debug_q4k_xmx_fp16_cache_lookup(vx, nrows_x, ncols_x, device, stream);

    if (fp16_entry == nullptr) {
        // Cache over cap or allocation failed. Fall back to the iter10
        // XMX SLM-dequant path so the call still produces correct output.
        ggml_sycl_debug_q4k_xmx_live(
            vx, x, dst, device, ncols_x, row_low, nrows_x, ncols_y,
            dst_col_stride, stream);
        return;
    }

    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int TK = 16;
    constexpr int kSgSize = 8;

    const int ne0 = ncols_x;
    const int n_tiles_n = (nrows_x + TN - 1) / TN;

    // x_half scratch reuse: same per-device arena as iter10. The layout
    // is [TM][ne0] row-major, exactly what iter10 expects, so we share
    // it verbatim.
    const std::size_t x_half_size = static_cast<std::size_t>(TM) * ne0;
    sycl::half * x_half = nullptr;
    if (device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        auto & arena = g_q4k_xmx_scratch[device];
        if (arena.x_half == nullptr || arena.capacity < x_half_size) {
            if (arena.x_half != nullptr) {
                sycl::free(arena.x_half, *stream);
            }
            arena.x_half = sycl::malloc_device<sycl::half>(x_half_size, *stream);
            arena.capacity = x_half_size;
        }
        x_half = arena.x_half;
    } else {
        x_half = sycl::malloc_device<sycl::half>(x_half_size, *stream);
    }
    GGML_ASSERT(x_half != nullptr);

    {
        const auto nc = ncols_y;
        const auto x_stride = static_cast<size_t>(ncols_x);
        const auto ne0_cap = ne0;
        const auto x_src = x;
        const auto x_h = x_half;
        stream->submit([&](sycl::handler & h) {
            h.parallel_for(
                sycl::range<1>(static_cast<std::size_t>(TM) * ne0),
                [=](sycl::id<1> idx) {
                    const std::size_t flat = idx[0];
                    const std::size_t m = flat / static_cast<std::size_t>(ne0_cap);
                    const std::size_t k = flat % static_cast<std::size_t>(ne0_cap);
                    const float v = (static_cast<int>(m) < nc)
                        ? x_src[m * x_stride + k] : 0.0f;
                    x_h[m * static_cast<std::size_t>(ne0_cap) + k] = sycl::half(v);
                });
        });
    }

    // iter12-rev3: read fp16 weights from the cached [N][K] natural layout,
    // stage into SLM the same way iter10 does (dense [TK][TN] row_major),
    // then joint_matrix_load tB from SLM. This proves the cache is correct
    // and isolates the "bad joint_matrix_load pattern" from the cache
    // itself. Once this is correct, rev4 can attack the SLM staging
    // separately. The only win over iter10 here is: the SLM population
    // is a pure cached-read-and-write instead of a scalar dequant, so
    // each thread's inner loop is ~2-3x less arithmetic per K-step.
    {
        // Cache stores uint16, but the iter12-rev3 SLM staging writes
        // sycl::half directly. Reinterpret the uint16 storage as
        // sycl::half* for this path.
        const sycl::half * fp16_base =
            reinterpret_cast<const sycl::half *>(fp16_entry->fp16_dev);
        const auto x_h = x_half;
        float * const y_base = dst;
        const auto nr = nrows_x;
        const auto nc = ncols_y;
        const auto ne0_val = ne0;
        const auto y_stride = dst_col_stride;

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<sycl::half, 1> slm_b(
                sycl::range<1>(TK * TN), cgh);
            sycl::local_accessor<float, 1> slm_c(
                sycl::range<1>(TM * TN), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(static_cast<std::size_t>(n_tiles_n) * kSgSize),
                    sycl::range<1>(kSgSize)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSgSize)]] {
                    auto sg = it.get_sub_group();
                    const int tile_n = static_cast<int>(it.get_group(0));
                    const int n_base = tile_n * TN;
                    const int lid = static_cast<int>(it.get_local_linear_id());

                    jm::joint_matrix<sycl::sub_group, sycl::half,
                        jm::use::a, TM, TK, jm::layout::row_major> tA;
                    jm::joint_matrix<sycl::sub_group, sycl::half,
                        jm::use::b, TK, TN, jm::layout::row_major> tB;
                    jm::joint_matrix<sycl::sub_group, float,
                        jm::use::accumulator, TM, TN> tC;
                    jm::joint_matrix_fill(sg, tC, 0.0f);

                    for (int k_start = 0; k_start < ne0_val; k_start += TK) {
                        // Load A tile from pre-converted fp16 x (same as iter10).
                        auto a_ptr = sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(x_h + k_start);
                        jm::joint_matrix_load(sg, tA, a_ptr,
                            static_cast<std::size_t>(ne0_val));

                        // Stage B tile into SLM from the natural [N][K]
                        // cached fp16 layout. Each of the TN=8 threads
                        // copies one row of TK=16 fp16 values.
                        if (lid < TN) {
                            const int row = n_base + lid;
                            if (row < nr) {
                                const sycl::half * src_row =
                                    fp16_base +
                                    static_cast<std::size_t>(row) *
                                    static_cast<std::size_t>(ne0_val) +
                                    static_cast<std::size_t>(k_start);
                                for (int ki = 0; ki < TK; ++ki) {
                                    slm_b[ki * TN + lid] = src_row[ki];
                                }
                            } else {
                                for (int ki = 0; ki < TK; ++ki) {
                                    slm_b[ki * TN + lid] = sycl::half(0.0f);
                                }
                            }
                        }

                        it.barrier(sycl::access::fence_space::local_space);

                        auto b_ptr = slm_b.template get_multi_ptr<
                            sycl::access::decorated::no>().get();
                        jm::joint_matrix_load(sg, tB,
                            sycl::address_space_cast<
                                sycl::access::address_space::local_space,
                                sycl::access::decorated::no>(b_ptr),
                            static_cast<std::size_t>(TN));

                        jm::joint_matrix_mad(sg, tC, tA, tB, tC);

                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    // Store accumulator to SLM, then scatter to dst.
                    auto c_slm_ptr = slm_c.template get_multi_ptr<
                        sycl::access::decorated::no>().get();
                    jm::joint_matrix_store(sg, tC,
                        sycl::address_space_cast<
                            sycl::access::address_space::local_space,
                            sycl::access::decorated::no>(c_slm_ptr),
                        static_cast<std::size_t>(TN),
                        jm::layout::row_major);

                    it.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        for (int m = 0; m < nc; ++m) {
                            for (int n = 0; n < TN; ++n) {
                                const int row = n_base + n;
                                if (row < nr) {
                                    y_base[m * y_stride + row] = c_slm_ptr[m * TN + n];
                                }
                            }
                        }
                    }
                });
        });
    }
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
            case GGML_TYPE_Q4_K: {
                // gemma4-ipex iter8: skip stock MMVQ entirely when the ESIMD
                // live path will overwrite the result on the gated shape.
                // Avoids doing the same matmul twice on gated calls.
                // iter10: also skip when the XMX live path will overwrite.
                if (ggml_sycl_debug_q4k_esimd_live_should_run(
                        static_cast<int>(ne00),
                        static_cast<int>(row_diff),
                        static_cast<int>(src1_ncols))) {
                    GGML_SYCL_DEBUG("Skipping stock Q4K MMVQ: ESIMD live path owns this shape\n");
                    break;
                }
                if (ggml_sycl_debug_q4k_xmx_live_should_run(
                        static_cast<int>(ne00),
                        static_cast<int>(row_diff),
                        static_cast<int>(src1_ncols))) {
                    GGML_SYCL_DEBUG("Skipping stock Q4K MMVQ: XMX live path owns this shape\n");
                    break;
                }
                if (ggml_sycl_debug_q4k_xmx_fp16_should_run(
                        static_cast<int>(ne00),
                        static_cast<int>(row_diff),
                        static_cast<int>(src1_ncols))) {
                    GGML_SYCL_DEBUG("Skipping stock Q4K MMVQ: XMX FP16 live path owns this shape\n");
                    break;
                }
                if (ggml_sycl_debug_q4k_xmx_fp16_scalar_should_run(
                        static_cast<int>(ne00),
                        static_cast<int>(row_diff),
                        static_cast<int>(src1_ncols))) {
                    GGML_SYCL_DEBUG("Skipping stock Q4K MMVQ: XMX FP16 scalar diagnostic owns this shape\n");
                    break;
                }
                // gemma4-ipex iter9: optional stock timing on the gated shape
                // (same criteria as live path). Run with LIVE env unset so
                // stock actually executes.
                const bool stock_timed = ggml_sycl_debug_q4k_stock_timing_should_run(
                    static_cast<int>(ne00),
                    static_cast<int>(row_diff),
                    static_cast<int>(src1_ncols));
                const auto stock_t0 = stock_timed
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                if (stock_timed) {
                    stream->wait();
                    const auto stock_t1 = std::chrono::steady_clock::now();
                    ggml_sycl_debug_q4k_stock_timing_record(
                        id,
                        std::chrono::duration<double, std::nano>(stock_t1 - stock_t0).count());
                }
                break;
            }
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
        ggml_sycl_debug_q4k_xmx_live(
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
        ggml_sycl_debug_q4k_xmx_fp16_live(
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
        ggml_sycl_debug_q4k_xmx_fp16_scalar_live(
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
