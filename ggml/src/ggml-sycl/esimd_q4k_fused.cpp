// iter18: ESIMD Q4_K fused multi-column matmul, ported from
// gemma4-ipex/src/esimd/linear_forward_q4k_fused_sycl.cpp.
// See esimd_q4k_fused.hpp for the contract and the SoA layout.

#include "esimd_q4k_fused.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace {

namespace esimd = sycl::ext::intel::esimd;

inline float esimd_q4k_fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    if (exp == 0x1fU) return 0.0f;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        float val = static_cast<float>(mant) * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000U) ? -val : val;
    }
    return sycl::bit_cast<float>(sign | ((exp + 112U) << 23) | (mant << 13));
}

void esimd_q4k_unpack_meta(
    const std::uint8_t * meta_ptr,
    float & scale_base,
    float & neg_min_base,
    float scales[8],
    float biases[8]) {

    scale_base = esimd_q4k_fp16_to_float(
        static_cast<std::uint16_t>(meta_ptr[0] | (meta_ptr[1] << 8)));
    neg_min_base = -esimd_q4k_fp16_to_float(
        static_cast<std::uint16_t>(meta_ptr[2] | (meta_ptr[3] << 8)));

    std::uint8_t raw_scales[8], raw_mins[8];
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

// Per-(device, src0_aos_ptr) cache of (payload, meta) device pointers.
// The cache lives for the lifetime of the process; entries are not
// freed because llama.cpp's weight buffers are also process-lifetime.
struct SoaCacheEntry {
    std::uint8_t * payload = nullptr;
    std::uint8_t * meta    = nullptr;
};

struct SoaCacheKey {
    int          device;
    const void * src0_aos;
    bool operator==(const SoaCacheKey & o) const noexcept {
        return device == o.device && src0_aos == o.src0_aos;
    }
};

struct SoaCacheKeyHash {
    std::size_t operator()(const SoaCacheKey & k) const noexcept {
        return std::hash<const void *>()(k.src0_aos) ^
               (static_cast<std::size_t>(k.device) * 0x9E3779B97F4A7C15ull);
    }
};

std::mutex                                                              g_soa_mu;
std::unordered_map<SoaCacheKey, SoaCacheEntry, SoaCacheKeyHash>          g_soa_cache;

} // namespace

// ──────────────────────────────────────────────────────────────────
// SoA cache: lazy AoS → SoA split, cached per (device, src0_aos)
// ──────────────────────────────────────────────────────────────────

bool ggml_sycl_esimd_q4k_get_or_build_soa(
    sycl::queue &       queue,
    int                 device,
    const void *        src0_data_aos,
    std::size_t         n_blocks,
    const std::uint8_t ** out_payload,
    const std::uint8_t ** out_meta) {

    if (out_payload == nullptr || out_meta == nullptr) return false;
    if (src0_data_aos == nullptr || n_blocks == 0) return false;

    SoaCacheKey key{device, src0_data_aos};

    {
        std::lock_guard<std::mutex> lk(g_soa_mu);
        auto it = g_soa_cache.find(key);
        if (it != g_soa_cache.end()) {
            *out_payload = it->second.payload;
            *out_meta    = it->second.meta;
            return true;
        }
    }

    // Allocate fresh SoA buffers on the device.
    const std::size_t payload_bytes = n_blocks * 128;
    const std::size_t meta_bytes    = n_blocks * 16;
    std::uint8_t * payload = sycl::malloc_device<std::uint8_t>(payload_bytes, queue);
    if (payload == nullptr) return false;
    std::uint8_t * meta    = sycl::malloc_device<std::uint8_t>(meta_bytes,    queue);
    if (meta == nullptr) {
        sycl::free(payload, queue);
        return false;
    }

    // Split kernel: each work item handles one block, copies the
    // first 16 bytes of block[i] (= block_q4_K::{d,dmin,scales}) into
    // meta[i*16 .. i*16+16] and the remaining 128 bytes (= block_q4_K::qs)
    // into payload[i*128 .. i*128+128]. The AoS source is the device
    // pointer src0_data_aos which points to N consecutive 144-byte
    // block_q4_K records.
    const std::uint8_t * aos = static_cast<const std::uint8_t *>(src0_data_aos);
    queue.submit([&](sycl::handler & cgh) {
        const std::uint8_t * aos_cap     = aos;
        std::uint8_t *       payload_cap = payload;
        std::uint8_t *       meta_cap    = meta;
        const std::size_t    nb          = n_blocks;
        cgh.parallel_for(
            sycl::range<1>(nb),
            [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];
                if (i >= nb) return;
                const std::uint8_t * src = aos_cap + i * 144;
                std::uint8_t *       m   = meta_cap + i * 16;
                std::uint8_t *       p   = payload_cap + i * 128;
                for (int b = 0; b < 16; ++b)  m[b] = src[b];
                for (int b = 0; b < 128; ++b) p[b] = src[16 + b];
            });
    }).wait();  // wait so the buffers are fully populated before the kernel reads them

    {
        std::lock_guard<std::mutex> lk(g_soa_mu);
        // Re-check in case another thread raced us. If so, prefer the
        // entry that's already there and free our duplicates.
        auto [it, inserted] = g_soa_cache.try_emplace(key, SoaCacheEntry{payload, meta});
        if (!inserted) {
            sycl::free(payload, queue);
            sycl::free(meta,    queue);
            *out_payload = it->second.payload;
            *out_meta    = it->second.meta;
            return true;
        }
        *out_payload = payload;
        *out_meta    = meta;
        return true;
    }
}

// ──────────────────────────────────────────────────────────────────
// ESIMD fused Q4K kernel: one thread per row, all columns at once
// ──────────────────────────────────────────────────────────────────

void ggml_sycl_esimd_q4k_fused_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_q4k_fused_args & args) {

    if (args.n_rows <= 0 || args.n_blocks_per_row <= 0 || args.n_cols <= 0) {
        return;
    }

    const std::size_t n_rows = static_cast<std::size_t>(args.n_rows);

    queue.submit([&](sycl::handler & cgh) {
        const auto payload_base = args.payload;
        const auto meta_base    = args.meta;
        const auto nbpr         = args.n_blocks_per_row;
        const auto x            = args.x;
        const auto y            = args.y;
        const auto nr           = args.n_rows;
        const auto nc           = args.n_cols;
        const auto x_stride     = args.x_col_stride;
        const auto y_stride     = args.y_col_stride;

        cgh.parallel_for(
            sycl::range<1>(n_rows),
            [=](sycl::id<1> row_id) SYCL_ESIMD_KERNEL {
                const int row = static_cast<int>(row_id[0]);
                if (row >= nr) return;

                float acc[32] = {};  // iter22: widened from [8] to allow ncols_y up to 32 (npl 32)

                for (int ib = 0; ib < nbpr; ++ib) {
                    const std::size_t block_idx =
                        static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(nbpr) +
                        static_cast<std::size_t>(ib);
                    const std::uint8_t * payload = payload_base + block_idx * 128;
                    const std::uint8_t * meta_ptr = meta_base + block_idx * 16;

                    float scale_base, neg_min_base;
                    float scales[8], biases[8];
                    esimd_q4k_unpack_meta(meta_ptr, scale_base, neg_min_base,
                                          scales, biases);

                    // Single 128-byte load of the entire Q4K block payload.
                    // ESIMD's block_load<uint8_t, 128> emits a v128i8 load
                    // (the same instruction IPEX uses) and is ~6.5x faster
                    // than 4× block_load<32>.
                    esimd::simd<std::uint32_t, 128> full_block =
                        esimd::convert<std::uint32_t>(
                            esimd::block_load<std::uint8_t, 128>(
                                const_cast<std::uint8_t *>(payload)));

                    for (int pair = 0; pair < 4; ++pair) {
                        const int lg = pair * 2;
                        const int hg = lg + 1;
                        const float ls = scales[lg], hs = scales[hg];
                        const float lb = biases[lg], hb = biases[hg];

                        esimd::simd<std::uint32_t, 32> packed_u32 =
                            full_block.template select<32, 1>(pair * 32);

                        esimd::simd<float, 32> w_lo =
                            esimd::convert<float>(packed_u32 & 0x0FU) * ls + lb;
                        esimd::simd<float, 32> w_hi =
                            esimd::convert<float>(packed_u32 >> 4) * hs + hb;

                        const std::size_t x_lo_off =
                            static_cast<std::size_t>(ib) * 256 + lg * 32;
                        const std::size_t x_hi_off =
                            static_cast<std::size_t>(ib) * 256 + hg * 32;

                        esimd::simd<float, 16> wl0 = w_lo.template select<16, 1>(0);
                        esimd::simd<float, 16> wl1 = w_lo.template select<16, 1>(16);
                        esimd::simd<float, 16> wh0 = w_hi.template select<16, 1>(0);
                        esimd::simd<float, 16> wh1 = w_hi.template select<16, 1>(16);

                        for (int c = 0; c < nc; ++c) {
                            const float * xc = x + c * x_stride;

                            esimd::simd<float, 16> x_lo_0 =
                                esimd::block_load<float, 16>(
                                    const_cast<float *>(xc + x_lo_off));
                            esimd::simd<float, 16> x_lo_1 =
                                esimd::block_load<float, 16>(
                                    const_cast<float *>(xc + x_lo_off + 16));
                            esimd::simd<float, 16> x_hi_0 =
                                esimd::block_load<float, 16>(
                                    const_cast<float *>(xc + x_hi_off));
                            esimd::simd<float, 16> x_hi_1 =
                                esimd::block_load<float, 16>(
                                    const_cast<float *>(xc + x_hi_off + 16));

                            acc[c] += esimd::reduce<float>(
                                wl0 * x_lo_0 + wl1 * x_lo_1, std::plus<>());
                            acc[c] += esimd::reduce<float>(
                                wh0 * x_hi_0 + wh1 * x_hi_1, std::plus<>());
                        }
                    }
                }

                for (int c = 0; c < nc; ++c) {
                    y[c * y_stride + row] = acc[c];
                }
            });
    });
}
