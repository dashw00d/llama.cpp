// iter19: ESIMD fused Q+K+V projection kernel.
// Ported from gemma4-ipex/src/esimd/fused_qkv_proj.cpp.
// See esimd_fused_qkv_proj.hpp for the contract.

#include "esimd_fused_qkv_proj.hpp"

#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace {

namespace esimd_ns = sycl::ext::intel::esimd;

inline float qkv_q4k_fp16(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    if (exp == 0x1fU) return 0.0f;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        float val = static_cast<float>(mant) * (1.0f / 1024.0f) * (1.0f / 16384.0f);
        return (h & 0x8000U) ? -val : val;
    }
    return sycl::bit_cast<float>(sign | ((exp + 112) << 23) | (mant << 13));
}

// Q4K row dot product for all n_cols columns. ESIMD vectorized loads.
inline void qkv_q4k_row_dot(
    const std::uint8_t * payload_base,
    const std::uint8_t * meta_base,
    int nbpr,
    int row,
    const float * x,
    std::size_t x_col_stride,
    int n_cols,
    float * acc) {

    for (int ib = 0; ib < nbpr; ++ib) {
        const std::size_t block_idx = static_cast<std::size_t>(row) * nbpr + ib;
        const std::uint8_t * payload = payload_base + block_idx * 128;
        const std::uint8_t * meta    = meta_base    + block_idx * 16;

        float sb = qkv_q4k_fp16(meta[0] | (static_cast<std::uint16_t>(meta[1]) << 8));
        float nm = -qkv_q4k_fp16(meta[2] | (static_cast<std::uint16_t>(meta[3]) << 8));
        const std::uint8_t * sc = meta + 4;

        for (int pair = 0; pair < 4; ++pair) {
            int lg = pair * 2, hg = lg + 1;
            float ls, hs, lb, hb;
            if (lg < 4) { ls = sb*(sc[lg]&0x3FU); lb = nm*(sc[lg+4]&0x3FU); }
            else { int i=lg-4; ls = sb*((sc[8+i]&0x0FU)|((sc[i]>>2)&0x30U)); lb = nm*((sc[8+i]>>4)|((sc[i+4]>>2)&0x30U)); }
            if (hg < 4) { hs = sb*(sc[hg]&0x3FU); hb = nm*(sc[hg+4]&0x3FU); }
            else { int i=hg-4; hs = sb*((sc[8+i]&0x0FU)|((sc[i]>>2)&0x30U)); hb = nm*((sc[8+i]>>4)|((sc[i+4]>>2)&0x30U)); }

            esimd_ns::simd<std::uint8_t, 32> packed =
                esimd_ns::block_load<std::uint8_t, 32>(
                    const_cast<std::uint8_t *>(payload + pair * 32));
            auto packed_u32 = esimd_ns::convert<std::uint32_t>(packed);
            auto w_lo = esimd_ns::convert<float>(packed_u32 & 0x0FU) * ls + lb;
            auto w_hi = esimd_ns::convert<float>(packed_u32 >> 4) * hs + hb;

            const std::size_t x_lo = static_cast<std::size_t>(ib) * 256 + lg * 32;
            const std::size_t x_hi = static_cast<std::size_t>(ib) * 256 + hg * 32;

            for (int c = 0; c < n_cols; ++c) {
                const float * xc = x + c * x_col_stride;
                auto xl0 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + x_lo));
                auto xl1 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + x_lo + 16));
                auto xh0 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + x_hi));
                auto xh1 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + x_hi + 16));

                auto wl0 = w_lo.template select<16, 1>(0);
                auto wl1 = w_lo.template select<16, 1>(16);
                auto wh0 = w_hi.template select<16, 1>(0);
                auto wh1 = w_hi.template select<16, 1>(16);

                acc[c] += esimd_ns::reduce<float>(wl0*xl0 + wl1*xl1, std::plus<>());
                acc[c] += esimd_ns::reduce<float>(wh0*xh0 + wh1*xh1, std::plus<>());
            }
        }
    }
}

// Q6K row dot product (kept for the Q6K V case; not exercised by
// Gemma 4 31B Q4_K_M which uses Q4K V).
inline void qkv_q6k_row_dot(
    const std::uint8_t * ql_base,
    const std::uint8_t * qh_base,
    const std::int8_t  * sc_base,
    const std::uint8_t * d_base,
    int nbpr,
    int row,
    const float * x,
    std::size_t x_col_stride,
    int n_cols,
    float * acc) {

    for (int ib = 0; ib < nbpr; ++ib) {
        const std::size_t block_idx =
            static_cast<std::size_t>(row) * nbpr + ib;
        const std::uint8_t * ql = ql_base + block_idx * 128;
        const std::uint8_t * qh = qh_base + block_idx * 64;
        const std::int8_t  * sc = sc_base + block_idx * 16;
        std::uint16_t d_raw = d_base[block_idx * 2] | (static_cast<std::uint16_t>(d_base[block_idx * 2 + 1]) << 8);
        float d_val = qkv_q4k_fp16(d_raw);

        for (int half = 0; half < 2; ++half) {
            const std::uint8_t * ql_h = ql + half * 64;
            const std::uint8_t * qh_h = qh + half * 32;
            const std::int8_t  * sc_h = sc + half * 8;
            for (int seg = 0; seg < 2; ++seg) {
                int base = seg * 16;
                float s0 = d_val * float(sc_h[seg+0]);
                float s2 = d_val * float(sc_h[seg+2]);
                float s4 = d_val * float(sc_h[seg+4]);
                float s6 = d_val * float(sc_h[seg+6]);

                auto ql0 = esimd_ns::convert<std::uint32_t>(
                    esimd_ns::block_load<std::uint8_t, 16>(
                        const_cast<std::uint8_t*>(ql_h + base)));
                auto ql1 = esimd_ns::convert<std::uint32_t>(
                    esimd_ns::block_load<std::uint8_t, 16>(
                        const_cast<std::uint8_t*>(ql_h + base + 32)));
                auto qh0 = esimd_ns::convert<std::uint32_t>(
                    esimd_ns::block_load<std::uint8_t, 16>(
                        const_cast<std::uint8_t*>(qh_h + base)));

                auto w1 = (esimd_ns::convert<float>((ql0 & 0x0FU) | ((qh0 & 0x03U) << 4)) - 32.0f) * s0;
                auto w2 = (esimd_ns::convert<float>((ql1 & 0x0FU) | (((qh0>>2) & 0x03U) << 4)) - 32.0f) * s2;
                auto w3 = (esimd_ns::convert<float>((ql0 >> 4) | (((qh0>>4) & 0x03U) << 4)) - 32.0f) * s4;
                auto w4 = (esimd_ns::convert<float>((ql1 >> 4) | ((qh0 >> 6) << 4)) - 32.0f) * s6;

                std::size_t x_off = static_cast<std::size_t>(ib) * 256 + half * 128 + base;
                for (int c = 0; c < n_cols; ++c) {
                    const float * xc = x + c * x_col_stride + x_off;
                    auto x0 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc));
                    auto x1 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + 32));
                    auto x2 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + 64));
                    auto x3 = esimd_ns::block_load<float, 16>(const_cast<float*>(xc + 96));
                    acc[c] += esimd_ns::reduce<float>(w1*x0 + w2*x1 + w3*x2 + w4*x3, std::plus<>());
                }
            }
        }
    }
}

} // anonymous namespace

void ggml_sycl_esimd_fused_qkv_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_fused_qkv_args & args) {

    if (args.ne0 <= 0 || args.n_cols <= 0) return;

    const int total_rows = args.q_n_rows + args.k_n_rows + args.v_n_rows;
    if (total_rows <= 0) return;

    queue.submit([=](sycl::handler & cgh) {
        const auto a = args;
        cgh.parallel_for(
            sycl::range<1>(static_cast<std::size_t>(total_rows)),
            [=](sycl::id<1> tid) SYCL_ESIMD_KERNEL {
                const int t = static_cast<int>(tid[0]);
                float acc[32] = {};  // iter22: widened from [8] to allow ncols_y up to 32 (npl 32)

                if (t < a.q_n_rows) {
                    qkv_q4k_row_dot(
                        a.q_payload, a.q_meta, a.q_nbpr, t,
                        a.x, a.x_col_stride, a.n_cols, acc);
                    for (int c = 0; c < a.n_cols; ++c)
                        a.q_out[c * a.q_y_col_stride + t] = acc[c];

                } else if (t < a.q_n_rows + a.k_n_rows) {
                    const int row = t - a.q_n_rows;
                    qkv_q4k_row_dot(
                        a.k_payload, a.k_meta, a.k_nbpr, row,
                        a.x, a.x_col_stride, a.n_cols, acc);
                    for (int c = 0; c < a.n_cols; ++c)
                        a.k_out[c * a.k_y_col_stride + row] = acc[c];

                } else {
                    const int row = t - a.q_n_rows - a.k_n_rows;
                    if (a.v_is_q4k) {
                        qkv_q4k_row_dot(
                            a.v_raw_blocks, a.v_meta, a.v_nbpr, row,
                            a.x, a.x_col_stride, a.n_cols, acc);
                    } else {
                        qkv_q6k_row_dot(
                            a.v_ql_base, a.v_qh_base, a.v_sc_base, a.v_d_base,
                            a.v_nbpr, row,
                            a.x, a.x_col_stride, a.n_cols, acc);
                    }
                    for (int c = 0; c < a.n_cols; ++c)
                        a.v_out[c * a.v_y_col_stride + row] = acc[c];
                }
            });
    });
}
