// iter19: ESIMD fused MLP gate+up+SiLU+mul kernel.
// Ported from gemma4-ipex/src/esimd/fused_mlp_gate_up.cpp.
// See esimd_fused_mlp_gate_up.hpp for the contract.

#include "esimd_fused_mlp_gate_up.hpp"

#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace {

namespace esimd_ns = sycl::ext::intel::esimd;

inline float mlp_gate_up_fp16(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    if (exp == 0 || exp == 0x1fU) return 0.0f;
    return sycl::bit_cast<float>(sign | ((exp + 112) << 23) | (mant << 13));
}

// Q4K row dot product for all n_cols columns -- vectorized block_load
// version. Same shape as the qkv version but renamed to avoid ODR
// conflict if both files are linked together.
inline void mlp_q4k_row_dot(
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

        float sb = mlp_gate_up_fp16(meta[0] | (static_cast<std::uint16_t>(meta[1]) << 8));
        float nm = -mlp_gate_up_fp16(meta[2] | (static_cast<std::uint16_t>(meta[3]) << 8));
        const std::uint8_t * sc = meta + 4;

        for (int pair = 0; pair < 4; ++pair) {
            int lg = pair * 2, hg = lg + 1;
            float ls, hs, lb, hb;
            if (lg < 4) { ls = sb*(sc[lg]&0x3FU); lb = nm*(sc[lg+4]&0x3FU); }
            else { int i=lg-4; ls = sb*((sc[8+i]&0x0FU)|((sc[i]>>2)&0x30U)); lb = nm*((sc[8+i]>>4)|((sc[i+4]>>2)&0x30U)); }
            if (hg < 4) { hs = sb*(sc[hg]&0x3FU); hb = nm*(sc[hg+4]&0x3FU); }
            else { int i=hg-4; hs = sb*((sc[8+i]&0x0FU)|((sc[i]>>2)&0x30U)); hb = nm*((sc[8+i]>>4)|((sc[i+4]>>2)&0x30U)); }

            auto packed = esimd_ns::convert<std::uint32_t>(
                esimd_ns::block_load<std::uint8_t, 32>(
                    const_cast<std::uint8_t *>(payload + pair * 32)));
            auto w_lo = esimd_ns::convert<float>(packed & 0x0FU) * ls + lb;
            auto w_hi = esimd_ns::convert<float>(packed >> 4) * hs + hb;

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

} // anonymous namespace

void ggml_sycl_esimd_fused_mlp_gate_up_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_fused_mlp_gate_up_args & args) {

    if (args.n_rows <= 0 || args.n_blocks_per_row <= 0 || args.n_cols <= 0) {
        return;
    }

    const std::size_t n_rows = static_cast<std::size_t>(args.n_rows);

    queue.submit([=](sycl::handler & cgh) {
        const auto a = args;
        cgh.parallel_for(
            sycl::range<1>(n_rows),
            [=](sycl::id<1> row_id) SYCL_ESIMD_KERNEL {
                const int row = static_cast<int>(row_id[0]);
                if (row >= a.n_rows) return;

                // iter22: widened from [8] to allow ncols_y up to 32 (npl 32)
                float gate_acc[32] = {};
                float up_acc[32]   = {};

                mlp_q4k_row_dot(
                    a.gate_payload, a.gate_meta, a.n_blocks_per_row, row,
                    a.x, a.x_col_stride, a.n_cols, gate_acc);

                mlp_q4k_row_dot(
                    a.up_payload, a.up_meta, a.n_blocks_per_row, row,
                    a.x, a.x_col_stride, a.n_cols, up_acc);

                // Fused epilogue: activation(gate) * up — no intermediate write.
                // Activation is selected per-call via args.activation:
                //   SILU : silu(g) = g / (1 + exp(-g))     [Qwen3, Llama3]
                //   GELU : gelu_tanh(g) = 0.5*g*(1+tanh(sqrt(2/pi)*g*(1+0.044715*g*g)))  [Gemma]
                // ESIMD context disallows sycl::tanh; implement via exp:
                //   tanh(x) = 1 - 2 / (exp(2x) + 1)
                // sycl::exp IS allowed in ESIMD (the silu path uses it).
                const bool use_gelu = (a.activation == GGML_SYCL_ESIMD_MLP_GELU);
                for (int c = 0; c < a.n_cols; ++c) {
                    const float g = gate_acc[c];
                    float act;
                    if (use_gelu) {
                        const float gg = g * g;
                        // 0.79788456f = sqrt(2/pi); 0.044715f = GELU_COEF_A
                        const float u = 0.79788456f * g * (1.0f + 0.044715f * gg);
                        const float th = 1.0f - 2.0f / (sycl::exp(2.0f * u) + 1.0f);
                        act = 0.5f * g * (1.0f + th);
                    } else {
                        act = g / (1.0f + sycl::exp(-g));
                    }
                    a.y[c * a.y_col_stride + row] = act * up_acc[c];
                }
            });
    });
}
