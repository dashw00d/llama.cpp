// iter19: ESIMD fused Q+K+V projection (true 1-launch fusion), ported
// from gemma4-ipex/src/esimd/fused_qkv_proj.cpp.
//
// One kernel launch handles all three projections via thread partition:
//   threads [0, q_n_rows)                       -> q_proj (Q4K)
//   threads [q_n_rows, q_n_rows + k_n_rows)     -> k_proj (Q4K)
//   threads [q_n_rows + k_n_rows, total)        -> v_proj (Q4K or Q6K)
//
// Each thread computes one output row for all n_cols columns. The
// activation `x` is shared by all three projections. SoA Q4K layout
// (payload[N*128] + meta[N*16]) — see esimd_q4k_fused.hpp for details.
//
// For Q4K-only V projections (Gemma 4 31B Q4_K_M has Q4K V), set
// v_is_q4k = true and pass v_payload + v_meta in v_raw_blocks +
// v_meta. Q6K V is supported but the SoA buffers must be built
// separately (not yet ported in iter19; the Q6K path is dormant).

#ifndef GGML_SYCL_ESIMD_FUSED_QKV_PROJ_HPP
#define GGML_SYCL_ESIMD_FUSED_QKV_PROJ_HPP

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

struct ggml_sycl_esimd_fused_qkv_args {
    // Q projection (Q4K SoA)
    const std::uint8_t * q_payload;
    const std::uint8_t * q_meta;
    int                  q_nbpr;
    int                  q_n_rows;
    float *              q_out;
    std::size_t          q_y_col_stride;

    // K projection (Q4K SoA)
    const std::uint8_t * k_payload;
    const std::uint8_t * k_meta;
    int                  k_nbpr;
    int                  k_n_rows;
    float *              k_out;
    std::size_t          k_y_col_stride;

    // V projection (Q4K SoA when v_is_q4k=true; Q6K SoA via v_ql_base
    // etc when v_is_q4k=false — Q6K path is dormant in iter19)
    const std::uint8_t * v_raw_blocks;   // Q4K: payload base; Q6K: legacy raw
    const std::uint8_t * v_meta;         // Q4K only
    const std::uint8_t * v_ql_base;      // Q6K only
    const std::uint8_t * v_qh_base;      // Q6K only
    const std::int8_t  * v_sc_base;      // Q6K only
    const std::uint8_t * v_d_base;       // Q6K only
    int                  v_nbpr;
    int                  v_n_rows;
    float *              v_out;
    std::size_t          v_y_col_stride;
    bool                 v_is_q4k;

    // Shared input activation
    const float * x;
    int           ne0;
    int           n_cols;
    std::size_t   x_col_stride;
};

void ggml_sycl_esimd_fused_qkv_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_fused_qkv_args & args);

#endif // GGML_SYCL_ESIMD_FUSED_QKV_PROJ_HPP
