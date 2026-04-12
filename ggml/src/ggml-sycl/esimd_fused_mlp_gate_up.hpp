// iter19: ESIMD fused MLP gate+up+SiLU+mul kernel, ported from
// gemma4-ipex/src/esimd/fused_mlp_gate_up.cpp.
//
// One thread per output row. Each thread:
//   1. Computes gate_dot = dot(x, gate_weight[row])  for all n_cols columns
//   2. Computes up_dot   = dot(x, up_weight[row])    for all n_cols columns
//   3. Writes y[row] = silu(gate_dot) * up_dot
// No intermediate memory writes -- gate and up results stay in registers.
//
// Eliminates 3 kernel launches and 2 intermediate memory writes per
// FFN block (gate matmul, up matmul, silu, mul).
//
// Both gate and up weights must be Q4K SoA payload+meta — same format
// as esimd_q4k_fused.hpp. Use the SoA cache from that header.

#ifndef GGML_SYCL_ESIMD_FUSED_MLP_GATE_UP_HPP
#define GGML_SYCL_ESIMD_FUSED_MLP_GATE_UP_HPP

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

struct ggml_sycl_esimd_fused_mlp_gate_up_args {
    const std::uint8_t * gate_payload;  // Q4K SoA
    const std::uint8_t * gate_meta;
    const std::uint8_t * up_payload;    // Q4K SoA
    const std::uint8_t * up_meta;
    int                  n_blocks_per_row;  // both gate and up share this
    int                  n_rows;            // output rows (e.g. 21504 for Gemma 4 31B FFN)
    const float *        x;
    float *              y;                 // silu(gate) * up output
    int                  n_cols;            // batch size
    std::size_t          x_col_stride;
    std::size_t          y_col_stride;
};

void ggml_sycl_esimd_fused_mlp_gate_up_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_fused_mlp_gate_up_args & args);

#endif // GGML_SYCL_ESIMD_FUSED_MLP_GATE_UP_HPP
