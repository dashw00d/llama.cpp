// iter27: ESIMD Q6_K multi-column fused matmul kernel, ported from
// gemma4-ipex/src/esimd/linear_forward_q6k_sycl.cpp.
//
// One thread per output row. Each thread computes the full Q6K dot
// product against all `n_cols` input columns. Supports two weight
// layouts:
//   * AoS: standard ggml block_q6_K[] (210 bytes/block, fields stored
//     as [ql(128), qh(64), sc(16), d(2)] within each block). Pass the
//     AoS pointer as `raw_blocks` and set the SoA pointers to nullptr.
//   * SoA: IPEX-style split where each field is stored contiguously
//     for all blocks:
//       ql_base[N*128] + qh_base[N*64] + sc_base[N*16] + d_base[N*2]
//     Pass the SoA pointers; `raw_blocks` is ignored.
//
// Per-block math:
//   - Load d (fp16 → f32).
//   - For each half (2) × each segment (2) × 4 groups of 16 values:
//     - Unpack 6-bit quants (4-bit low from ql + 2-bit high from qh).
//     - Scale by d * sc[half*8 + seg + {0,2,4,6}] - 32 (zero point).
//     - Inner product with the matching 16 elements of x per column.
//
// Optional inline residual add: if `residual` is non-null, the kernel
// computes `y[c, row] = matmul_dot + residual[c, row]`.
//
// Not yet wired into any dispatch helper — in-tree as dormant
// infrastructure until a Q6K-for-V model is the bench target. None of
// Gemma 4 31B Q4_K_M or Qwen3-32B Q4_K_M use Q6K for the V projection
// (both have Q4K V), so iter27 ports the code but doesn't activate it.

#ifndef GGML_SYCL_ESIMD_Q6K_FUSED_HPP
#define GGML_SYCL_ESIMD_Q6K_FUSED_HPP

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

struct ggml_sycl_esimd_q6k_fused_args {
    // AoS mode: set this to the standard ggml block_q6_K[] pointer.
    // Leave the SoA pointers null.
    const std::uint8_t * raw_blocks;

    // SoA mode: set these four pointers to the split Q6K fields.
    // Leave raw_blocks null.
    const std::uint8_t * ql_base;  // n_blocks × 128 bytes
    const std::uint8_t * qh_base;  // n_blocks × 64 bytes
    const std::int8_t  * sc_base;  // n_blocks × 16 bytes (signed)
    const std::uint8_t * d_base;   // n_blocks × 2 bytes (fp16)

    int                  n_blocks_per_row;
    const float *        x;
    float *              y;
    int                  n_rows;
    int                  n_cols;       // up to 8 supported by the acc[] array
    std::size_t          x_col_stride;
    std::size_t          y_col_stride;

    // Optional inline residual add. If null, no residual.
    const float *        residual;
    std::size_t          residual_col_stride;
};

void ggml_sycl_esimd_q6k_fused_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_q6k_fused_args & args);

#endif // GGML_SYCL_ESIMD_Q6K_FUSED_HPP
