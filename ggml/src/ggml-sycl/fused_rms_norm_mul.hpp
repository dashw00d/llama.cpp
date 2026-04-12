// iter21: fused RMS_NORM + MUL kernel, ported from the gemma4-ipex
// cleanroom (`try_dispatch_local_rms_norm` in
// src/ggml_sycl_clone_exports.cpp:5390-5444).
//
// Computes:   y[row][i] = x[row][i] * inv * (norm_weight ? norm_weight[i] : 1)
// where       inv = 1 / sqrt(mean(x[row][.]^2) + eps)
//
// One launch handles RMS_NORM and the optional MUL-by-broadcast-weight
// in a single kernel. Saves one kernel launch per (RMS_NORM, MUL) pair.
//
// Layout:
//   * One work-group per row.
//   * Work-group size = min(ne0, 256).
//   * Cooperative reduction in shared local memory for the sum-of-squares.
//   * Each thread strides the row by `wg`.
//
// Restrictions:
//   * src and dst must be contiguous f32, same shape.
//   * ne0 (row width) must be <= 8192. Larger rows aren't supported by
//     this kernel because the SLM scratch + reduction depth doesn't
//     scale beyond a single work-group. Fall back to stock for those.
//   * norm_weight (if non-null) must be a contiguous f32 vector of
//     length ne0 — i.e. the broadcast pattern (ne[1]==ne[2]==ne[3]==1).
//
// Gemma 1/2/3/4 quirk: the famous `(1+w)*x/rms` rescaling is baked
// into the GGUF weights at convert time, NOT applied at runtime.
// `build_norm` in `llama-graph.cpp` just calls `ggml_rms_norm` then
// `ggml_mul(cur, mw)`, no special-case. So this kernel can multiply
// by `norm_weight[i]` directly without any extra adjustment.

#ifndef GGML_SYCL_FUSED_RMS_NORM_MUL_HPP
#define GGML_SYCL_FUSED_RMS_NORM_MUL_HPP

#include <cstddef>
#include <sycl/sycl.hpp>

struct ggml_sycl_fused_rms_norm_mul_args {
    const float * src;          // input  [n_rows, ne0]
    float *       dst;          // output [n_rows, ne0]
    const float * norm_weight;  // [ne0] or nullptr
    std::size_t   ne0;
    std::size_t   n_rows;
    float         eps;
};

void ggml_sycl_fused_rms_norm_mul_dispatch(
    sycl::queue & queue,
    const ggml_sycl_fused_rms_norm_mul_args & args);

#endif // GGML_SYCL_FUSED_RMS_NORM_MUL_HPP
