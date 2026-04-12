// iter24: fused Q4K down-projection matmul + residual ADD kernel.
// One launch handles `down_matmul(gated_intermediate, down_weight) + residual`
// instead of dispatching the matmul and the elementwise add as two ops.
//
// Architectural shape: same cooperative-warp Q4K pattern as iter15's
// `run_q4k_scalar_cooperative` (one subgroup per output row, threads in
// the subgroup stride the K-dimension blocks, sub-group reduce, thread 0
// adds the residual + writes the final value). This gives us per-call
// kernel time on par with stock's cooperative MMVQ -- the iter17/18/19
// 1-thread-per-row regression doesn't apply because we use the same
// SG=16 layout stock uses.
//
// Saves one kernel launch per FFN block per layer (60 launches per
// graph_compute on Gemma 4 31B). Per-launch driver overhead is ~50us
// according to iter23 profiling, so the theoretical TG win is ~3 ms
// per decode step or ~1.5 % of TG. Small but measurable.

#ifndef GGML_SYCL_FUSED_DOWN_RESIDUAL_HPP
#define GGML_SYCL_FUSED_DOWN_RESIDUAL_HPP

#include <cstddef>
#include <sycl/sycl.hpp>

struct ggml_sycl_fused_down_residual_args {
    // Q4K weight: [n_rows_out, ncols_x] in `block_q4_K[n_rows_out * ncols_x/256]`
    // (standard ggml AoS, NOT the SoA payload+meta split — we use the
    // same direct AoS read as the iter15 cooperative kernel to avoid
    // the iter18 SoA-cache overhead for a small fusion savings target).
    const void *  vx;

    // Gated intermediate (FFN gate-up output): [ncols_x, n_cols] f32.
    const float * x;
    std::size_t   x_col_stride;   // stride between columns (= ncols_x for contiguous)

    // Residual to add elementwise to the matmul output. Same shape as dst.
    const float * residual;
    std::size_t   residual_col_stride;

    // Output: [n_rows_out, n_cols] f32. The fused kernel writes
    // dst[c, row] = down_matmul(x, weight)[c, row] + residual[c, row].
    float *       dst;
    std::size_t   dst_col_stride;

    int           ncols_x;        // K dimension (FFN intermediate dim, e.g. 21504)
    int           n_rows_out;     // M dimension (hidden dim, e.g. 5376)
    int           n_cols;         // N dimension (batch size, e.g. 8)
};

void ggml_sycl_fused_down_residual_dispatch(
    sycl::queue & queue,
    const ggml_sycl_fused_down_residual_args & args);

#endif // GGML_SYCL_FUSED_DOWN_RESIDUAL_HPP
