// iter21: fused RMS_NORM + MUL kernel.
// Ported verbatim from the gemma4-ipex cleanroom
// (`try_dispatch_local_rms_norm` in
// src/ggml_sycl_clone_exports.cpp:5390-5444).
// See fused_rms_norm_mul.hpp for the contract.

#include "fused_rms_norm_mul.hpp"

#include <algorithm>

void ggml_sycl_fused_rms_norm_mul_dispatch(
    sycl::queue & queue,
    const ggml_sycl_fused_rms_norm_mul_args & args) {

    if (args.src == nullptr || args.dst == nullptr) return;
    if (args.ne0 == 0 || args.n_rows == 0) return;
    if (args.ne0 > 8192) return;  // hard cap from cleanroom -- larger needs WG split

    const std::size_t ne0    = args.ne0;
    const std::size_t n_rows = args.n_rows;
    const float       eps    = args.eps;
    const std::size_t wg     = std::min<std::size_t>(ne0, 256);
    const float *     src    = args.src;
    float *           dst    = args.dst;
    const float *     d_nw   = args.norm_weight;

    // iter28 (fix B4): `[&]` → `[=]` to avoid the iter17-class dangling-
    // capture bug. Matches the fix applied to esimd_q4k_fused.cpp:184.
    queue.submit([=](sycl::handler & h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(wg), h);
        h.parallel_for(
            sycl::nd_range<1>(n_rows * wg, wg),
            [=](sycl::nd_item<1> it) {
                const std::size_t row  = it.get_group(0);
                const std::size_t lid  = it.get_local_id(0);
                const std::size_t base = row * ne0;

                // Sum of squares for this row, strided across the work-group.
                float ss = 0.0f;
                for (std::size_t i = lid; i < ne0; i += wg) {
                    ss += src[base + i] * src[base + i];
                }

                // iter28 (fix B5): the cleanroom used a tree reduction
                // `for (s = wg/2; s > 0; s >>= 1)` which drops partials
                // if `wg` is not a power of two. `wg = min(ne0, 256)`
                // is safe for Gemma 4 (ne0 ∈ {256, 5376→256}) and any
                // pow2 head_dim, but breaks for models with head_dim
                // 192 / non-pow2 ne0 < 256. Use sycl::reduce_over_group
                // which handles arbitrary work-group sizes correctly.
                const float total = sycl::reduce_over_group(
                    it.get_group(), ss, sycl::plus<float>());
                // Broadcast via SLM to avoid each thread recomputing
                // the sqrt (also keeps the scratch buffer usage we
                // already allocate for ABI stability).
                if (lid == 0) scratch[0] = total;
                it.barrier(sycl::access::fence_space::local_space);

                const float inv = 1.0f / sycl::sqrt(scratch[0] / static_cast<float>(ne0) + eps);

                // Apply inv (and the optional broadcast weight) elementwise.
                if (d_nw != nullptr) {
                    for (std::size_t i = lid; i < ne0; i += wg) {
                        dst[base + i] = src[base + i] * inv * d_nw[i];
                    }
                } else {
                    for (std::size_t i = lid; i < ne0; i += wg) {
                        dst[base + i] = src[base + i] * inv;
                    }
                }
            });
    });
}
