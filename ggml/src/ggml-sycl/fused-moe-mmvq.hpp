//
// Fused Multi-Expert MMVQ Kernel for MoE Decode Path
//
// Replaces N individual kernel launches (one per active expert) with a single
// fused kernel that dispatches all experts in parallel using the grid's
// expert dimension.
//
// Target: Single-token decode (batch_size=1) with quantized experts.
// Intel Arc A770: sub_group_size=16 (WARP_SIZE=16)
//

#ifndef GGML_SYCL_FUSED_MOE_MMVQ_HPP
#define GGML_SYCL_FUSED_MOE_MMVQ_HPP

#include "common.hpp"

// Maximum number of active experts per token in a single fused dispatch.
#define FUSED_MOE_MAX_EXPERTS 128

// Function pointer type for fused MoE kernel launchers.
// Signature: (expert_ptrs, src1_q8, dst_ptrs, ncols, nrows, n_experts, stream)
typedef void (*fused_moe_kernel_fn_t)(
    const char * const *, const void *, float * const *,
    const int, const int, const int, dpct::queue_ptr);

// Resolve the fused MoE kernel launcher for a given quantization type.
// Returns nullptr if unsupported.
fused_moe_kernel_fn_t get_fused_moe_kernel(ggml_type type, bool use_reorder);

#endif // GGML_SYCL_FUSED_MOE_MMVQ_HPP
