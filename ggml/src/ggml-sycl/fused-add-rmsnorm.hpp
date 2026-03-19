//
// Fused ADD + RMSNorm kernel for SYCL.
//
// Replaces the pattern: dst_add = a + b; dst_norm = RMSNorm(dst_add)
// with a single kernel that reads a and b once, computes the sum and
// RMSNorm in a single pass.
//
// Saves 1 kernel launch per layer boundary (96 times per token in
// Qwen3-30B-A3B: 48 attn→MoE + 48 MoE→attn transitions).
//

#ifndef GGML_SYCL_FUSED_ADD_RMSNORM_HPP
#define GGML_SYCL_FUSED_ADD_RMSNORM_HPP

#include "common.hpp"

// Try to fuse ADD + RMS_NORM at position i in the node list.
// Returns 2 if fusion was applied (consumed 2 nodes), 0 otherwise.
int try_fused_add_rmsnorm(ggml_backend_sycl_context & ctx,
                          ggml_tensor ** nodes, int n_nodes, int i);

#endif // GGML_SYCL_FUSED_ADD_RMSNORM_HPP
