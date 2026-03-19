//
// Fused expert aggregation kernel for MoE layers.
// Replaces: MUL(experts, weights) + VIEW×N + ADD×(N-1) → single kernel.
//
// Pattern detected in compute graph:
//   MUL node (op=MUL, name contains "ffn_moe_weighted")
//     src[0] = experts [n_embd, n_expert_used, n_tokens]
//     src[1] = weights [1, n_expert_used, n_tokens]
//   Followed by VIEW nodes (no-ops) and a chain of ADD nodes that sum the expert slices.
//
// The fused kernel computes:
//   output[embd, token] = sum_k( weights[k, token] * experts[embd, k, token] )
// in a single launch, replacing N+1 kernel launches with 1.
//
#pragma once

#include "common.hpp"

// Try to detect and execute the fused expert aggregation pattern starting at node index i.
// Returns the number of nodes consumed (to skip) if fusion was applied, or 0 if not applicable.
//
// The pattern is:
//   nodes[i]   = MUL (experts × weights)  — "ffn_moe_weighted"
//   nodes[i+1..i+N] = VIEW (no-ops, slicing expert dim)
//   nodes[i+N+1..i+2N-1] = ADD chain (summing expert slices)
//
// We detect this by:
//   1. Node i is MUL with name containing "ffn_moe_weighted"
//   2. The MUL's src[0] has shape [n_embd, n_expert_used, n_tokens] with n_expert_used > 1
//   3. After skipping VIEWs, there's a chain of (n_expert_used - 1) ADD nodes
//   4. The last ADD's output goes to the same buffer as the expected moe_out
int try_fused_expert_agg(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int n_nodes, int i);
