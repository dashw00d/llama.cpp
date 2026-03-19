//
// Fused softmax + argsort + get_rows for MoE top-k expert selection.
//
// Replaces: SOFT_MAX(logits) → ARGSORT(probs) → GET_ROWS(probs, top_k_ids)
// with a single kernel: fused_topk_select(logits) → (expert_ids, expert_weights)
//
// Saves 2 kernel launches per MoE layer (48 layers × 3 GPUs = 288 launches/token).
//

#ifndef GGML_SYCL_FUSED_TOPK_SELECT_HPP
#define GGML_SYCL_FUSED_TOPK_SELECT_HPP

#include "common.hpp"

// Try to fuse SOFT_MAX + ARGSORT + GET_ROWS at position i in the node list.
// Returns number of nodes consumed if fusion was applied, 0 otherwise.
int try_fused_topk_select(ggml_backend_sycl_context & ctx,
                          ggml_tensor ** nodes, int n_nodes, int i);

#endif // GGML_SYCL_FUSED_TOPK_SELECT_HPP
