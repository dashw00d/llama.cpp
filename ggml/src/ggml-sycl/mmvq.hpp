//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"

#include <vector>


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Iter16: graph-level QKV fusion entry points. Defined in mmvq.cpp.
struct SyclQ4KFusionRestoreEntry {
    ggml_tensor * node;
    int           original_op;
};
using SyclQ4KFusionRestoreList = std::vector<SyclQ4KFusionRestoreEntry>;

SyclQ4KFusionRestoreList ggml_sycl_q4k_qkv_prefuse_pass(
    sycl::queue * stream,
    int           device,
    ggml_cgraph * cgraph);

void ggml_sycl_q4k_qkv_restore_nodes(SyclQ4KFusionRestoreList & restore_list);

// Iter16 in-loop fusion: called from graph_compute_impl per node. If the
// node at index `i` starts a fusible Q4_K QKV triple (where K and V can
// be at indices i+k_off and i+v_off respectively, NOT necessarily i+1
// and i+2 since there are usually intermediate reshape/RoPE nodes
// between projections), this dispatches the fused kernel on the
// backend's stream and marks the K and V nodes as GGML_OP_NONE so the
// impl loop skips them when it reaches them. Returns true on a fused
// dispatch, false otherwise. The intermediate ops between Q and K/V
// run normally because they depend on the Q output written by the
// fused kernel. The restore list collects the (node, original_op)
// pairs that the caller must pass to ggml_sycl_q4k_qkv_restore_nodes
// after impl returns.
bool ggml_sycl_q4k_qkv_fuse_inline(
    ggml_backend_sycl_context * sycl_ctx,
    ggml_cgraph *               cgraph,
    int                         i,
    SyclQ4KFusionRestoreList *  restore_list);

// iter19: FFN gate+up+SiLU+mul fusion. Pattern: a Q4_K MUL_MAT (gate)
// followed by exactly one more Q4_K MUL_MAT (up) sharing src[1], then
// silu(gate), then mul(silu, up). Detected via tensor identity match.
// On success dispatches the fused ESIMD kernel and marks gate/up/silu/
// mul as GGML_OP_NONE so the impl loop skips them. Caller threads the
// same restore list as for QKV fusion.
bool ggml_sycl_q4k_mlp_fuse_inline(
    ggml_backend_sycl_context * sycl_ctx,
    ggml_cgraph *               cgraph,
    int                         i,
    SyclQ4KFusionRestoreList *  restore_list);

// iter21: fused RMS_NORM + MUL detection + dispatch.
// Pattern: a GGML_OP_RMS_NORM node followed (within window) by a
// GGML_OP_MUL where one src is the RMS_NORM output and the other src
// is a broadcast vector (ne[1]==ne[2]==ne[3]==1). On match dispatches
// the fused kernel writing into the MUL node's dst, marks both as
// GGML_OP_NONE. Optional standalone path: if no MUL match, the
// helper can still dispatch a plain RMS_NORM (the cleanroom does this
// in standalone mode but iter21 keeps stock for the standalone case
// to avoid touching the working path).
bool ggml_sycl_fused_rms_mul_inline(
    ggml_backend_sycl_context * sycl_ctx,
    ggml_cgraph *               cgraph,
    int                         i,
    SyclQ4KFusionRestoreList *  restore_list);

#endif // GGML_SYCL_MMVQ_HPP
