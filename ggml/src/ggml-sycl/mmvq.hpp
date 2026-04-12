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

#endif // GGML_SYCL_MMVQ_HPP
