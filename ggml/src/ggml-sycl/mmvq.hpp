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


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Function pointer type for inner MMVQ kernel launchers.
// All have signature: (src0_data, src1_q8_data, dst, ncols, nrows, stream)
typedef void (*mmvq_kernel_fn_t)(const void *, const void *, float *, const int, const int, dpct::queue_ptr);

// Resolve the MMVQ kernel launcher for a given quantization type and reorder flag.
// Returns nullptr if the type is not supported by MMVQ.
mmvq_kernel_fn_t get_mmvq_kernel(ggml_type type, bool use_reorder);

#endif // GGML_SYCL_MMVQ_HPP
