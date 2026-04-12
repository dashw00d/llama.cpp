//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Flash attention disabled for this fork. The upstream implementation requires
// sycl::ext::oneapi::work_group_static (oneAPI >= 2025.1) and the container
// ships oneAPI 2025.0.4. When Flash Attention is requested, the SYCL backend
// reports it as unsupported and llama.cpp falls back to the unfused attention
// path (KQ matmul + softmax + KQV matmul), which runs on SYCL.
//
// To restore the upstream implementation, upgrade the container or port the
// body of fattn-tile.hpp / fattn-vec.hpp to use group_local_memory_for_overwrite.

#include "common.hpp"
#include "fattn.hpp"

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & /*ctx*/, ggml_tensor * /*dst*/) {
    GGML_ABORT("fattn disabled (oneAPI 2025.0.4) — llama.cpp should route around via -fa off or the unsupported path");
}

bool ggml_sycl_flash_attn_ext_supported(int /*device*/, const ggml_tensor * /*dst*/) {
    return false;
}
