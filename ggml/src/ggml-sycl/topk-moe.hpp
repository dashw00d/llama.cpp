#pragma once

#include <sycl/sycl.hpp>
#include "common.hpp"
#include "ggml.h"

struct ggml_sycl_topk_moe_args {
    bool sigmoid;
    bool delayed_softmax;
};

void ggml_sycl_op_topk_moe(ggml_backend_sycl_context & ctx,
                            const ggml_tensor * logits,
                            ggml_tensor * weights,
                            ggml_tensor * ids,
                            const ggml_tensor * clamp,
                            const ggml_tensor * scale,
                            const ggml_tensor * bias,
                            const ggml_sycl_topk_moe_args & args);

bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op,
                                    const ggml_tensor * weights,
                                    const ggml_tensor * logits,
                                    const ggml_tensor * ids);
