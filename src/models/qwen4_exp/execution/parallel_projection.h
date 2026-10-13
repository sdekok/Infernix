#pragma once

#include "core/parallel_partition.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/projection_fp32.h"
#include <stdexcept>

namespace infernix::models::qwen4_exp::execution {
inline void parallel_projection(const Tensor& x, const Weight& w, Tensor& out,
                                ops::LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    if (out.dtype == DType::BF16) {
        ops::linear(x, w, out, policy, workspace, stream);
    } else if (out.dtype != DType::FP32) {
        throw std::invalid_argument("parallel projection: expected BF16 or FP32 output");
    } else if (w.qtype == QType::BF16) {
        Tensor matrix(const_cast<void*>(w.qdata), DType::BF16, {w.k, w.n});
        const Tensor* rows[] = {&matrix};
        ops::projection_fp32(x, rows, out, stream);
    } else {
        ops::projection_fp32(x, w, out, stream);
    }
}
} // namespace infernix::models::qwen4_exp::execution
