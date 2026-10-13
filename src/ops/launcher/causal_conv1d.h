#pragma once

// infernix::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// Registered split partitions, named by their row profile. The wrapper resolves one of these from
// the destination row counts and refuses anything else.
enum class CausalConvSplitGeometry {
    Rows2048x2048x4096,
    Rows2048x2048x6144,
    Rows512x512x1536,
    Rows640x640x1920,
    Rows768x768x2304,
    Rows1024x1024x3072,
};

inline constexpr std::int32_t kCausalConvParallelMaxTokens = 32;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out0, Tensor& out1, Tensor& out2,
                                        CausalConvSplitGeometry geometry, cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_smallt_split_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out0, Tensor& out1, Tensor& out2,
                                       CausalConvSplitGeometry geometry, cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                   const Tensor& valid_columns, const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& out,
                                   cudaStream_t stream);
void causal_conv1d_from_states_launch(const Tensor& x, const Tensor& weight,
                                      const Tensor& conv_states, const Tensor& initial_state_slots,
                                      Tensor& out, cudaStream_t stream);

} // namespace infernix::ops::detail
