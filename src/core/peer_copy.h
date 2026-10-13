#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <span>

namespace infernix {

// Reads a contiguous peer allocation and writes a row range of a local matrix.
// The caller must enable peer access and order source completion before this call.
void gather_peer_rows(const Tensor& peer, Tensor& output, int first_row, cudaStream_t stream);

// Copies only this owner's routed output slots. Other slots are left untouched.
void gather_peer_experts(const Tensor& peer, const Tensor& ids, Tensor& output,
                         int rank, int ranks, cudaStream_t stream);
void gather_peer_expert_range(const Tensor& peer, const Tensor& ids, Tensor& output,
                             int first_expert, int count, cudaStream_t stream);
// Graph-safe, word-aligned P2P transfer performed by the destination device.
void copy_peer_bytes(void* destination, const void* source, std::size_t bytes, cudaStream_t stream);

struct PeerBroadcastTarget {
    void* data;
    int device;
    cudaStream_t stream;
    cudaEvent_t piece_ready;
};
// Scatter distinct word-aligned pieces, then exchange them between receivers.
// All pairs need P2P. Each target stream waits for source_ready, and returns
// ordered after its complete copy. Events and buffers must outlive all consumers;
// the caller joins the streams before reusing them. Restores the current device.
void broadcast_peer_bytes(const void* source, std::size_t bytes,
                          std::span<const PeerBroadcastTarget> targets, cudaEvent_t source_ready);

} // namespace infernix
