#include "core/peer_copy.h"
#include "core/device.h"
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernix {
namespace {
unsigned copy_blocks(std::int64_t elements) {
    return static_cast<unsigned>(std::clamp<std::int64_t>((elements + 255) / 256, 1, 256));
}
template <class Word>
__global__ void rows_kernel(const Word* input, Word* output, int rows, int columns,
                            int input_stride, int output_rows, int first) {
    const std::int64_t count = std::int64_t(rows) * columns;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x) {
        output[(i / rows) * output_rows + first + i % rows] = input[(i / rows) * input_stride + i % rows];
    }
}

__global__ void experts_kernel(const std::uint16_t* input, const std::int32_t* ids,
                               std::uint16_t* output, int hidden, int entries, int rank, int ranks, int last = 0) {
    const std::int64_t count = std::int64_t(hidden) * entries;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::int64_t(blockDim.x) * gridDim.x) {
        const int expert = ids[i / hidden];
        if (ranks > 0 ? expert % ranks == rank : expert >= rank && expert < last) { output[i] = input[i]; }
    }
}
}

void gather_peer_rows(const Tensor& peer, Tensor& output, int first, cudaStream_t stream) {
    if (!output.is_contiguous() || peer.dtype != output.dtype ||
        peer.ne[1] != output.ne[1] || first < 0 || first + peer.ne[0] > output.ne[0] ||
        peer.ne[2] != 1 || peer.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument("peer row gather: invalid matrix views");
    }
    // Wide activations use the copy engine. Remote scalar loads serialize many
    // PCIe read requests; keep the kernel route for latency-sensitive small views.
    if ((peer.dtype == DType::BF16 || peer.dtype == DType::FP32) && peer.bytes() >= (64ULL << 10)) {
        const std::size_t word = peer.dtype == DType::BF16 ? 2 : 4;
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(output.data) + first * word,
                                    output.ne[0] * word, peer.data, peer.nb[1],
                                    peer.ne[0] * word, peer.ne[1], cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (peer.dtype == DType::BF16) {
        rows_kernel<<<copy_blocks(peer.numel()), 256, 0, stream>>>(static_cast<const std::uint16_t*>(peer.data),
            static_cast<std::uint16_t*>(output.data), peer.ne[0], peer.ne[1], peer.nb[1] / 2, output.ne[0], first);
    } else if (peer.dtype == DType::FP32) {
        rows_kernel<<<copy_blocks(peer.numel()), 256, 0, stream>>>(static_cast<const std::uint32_t*>(peer.data),
            static_cast<std::uint32_t*>(output.data), peer.ne[0], peer.ne[1], peer.nb[1] / 4, output.ne[0], first);
    } else {
        throw std::invalid_argument("peer row gather: expected BF16 or FP32");
    }
    CUDA_CHECK(cudaGetLastError());
}

void gather_peer_experts(const Tensor& peer, const Tensor& ids, Tensor& output,
                         int rank, int ranks, cudaStream_t stream) {
    if (peer.dtype != DType::BF16 || output.dtype != DType::BF16 || ids.dtype != DType::I32 ||
        !peer.is_contiguous() || !output.is_contiguous() || !ids.is_contiguous() ||
        peer.ne[0] != output.ne[0] || peer.ne[1] != output.ne[1] ||
        ids.numel() != output.ne[1] || ranks <= 0 || rank < 0 || rank >= ranks) {
        throw std::invalid_argument("peer expert gather: invalid views or rank");
    }
    experts_kernel<<<copy_blocks(output.numel()), 256, 0, stream>>>(static_cast<const std::uint16_t*>(peer.data),
        static_cast<const std::int32_t*>(ids.data), static_cast<std::uint16_t*>(output.data),
        output.ne[0], output.ne[1], rank, ranks);
    CUDA_CHECK(cudaGetLastError());
}
void gather_peer_expert_range(const Tensor& peer, const Tensor& ids, Tensor& output,
                             int first, int count, cudaStream_t stream) {
    if (peer.dtype != DType::BF16 || output.dtype != DType::BF16 || ids.dtype != DType::I32 ||
        !peer.is_contiguous() || !output.is_contiguous() || !ids.is_contiguous() ||
        peer.ne[0] != output.ne[0] || peer.ne[1] != output.ne[1] || ids.numel() != output.ne[1] ||
        first < 0 || count <= 0) { throw std::invalid_argument("peer expert range: invalid views or interval"); }
    experts_kernel<<<copy_blocks(output.numel()), 256, 0, stream>>>(static_cast<const std::uint16_t*>(peer.data),
        static_cast<const std::int32_t*>(ids.data), static_cast<std::uint16_t*>(output.data),
        output.ne[0], output.ne[1], first, 0, first + count);
    CUDA_CHECK(cudaGetLastError());
}
void copy_peer_bytes(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0 || bytes % 4 || bytes / 4 > std::size_t(std::numeric_limits<int>::max()) ||
        reinterpret_cast<std::uintptr_t>(source) % 4 || reinterpret_cast<std::uintptr_t>(destination) % 4) {
        throw std::invalid_argument("peer copy requires aligned nonempty words");
    }
    if (bytes >= (64ULL << 10)) {
        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
        return;
    }
    Tensor input(const_cast<void*>(source), DType::FP32, {static_cast<int>(bytes / 4), 1});
    Tensor output(destination, DType::FP32, {static_cast<int>(bytes / 4), 1});
    gather_peer_rows(input, output, 0, stream);
}

void broadcast_peer_bytes(const void* source, std::size_t bytes,
                          std::span<const PeerBroadcastTarget> targets, cudaEvent_t source_ready) {
    if (targets.empty() || bytes % 4 || bytes / 4 < targets.size() || !source || !source_ready) {
        throw std::invalid_argument("peer broadcast requires nonempty aligned pieces and a ready event");
    }
    int previous; CUDA_CHECK(cudaGetDevice(&previous));
    struct Restore { int device; ~Restore() { (void)cudaSetDevice(device); } } restore{previous};
    const auto begin = [&](std::size_t i) { return (bytes / 4 * i / targets.size()) * 4; };
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i]; CUDA_CHECK(cudaSetDevice(target.device));
        CUDA_CHECK(cudaStreamWaitEvent(target.stream, source_ready, 0));
        copy_peer_bytes(static_cast<std::byte*>(target.data) + begin(i),
                        static_cast<const std::byte*>(source) + begin(i), begin(i + 1) - begin(i), target.stream);
        CUDA_CHECK(cudaEventRecord(target.piece_ready, target.stream));
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i]; CUDA_CHECK(cudaSetDevice(target.device));
        for (std::size_t step = 1; step < targets.size(); ++step) {
            const auto owner = (i + step) % targets.size();
            CUDA_CHECK(cudaStreamWaitEvent(target.stream, targets[owner].piece_ready, 0));
            copy_peer_bytes(static_cast<std::byte*>(target.data) + begin(owner),
                            static_cast<const std::byte*>(targets[owner].data) + begin(owner),
                            begin(owner + 1) - begin(owner), target.stream);
        }
    }
    // All piece-read waits have been queued before these events are re-recorded.
    // Join every reader before any receiver can reuse its broadcast buffer as
    // scratch (GDN's next all-gather does exactly that).
    for (const auto& target : targets) {
        CUDA_CHECK(cudaSetDevice(target.device));
        CUDA_CHECK(cudaEventRecord(target.piece_ready, target.stream));
    }
    for (const auto& target : targets) {
        CUDA_CHECK(cudaSetDevice(target.device));
        for (const auto& peer : targets) {
            if (peer.device != target.device) { CUDA_CHECK(cudaStreamWaitEvent(target.stream, peer.piece_ready, 0)); }
        }
    }
}
} // namespace infernix
