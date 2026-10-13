#pragma once

#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/head_partition.h"
#include <algorithm>
#include <vector>

namespace infernix::models::qwen4_exp {

inline constexpr std::size_t kParallelStateChunk = 2ULL << 20;

// Head-major planes reserve a whole, aligned range for each head. CUDA VMM
// maps each chunk to exactly one owner; the logical page ids stay shared.
inline std::vector<int> parallel_kv_owners(const DeviceKVPagePoolLayout& pool,
                                          std::size_t bytes, int planes_per_layer, int ranks, bool pipeline = false, bool fixed = false) {
    if (pool.spec.geometry.device_plane_order != PagedKVPlaneOrder::HeadMajor || planes_per_layer <= 0) {
        throw std::invalid_argument("parallel KV requires head-major layer planes");
    }
    std::vector<int> owners((bytes + kParallelStateChunk - 1) / kParallelStateChunk, 0);
    for (std::size_t p = 0; p < pool.planes.size(); ++p) {
        const auto& plane = pool.planes[p];
        const auto offset = plane.storage.region.offset;
        const int heads = plane.geometry.head_extent;
        const auto head_bytes = plane.storage.region.bytes / heads;
        if (offset % kParallelStateChunk || head_bytes % kParallelStateChunk) {
            throw std::invalid_argument("parallel KV head range is not allocation-aligned");
        }
        for (int head = 0; head < heads; ++head) {
            const int owner = heads == 1 ? 0 : attention_head_owner(p / planes_per_layer, head, heads, ranks, pipeline, fixed);
            const auto first = (offset + head * head_bytes) / kParallelStateChunk;
            std::fill_n(owners.begin() + first, head_bytes / kParallelStateChunk, owner);
        }
    }
    return owners;
}

// A single head in a HeadMajor pool is already contiguous in page-major form.
inline Tensor parallel_kv_head(const Tensor& plane, int head) {
    if (!plane.data) { return {}; }
    return plane.slice(3, head, 1).view({plane.ne[0], plane.ne[1], 1, plane.ne[2]});
}
} // namespace infernix::models::qwen4_exp
