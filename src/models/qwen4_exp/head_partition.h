#pragma once

#include "core/parallel_partition.h"

namespace infernix::models::qwen4_exp {

// TP2+PP2 assigns each half of the decoder to one GPU pair. Ordinary TP
// partitions every layer over all selected GPUs.
inline int pipeline_first(int rank, int layers, int ranks, bool pipeline = false) { return pipeline && ranks == 4 ? (rank / 2) * (layers / 2) : 0; }
inline int pipeline_layers(int layers, int ranks, bool pipeline = false) { return pipeline && ranks == 4 ? layers / 2 : layers; }
inline bool pipeline_owns(int layer, int rank, int layers, int ranks, bool pipeline = false) {
    return layer >= pipeline_first(rank, layers, ranks, pipeline) && layer < pipeline_first(rank, layers, ranks, pipeline) + pipeline_layers(layers, ranks, pipeline);
}
inline int parallel_expert_owner(int layer, int expert, int experts, int layers, int ranks, bool pipeline = false) {
    return pipeline && ranks == 4 ? (layer >= layers / 2 ? 2 : 0) + expert / (experts / 2) : expert % ranks;
}

// Each whole KV head and its query group share one owner: rotating ranks by
// default, the first pair for fixed ownership, or the layer's pipeline pair.
// Pipeline MTP uses the second pair.
inline int attention_head_owner(int layer, int head, int kv_heads, int ranks, bool pipeline = false, bool fixed = false) {
    if (layer < 0 || head < 0 || head >= kv_heads || kv_heads != 2 || ranks < 2 || ranks > 4) {
        throw std::invalid_argument("Flash-Next attention requires two KV heads and 2-4 ranks");
    }
    return pipeline && ranks == 4 ? (layer >= 6 ? 2 : 0) + head : fixed ? head : (std::int64_t(layer) * kv_heads + head) % ranks;
}

struct GdnHeadPartition {
    RowPartition keys;
    RowPartition values;
};

inline GdnHeadPartition gdn_head_partition(int keys, int values, int rank, int ranks, bool pipeline = false) {
    if (keys < ranks || values <= 0 || values % keys) {
        throw std::invalid_argument("GDN value heads must form complete key-head groups");
    }
    const auto part = partition_rows(keys, pipeline && ranks == 4 ? rank % 2 : rank, pipeline && ranks == 4 ? 2 : ranks, 1);
    const int group = values / keys;
    return {part, {part.begin * group, part.count * group}};
}
} // namespace infernix::models::qwen4_exp
