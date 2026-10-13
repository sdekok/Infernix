#include "models/qwen4_exp/execution/parallel_state.h"

namespace infernix::models::qwen4_exp::execution {
namespace {
struct Layout {
    LinearAttentionStatePoolLayout state;
    GdnReplayRecordLayout records;
    std::size_t state_bytes, record_bytes;
};
Layout plan(const TextConfig& c, int ranks, int rank, int lanes, int width, bool pipeline) {
    const auto p = gdn_head_partition(c.gdn.key_heads, c.gdn.value_heads, rank, ranks, pipeline);
    const int channels = 2 * p.keys.count * c.gdn.key_head_dim + p.values.count * c.gdn.value_head_dim;
    LayoutBuilder state, records;
    Layout out;
    out.state = plan_linear_attention_state_pool(state, {.layers = static_cast<std::uint32_t>(pipeline_layers(c.gdn_layers, ranks, pipeline)), .conv_channels = channels,
        .conv_width = int(c.gdn.conv_kernel - 1), .value_heads = p.values.count,
        .value_head_dim = int(c.gdn.value_head_dim), .key_head_dim = int(c.gdn.key_head_dim), .slot_count = lanes});
    out.state_bytes = state.finish(256);
    if (width > 1) {
        out.records = plan_gdn_replay_records(records, {.layers = pipeline_layers(c.gdn_layers, ranks, pipeline), .record_capacity = lanes,
            .width = width, .conv_channels = channels, .qk_heads = p.keys.count, .value_heads = p.values.count,
            .key_dim = int(c.gdn.key_head_dim), .value_dim = int(c.gdn.value_head_dim)});
    }
    out.record_bytes = width > 1 ? records.finish(256) : 0;
    return out;
}
}
ParallelGdnState::ParallelGdnState(const TextConfig& c, std::span<const int> devices, int lanes, int width, bool pipeline)
    : primary_(devices.front()) {
    try {
        shards.reserve(devices.size());
        for (std::size_t r = 0; r < devices.size(); ++r) {
            CUDA_CHECK(cudaSetDevice(devices[r]));
            const auto layout = plan(c, devices.size(), r, lanes, width, pipeline);
            GdnStateShard shard;
            shard.first_layer = pipeline_first(r, c.gdn_layers, devices.size(), pipeline);
            shard.device = devices[r]; shard.heads = gdn_head_partition(c.gdn.key_heads, c.gdn.value_heads, r, devices.size(), pipeline);
            shard.backing = DeviceBuffer(layout.state_bytes); shard.backing.fill(0);
            shard.state = std::make_unique<LinearAttentionStatePool>(DeviceSpan{shard.backing.p, shard.backing.bytes}, layout.state);
            if (layout.record_bytes) {
                shard.record_backing = DeviceBuffer(layout.record_bytes);
                shard.records = GdnReplayRecords({shard.record_backing.p, shard.record_backing.bytes}, layout.records);
            }
            shard.folds.resize(width + 1);
            shards.push_back(std::move(shard));
        }
    } catch (...) { (void)cudaSetDevice(primary_); throw; }
    CUDA_CHECK(cudaSetDevice(primary_));
}
ParallelGdnState::~ParallelGdnState() {
    for (auto& shard : shards) {
        (void)cudaSetDevice(shard.device);
        shard.folds.clear(); shard.state.reset();
        shard.record_backing = DeviceBuffer{}; shard.backing = DeviceBuffer{};
    }
    (void)cudaSetDevice(primary_);
}
std::size_t ParallelGdnState::bytes(int rank) const {
    return shards.at(rank).backing.bytes + shards.at(rank).record_backing.bytes;
}
std::size_t ParallelGdnState::planned_bytes(const TextConfig& c, int ranks, int rank, int lanes, int width, bool pipeline) {
    const auto layout = plan(c, ranks, rank, lanes, width, pipeline);
    return layout.state_bytes + layout.record_bytes;
}
} // namespace infernix::models::qwen4_exp::execution
