#pragma once

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "infernix/ops/gdn_replay.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/head_partition.h"
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp::execution {
struct GdnStateShard {
    int device = 0;
    int first_layer = 0;
    bool owns(int layer) const { return state && layer >= first_layer && layer < first_layer + state->layer_count(); }
    GdnHeadPartition heads;
    DeviceBuffer backing, record_backing;
    std::unique_ptr<LinearAttentionStatePool> state;
    GdnReplayRecords records;
    std::vector<std::optional<ops::GdnReplayFoldPlan>> folds;
};

// Mutable Program data. Every recurrence, convolution channel and speculative
// transition record has one owner matching its input-projection head group.
class ParallelGdnState {
public:
    ParallelGdnState(const TextConfig& config, std::span<const int> devices, int lanes, int width, bool pipeline = false);
    ~ParallelGdnState();
    std::vector<GdnStateShard> shards;
    [[nodiscard]] std::size_t bytes(int rank) const;
    [[nodiscard]] static std::size_t planned_bytes(const TextConfig& c, int ranks, int rank, int lanes, int width, bool pipeline = false);
private:
    int primary_;
};
} // namespace infernix::models::qwen4_exp::execution
