#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace infernix {

struct RowPartition {
    std::int32_t begin = 0;
    std::int32_t count = 0;
};

// Partition complete row groups. The final shard may have a short last group;
// encoded quantization groups along K are never split.
inline RowPartition partition_rows(std::int32_t rows, std::int32_t rank,
                                   std::int32_t ranks, std::int32_t alignment = 16) {
    if (rows <= 0 || ranks <= 0 || rank < 0 || rank >= ranks || alignment <= 0) {
        throw std::invalid_argument("invalid tensor-parallel row partition");
    }
    const std::int64_t groups = (std::int64_t(rows) + alignment - 1) / alignment;
    const auto first = std::min<std::int64_t>(rows, groups * rank / ranks * alignment);
    const auto last = std::min<std::int64_t>(rows, groups * (rank + 1) / ranks * alignment);
    return {static_cast<std::int32_t>(first), static_cast<std::int32_t>(last - first)};
}

inline void validate_parallel_devices(std::span<const int> devices, int primary) {
    if (devices.empty()) { return; }
    if (devices.size() < 2 || devices.size() > 4 || devices.front() != primary) {
        throw std::invalid_argument("parallel devices must contain 2-4 ordinals, with --device first");
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] < 0 || std::find(devices.begin(), devices.begin() + i, devices[i]) != devices.begin() + i) {
            throw std::invalid_argument("parallel device ordinals must be nonnegative and unique");
        }
    }
}

} // namespace infernix
