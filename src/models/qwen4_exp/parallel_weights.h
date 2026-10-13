#pragma once

#include "artifact/materializer.h"
#include "core/parallel_partition.h"
#include "core/weight.h"
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {
// A segment retains encoded rows verbatim. Paired gate/up and HC stream rows
// are assigned together, so their fused kernels can operate on local weights.
enum class WeightPartition { Rows, Paired, StreamFeatures, Experts };
struct ParallelWeightPlan {
    artifact::ObjectHandle object;
    WeightPartition kind = WeightPartition::Rows;
    int groups = 1;
    // Model-selected semantic head rows. Empty means the regular partition above.
    // An individual rank may own no rows of a two-owner attention projection.
    std::vector<std::vector<RowPartition>> rank_rows;
};
struct ParallelWeightShard {
    int device = 0;
    RowPartition units;
    std::vector<RowPartition> segments;
    Weight weight;
    DeviceBuffer codes, scales;
};
struct PartitionedWeight {
    const WeightParent* parent = nullptr;
    WeightPartition kind = WeightPartition::Rows;
    int groups = 1;
    // Dimensions/encoding only. Null data deliberately prevents accidentally
    // executing a full matrix on the primary GPU.
    Weight logical;
    std::vector<ParallelWeightShard> shards;
};

[[nodiscard]] Weight parallel_weight_metadata(const WeightGeometry& geometry, int rows);
[[nodiscard]] std::vector<RowPartition> parallel_weight_segments(int rows, WeightPartition kind,
                                                                 int groups, int rank, int ranks);
[[nodiscard]] std::vector<RowPartition> parallel_weight_segments(int rows, const ParallelWeightPlan& plan,
                                                                 int rank, int ranks);

// Immutable Model-owned backing, read directly from the artifact into each GPU's
// final row shards. Neither a full primary copy nor a full host copy is created.
class ParallelWeightStorage {
public:
    ParallelWeightStorage(const artifact::Reader& reader, const artifact::MaterializedArtifact& backing,
                          std::span<const ParallelWeightPlan> plans, std::span<const int> devices,
                          int primary);
    ~ParallelWeightStorage();
    [[nodiscard]] const PartitionedWeight* find(const WeightParent* parent) const noexcept;
    [[nodiscard]] std::uint64_t bytes(int device) const noexcept;
private:
    int primary_;
    std::vector<PartitionedWeight> weights_;
};
} // namespace infernix::models::qwen4_exp
