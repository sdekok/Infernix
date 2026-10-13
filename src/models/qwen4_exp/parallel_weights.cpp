#include "models/qwen4_exp/parallel_weights.h"
#include "artifact/reader.h"
#include "core/device.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernix::models::qwen4_exp {
Weight parallel_weight_metadata(const WeightGeometry& g, int rows) {
    if (g.shape.size() != 2 || g.shape[1] > std::uint64_t(std::numeric_limits<int>::max()) || rows <= 0) {
        throw std::invalid_argument("parallel weights require a positive matrix");
    }
    if (!((g.format == QType::BF16 && g.layout == QuantLayout::Contiguous) ||
          ((g.format == QType::Q8_G32_FP16 || g.format == QType::Q4_G64_FP16) && g.layout == QuantLayout::RowSplit))) {
        throw std::invalid_argument("parallel weights currently require BF16 or row-split Q8/Q4");
    }
    Weight w;
    w.qtype = g.format; w.layout = g.layout; w.ndim = 2;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = static_cast<int>(g.shape[1]);
    w.padded_shape[1] = g.layout == QuantLayout::Contiguous ? w.k : static_cast<int>(g.padded_columns);
    if (g.layout == QuantLayout::RowSplit) {
        w.group_size = w.group = static_cast<int>(g.group_size);
        w.scale_dtype = DType::FP16;
        w.scale_ne[0] = w.padded_shape[1] / w.group;
        w.scale_ne[1] = rows;
        w.scale_nb[0] = 2;
        w.scale_nb[1] = g.scale_bytes_per_row;
        w.scale_nb[2] = w.scale_nb[3] = w.scale_nb[1] * rows;
    }
    return w;
}

std::vector<RowPartition> parallel_weight_segments(int rows, WeightPartition kind, int groups, int rank, int ranks) {
    if (groups <= 0 || rows % groups != 0) { throw std::invalid_argument("parallel weight groups do not divide rows"); }
    if (kind == WeightPartition::Experts) {
        const auto experts = partition_rows(groups, rank, ranks, 1);
        return {{experts.begin * (rows / groups), experts.count * (rows / groups)}};
    }
    const int parts = kind == WeightPartition::Rows ? 1 : groups;
    const int units = rows / parts;
    const auto p = partition_rows(units, rank, ranks, units < ranks * 16 ? 1 : 16);
    if (p.count == 0) { throw std::invalid_argument("parallel weight has an empty shard"); }
    std::vector<RowPartition> out;
    for (int part = 0; part < parts; ++part) { out.push_back({part * units + p.begin, p.count}); }
    return out;
}

ParallelWeightStorage::ParallelWeightStorage(const artifact::Reader& reader,
    const artifact::MaterializedArtifact& backing, std::span<const ParallelWeightPlan> plans,
    std::span<const int> devices, int primary) : primary_(primary) {
    validate_parallel_devices(devices, primary);
    weights_.reserve(plans.size());
    // Bounded pageable staging is temporary and is released after loading.
    std::vector<std::byte> staging(8ULL << 20);
    try {
        for (const auto& plan : plans) {
            const auto& g = reader.geometry(plan.object);
            PartitionedWeight weight;
            weight.parent = &backing.streamed_parent(plan.object);
            weight.kind = plan.kind; weight.groups = plan.groups;
            if (g.shape[0] > std::uint64_t(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("parallel matrix exceeds supported row count");
            }
            weight.logical = parallel_weight_metadata(g, static_cast<int>(g.shape[0]));
            const auto file_base = artifact::object_offset(reader.directory().object(plan.object));
            const std::size_t code_row = g.layout == QuantLayout::Contiguous ? g.shape[1] * 2 : g.code_bytes_per_row;
            for (std::size_t rank = 0; rank < devices.size(); ++rank) {
                CUDA_CHECK(cudaSetDevice(devices[rank]));
                ParallelWeightShard shard;
                shard.device = devices[rank];
                shard.segments = parallel_weight_segments(weight.logical.n, plan,
                                                            static_cast<int>(rank), static_cast<int>(devices.size()));
                if (shard.segments.empty()) {
                    shard.weight = weight.logical;
                    shard.weight.n = shard.weight.shape[0] = shard.weight.padded_shape[0] = 0;
                    weight.shards.push_back(std::move(shard));
                    continue;
                }
                shard.units = plan.kind == WeightPartition::Experts
                    ? partition_rows(plan.groups, static_cast<int>(rank), static_cast<int>(devices.size()), 1)
                    : shard.segments.front();
                int rows = 0;
                for (const auto p : shard.segments) { rows += p.count; }
                shard.weight = parallel_weight_metadata(g, rows);
                shard.codes = DeviceBuffer(std::size_t(rows) * code_row);
                if (g.scale_bytes_per_row) { shard.scales = DeviceBuffer(std::size_t(rows) * g.scale_bytes_per_row); }
                const auto plane = [&](DeviceBuffer& output, std::uint64_t plane_offset, std::size_t row_bytes) {
                    std::size_t destination = 0;
                    for (const auto part : shard.segments) {
                        const std::size_t bytes = std::size_t(part.count) * row_bytes;
                        for (std::size_t at = 0; at < bytes;) {
                            const std::size_t count = std::min(staging.size(), bytes - at);
                            reader.read_into(file_base + plane_offset + std::uint64_t(part.begin) * row_bytes + at,
                                             std::span(staging).first(count));
                            output.copy_from_host(staging.data(), count, destination + at);
                            at += count;
                        }
                        destination += bytes;
                    }
                };
                plane(shard.codes, 0, code_row);
                if (shard.scales.bytes) { plane(shard.scales, g.scale_offset, g.scale_bytes_per_row); }
                shard.weight.payload = shard.weight.qdata = shard.codes.p;
                shard.weight.payload_bytes = shard.codes.bytes + shard.scales.bytes;
                shard.weight.scales = shard.scales.p;
                weight.shards.push_back(std::move(shard));
            }
            weights_.push_back(std::move(weight));
        }
    } catch (...) { (void)cudaSetDevice(primary_); throw; }
    CUDA_CHECK(cudaSetDevice(primary_));
}

std::vector<RowPartition> parallel_weight_segments(int rows, const ParallelWeightPlan& plan, int rank, int ranks) {
    if (plan.rank_rows.empty()) { return parallel_weight_segments(rows, plan.kind, plan.groups, rank, ranks); }
    if (plan.rank_rows.size() != std::size_t(ranks) || rank < 0 || rank >= ranks) {
        throw std::invalid_argument("semantic weight partition rank count mismatch");
    }
    // Validate exact global coverage once per metadata query: no duplicate head
    // weights, omitted rows, or reordering inside one rank's local projection.
    std::vector<RowPartition> all;
    for (const auto& owner : plan.rank_rows) {
        int end = 0;
        for (const auto p : owner) {
            if (p.count <= 0 || p.begin < end || p.begin > rows - p.count) {
                throw std::invalid_argument("invalid semantic weight row interval");
            }
            end = p.begin + p.count;
            all.push_back(p);
        }
    }
    std::sort(all.begin(), all.end(), [](auto a, auto b) { return a.begin < b.begin; });
    int end = 0;
    for (const auto p : all) {
        if (p.begin != end) { throw std::invalid_argument("semantic weight rows overlap or have a gap"); }
        end += p.count;
    }
    if (end != rows) { throw std::invalid_argument("semantic weight rows do not cover their parent"); }
    return plan.rank_rows[rank];
}

ParallelWeightStorage::~ParallelWeightStorage() {
    for (auto& weight : weights_) for (auto& shard : weight.shards) {
        (void)cudaSetDevice(shard.device);
        shard.codes = DeviceBuffer{}; shard.scales = DeviceBuffer{};
    }
    (void)cudaSetDevice(primary_);
}
const PartitionedWeight* ParallelWeightStorage::find(const WeightParent* parent) const noexcept {
    for (const auto& weight : weights_) if (weight.parent == parent) { return &weight; }
    return nullptr;
}
std::uint64_t ParallelWeightStorage::bytes(int device) const noexcept {
    std::uint64_t count = 0;
    for (const auto& weight : weights_) for (const auto& shard : weight.shards) if (shard.device == device) {
        count += shard.codes.bytes + shard.scales.bytes;
    }
    return count;
}
} // namespace infernix::models::qwen4_exp
