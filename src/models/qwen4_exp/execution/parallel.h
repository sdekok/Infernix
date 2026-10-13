#pragma once

#include "models/qwen4_exp/execution/parameters.h"
#include "infernix/ops/qsa.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/rope.h"
#include <memory>
#include <span>

namespace infernix::models::qwen4_exp {
class ExpertResidency;
namespace execution {
class ParallelGdnState;
struct ForwardBatch;
struct AttentionCall;

// Program-owned native P2P execution. Whole output rows keep each dot product on
// one GPU; whole experts keep their existing precision/cast boundaries. Mutable
// sequence state follows head ownership; the ordered expert combine stays primary.
class ParallelExecution {
public:
    ParallelExecution(const Parameters& parameters, DeviceContext& primary,
                      std::span<const int> devices, int max_columns,
                      std::uint32_t graph_count, std::optional<std::uint64_t> headroom,
                      int max_context, KvCacheStorage kv_storage, int static_columns);
    ~ParallelExecution();
    ParallelExecution(const ParallelExecution&) = delete;
    ParallelExecution& operator=(const ParallelExecution&) = delete;

    int size() const noexcept;
    void attach_caches(ExpertResidency& primary_cache);
    bool enter_wide(int columns);
    void leave_wide() noexcept;
    void return_wide_work();
    // One streamed chunk or layer-walk span. Peer rings borrow cache frames and
    // remain protected until all prefetches and consumers have joined.
    void begin_prefill_stream();
    void end_prefill_stream();
    void restart_pipeline_stream(int stage);
    bool pipeline() const;
    void switch_pipeline_stage();
    void synchronize_pipeline();
    // Returns false for an explicitly primary-only projection.
    bool project(const Tensor& x, const LinearParameters& parameters, Tensor& output);
    void head(const Tensor& x, const LinearParameters& weight, Tensor& output);
    void swiglu(const Tensor& x, const LinearParameters& weight, Tensor& output);
    void mix(const Tensor& residual, const HyperConnectionParameters& parameters, int streams, int rank,
             float epsilon, Tensor& output, Tensor* inject, WorkspaceArena& workspace);
    void resident_experts(const Tensor& x, const ops::MoeRouting& routing, const MtpParameters& parameters,
                          Tensor& output);
    void begin_experts(std::uint32_t layer, const Tensor& x, const ops::MoeRouting& routing,
                       bool streamed, bool release_stream);
    void combine_experts(std::uint32_t layer, const ops::MoeRouting& routing, const Tensor& outputs, const Tensor& shared, Tensor& y);
    const std::int32_t* expert_owners(std::uint32_t layer) const;
    Tensor attention(const AttentionParameters& parameters, const Tensor& x,
                     const AttentionCall& call, const Tensor& block_tables,
                     const ops::PreparedRope& rope, WorkspaceArena& work);
    bool reserve_peer_kv(std::span<const std::size_t> added);
    void return_peer_kv(std::span<const std::size_t> freed);
    Tensor gdn(const Tensor& x, const GdnParameters& parameters, std::uint32_t layer,
               const ForwardBatch& batch, ParallelGdnState& state, WorkspaceArena& work);
    void reset_gdn(ParallelGdnState& state, int slot);
    void commit_gdn(ParallelGdnState& state, int width, std::span<const ops::GdnReplayFoldRow> rows);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace execution
} // namespace infernix::models::qwen4_exp
