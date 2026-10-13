#include "models/qwen4_exp/execution/parallel.h"
#include "models/qwen4_exp/execution/parallel_projection.h"
#include "core/device.h"
#include "core/peer_copy.h"
#include "core/vram_budget.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/resident_moe.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/rows.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/expert_stream.h"
#include "models/qwen4_exp/parallel_kv.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/sigmoid_mul.h"
#include "infernix/ops/gdn_gating.h"
#include "models/qwen4_exp/execution/parallel_state.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace infernix::models::qwen4_exp::execution {
namespace {
struct RestoreDevice {
    int device;
    ~RestoreDevice() { (void)cudaSetDevice(device); }
};
void shared_product(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& output,
                    WorkspaceArena& work, cudaStream_t stream) {
    if (weight.qtype == QType::Q8_G32_FP16 || x.ne[1] <= 16) {
        ops::linear_swiglu(x, weight, output, policy, work, stream);
    } else {
        const int I = output.ne[0], T = x.ne[1];
        auto scope = work.scope();
        Tensor projected = work.alloc(DType::BF16, {2 * I, T});
        ops::linear(x, weight, projected, policy, work, stream);
        Tensor gate = work.alloc(DType::BF16, {I, T}), up = work.alloc(DType::BF16, {I, T});
        Tensor* parts[] = {&gate, &up};
        ops::split_rows(projected, parts, stream);
        ops::silu_mul(gate, up, output, stream);
    }
}
}

struct ParallelExecution::Impl {
    struct ScratchBuffer {
        DeviceBuffer owned;
        void* p = nullptr;
        std::size_t bytes = 0;
        ScratchBuffer& operator=(DeviceBuffer buffer) {
            owned = std::move(buffer); restore(); return *this;
        }
        void use(DeviceSpan span) { p = span.data; bytes = span.bytes; }
        void restore() { p = owned.p; bytes = owned.bytes; }
    };
    struct Rank {
        std::unique_ptr<DeviceContext> owned_device;
        DeviceContext* device = nullptr;
        ScratchBuffer input, output;
        DeviceBuffer hc_partials, hc_inverse, hc_gathered, hc_norm;
        Tensor gdn_output;
        std::unique_ptr<WorkspaceArena> workspace;
        std::unique_ptr<WorkspaceArena> state_work;
        ScratchBuffer expert_input, expert_ids, expert_outputs, expert_dispatch, expert_work, resident_work, expert_partials;
        DeviceBuffer expert_scales, staging;
        std::array<std::size_t, 9> wide_sizes{};
        std::array<DeviceSpan, 9> wide_views{};
        std::size_t wide_workspace_bytes = 0, wide_bytes = 0;
        ExpertResidency::FrameLease wide_lease;
        DeviceBuffer wide_fallback;
        std::unique_ptr<WorkspaceArena> wide_workspace;
        std::array<ScratchBuffer*, 9> buffers() {
            return {&input, &output, &expert_input, &expert_ids, &expert_outputs,
                    &expert_dispatch, &expert_work, &resident_work, &expert_partials};
        }
        std::unique_ptr<VramBudgetSource> vram;
        std::unique_ptr<ExpertResidency> cache;
        ExpertResidency::FrameLease prefill_lease;
        std::unique_ptr<ExpertStream> prefill_stream;
        std::uint64_t prefill_streamed_before = 0;
        std::uint64_t free_reserve = 0;
        cudaEvent_t start = nullptr, done = nullptr, experts_done = nullptr, gdn_projected = nullptr, broadcast_ready = nullptr;
        ~Rank() {
            if (device) { device->bind_to_current_thread_noexcept(); (void)cudaStreamSynchronize(device->stream); }
            if (start) { (void)cudaEventDestroy(start); }
            if (done) { (void)cudaEventDestroy(done); }
            if (experts_done) { (void)cudaEventDestroy(experts_done); }
            if (gdn_projected) { (void)cudaEventDestroy(gdn_projected); }
            if (broadcast_ready) { (void)cudaEventDestroy(broadcast_ready); }
        }
    };
    const Parameters& parameters;
    DeviceContext& primary;
    const bool pipeline;
    const bool fixed_kv;
    int columns;
    int max_context;
    int static_columns;
    ExpertResidency* primary_cache = nullptr;
    bool wide_active = false;
    std::vector<std::unique_ptr<Rank>> ranks;
    std::unique_ptr<Rank> pipeline_root;
    cudaStream_t pipeline_stream = nullptr;
    static constexpr int staging_slots = 32;
    // Fused HC only serves T <= 16, rank <= 512, four residual streams.
    static constexpr std::size_t hc_bytes = 4 * 528 * 16 * sizeof(float);

    Impl(const Parameters& p, DeviceContext& d, std::span<const int> devices, int max_columns,
         std::uint32_t graph_count, std::optional<std::uint64_t> headroom, int context, KvCacheStorage storage, int narrow_columns)
        : parameters(p), primary(d), pipeline(p.model.options().pipeline_parallel), fixed_kv(p.model.options().fixed_kv_owners), columns(max_columns), max_context(context), static_columns(std::min(max_columns, narrow_columns)) {
        RestoreDevice restore{primary.device};
        validate_parallel_devices(devices, primary.device);
        if (devices.empty() || columns <= 0 || static_columns <= 0 || parameters.model.expert_store()) {
            throw std::invalid_argument("parallel execution requires a nonempty plan and the full expert bank in RAM");
        }
        int max_k = 10240, max_n = 10240, head_n = 0;
        const auto consider = [&](const LinearParameters& linear) {
            if (!linear.parallel || linear.parallel->shards.size() != devices.size()) {
                throw std::invalid_argument("parallel execution requires load-time weight shards");
            }
            max_k = std::max(max_k, linear.weight.k);
            for (const auto& shard : linear.parallel->shards) { max_n = std::max(max_n, shard.weight.n); }
        };
        const auto consider_head = [&](const LinearParameters& linear) {
            if (!linear.parallel || linear.parallel->shards.size() != devices.size()) {
                throw std::invalid_argument("parallel head requires load-time weight shards");
            }
            max_k = std::max(max_k, linear.weight.k);
            for (const auto& shard : linear.parallel->shards) { head_n = std::max(head_n, shard.weight.n); }
        };
        const auto hc = [&](const HyperConnectionParameters& h) { consider(h.down); consider(h.up); };
        const auto attention = [&](const AttentionParameters& a) {
            consider(a.projection); if (a.index_projection) { consider(*a.index_projection); } consider(a.output);
        };
        for (const auto& layer : parameters.layers) {
            hc(layer.attn_hc); hc(layer.mlp_hc);
            if (const auto* a = std::get_if<AttentionParameters>(&layer.mixer)) { attention(*a); }
            else { const auto& g = std::get<GdnParameters>(layer.mixer); consider(g.projection); consider(g.control); consider(g.output); }
            consider(*layer.moe.parallel_router); consider(layer.moe.shared_gate_up); consider(layer.moe.shared_down);
            if (layer.ple) { consider(layer.ple->projection); }
        }
        hc(parameters.final_mixer); consider_head(*parameters.parallel_head);
        if (parameters.mtp) {
            const auto& m = *parameters.mtp;
            hc(m.attn_hc); hc(m.mlp_hc); hc(m.final_mixer); attention(m.attention);
            consider(m.embedding_projection); consider(m.hidden_projection); consider(*m.parallel_router);
            consider(m.shared_gate_up); consider(m.shared_down);
            if (!m.parallel_experts_gate_up || !m.parallel_experts_down) { throw std::invalid_argument("MTP experts must be partitioned"); }
        }
        if (parameters.draft_head.rows) { consider_head(*parameters.draft_head.rows); }
        for (std::size_t i = 0; i < devices.size(); ++i) {
            auto rank = std::make_unique<Rank>();
            if (i == 0) { rank->device = &primary; }
            else { rank->owned_device = std::make_unique<DeviceContext>(devices[i]); rank->device = rank->owned_device.get(); }
            rank->device->bind_to_current_thread();
            if (rank->device->compute_capability() != 120) { throw std::invalid_argument("parallel execution requires SM120 GPUs"); }
            CUDA_CHECK(cudaEventCreateWithFlags(&rank->start, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank->done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank->experts_done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank->gdn_projected, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank->broadcast_ready, cudaEventDisableTiming));
            ranks.push_back(std::move(rank));
        }
        for (int from : devices) for (int to : devices) {
            if (from == to) { continue; }
            int access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&access, from, to));
            if (!access) { throw std::runtime_error("parallel execution requires P2P between every selected GPU pair"); }
            CUDA_CHECK(cudaSetDevice(from));
            const auto error = cudaDeviceEnablePeerAccess(to, 0);
            if (error == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); } else { CUDA_CHECK(error); }
        }
        const auto& c = parameters.model.config().text;
        const auto activation = parameters.layers.front().moe.bank->activation;
        const auto stride = parameters.layers.front().moe.bank->planes.record_stride;
        const auto align = [](std::size_t n) { return (n + 255) / 256 * 256; };
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            auto& rank = *ranks[i]; rank.device->bind_to_current_thread();
            const auto sizes = [&](int width) -> std::array<std::size_t, 9> {
                const auto entries = std::size_t(c.moe.top_k) * width;
                return {std::size_t(max_k) * width * 2,
                    std::max({std::size_t(max_n) * width, std::size_t(head_n) * std::min(width, static_columns),
                              i ? std::size_t(0) : std::size_t(c.hidden_size) * width}) * 4,
                    std::size_t(c.hidden_size) * width * 2, entries * 8, entries * c.hidden_size * 2,
                    i ? ops::moe_dispatch_bytes(c.moe.experts, static_cast<int>(entries)) : 0,
                    i ? ops::moe_experts_workspace_bytes(c.moe.experts, static_cast<int>(entries), width, activation) : 0,
                    parameters.mtp ? ops::resident_moe_workspace_bytes(entries, c.moe.intermediate) : 0,
                    i ? std::size_t(c.hidden_size) * width * 4 : 0};
            };
            auto geometry = qsa_geometry(c); geometry.heads /= geometry.kv_heads; geometry.kv_heads = 1;
            const auto heads = gdn_head_partition(c.gdn.key_heads, c.gdn.value_heads, i, ranks.size(), pipeline);
            const auto state_bytes = [&](int width) {
                return std::max(ops::qsa_attention_workspace_bytes(geometry, width, max_context, storage),
                    ops::gated_delta_net_workspace_capacity_bytes(heads.keys.count, heads.values.count, 1, width)) +
                    std::size_t(16) * max_k * width + (1ULL << 20);
            };
            const auto workspace_bytes = [&](int width) {
                return std::max<std::size_t>(1ULL << 20, 8ULL * c.moe.shared_intermediate * width + 4096);
            };
            const auto narrow = sizes(static_columns); rank.wide_sizes = sizes(columns);
            // These are sources/destinations of concurrent pitched peer DMA.
            // Keep native allocations: VMM source views slow that traffic
            // on this topology. Heads only project narrow token selections, so
            // their vocabulary extent must not multiply the prefill width.
            const auto buffers = rank.buffers();
            for (std::size_t j = 0; j < buffers.size(); ++j) {
                const bool native_transfer = j < 2;
                *buffers[j] = DeviceBuffer(native_transfer ? rank.wide_sizes[j] : narrow[j]);
                if (native_transfer) { rank.wide_sizes[j] = 0; }
                rank.wide_bytes += align(rank.wide_sizes[j]);
            }
            rank.workspace = std::make_unique<WorkspaceArena>(workspace_bytes(static_columns));
            rank.state_work = std::make_unique<WorkspaceArena>(state_bytes(columns));
            rank.wide_workspace_bytes = workspace_bytes(columns);
            rank.wide_bytes += align(rank.wide_workspace_bytes);
            rank.hc_norm = DeviceBuffer(std::size_t(c.residual_width()) * 2);
            rank.hc_partials = DeviceBuffer(hc_bytes); rank.hc_inverse = DeviceBuffer(4 * 16 * sizeof(float));
            if (i == 0) {
                rank.hc_gathered = DeviceBuffer(hc_bytes);
                if (pipeline) {
                pipeline_root = std::make_unique<Rank>(); auto& aux = *pipeline_root; aux.device = &primary;
                CUDA_CHECK(cudaStreamCreateWithFlags(&pipeline_stream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&aux.start, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&aux.done, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&aux.experts_done, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&aux.gdn_projected, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&aux.broadcast_ready, cudaEventDisableTiming));
                aux.input = DeviceBuffer(rank.input.bytes); aux.output = DeviceBuffer(rank.output.bytes);
                aux.workspace = std::make_unique<WorkspaceArena>(workspace_bytes(columns));
                aux.state_work = std::make_unique<WorkspaceArena>(state_bytes(columns));
                aux.hc_norm = DeviceBuffer(rank.hc_norm.bytes); aux.hc_partials = DeviceBuffer(hc_bytes);
                aux.hc_inverse = DeviceBuffer(4 * 16 * sizeof(float)); aux.hc_gathered = DeviceBuffer(hc_bytes);
                }
                continue;
            }
            rank.expert_scales = DeviceBuffer(std::size_t(c.num_hidden_layers) * c.moe.experts * sizeof(ops::offloaded_moe::ExpertScales));
            rank.staging = DeviceBuffer(stride * staging_slots);
            std::vector<const std::uint8_t*> banks;
            for (std::size_t l = 0; l < parameters.layers.size(); ++l) {
                const auto* bank = parameters.layers[l].moe.bank;
                banks.push_back(reinterpret_cast<const std::uint8_t*>(bank->planes.device_records));
                rank.expert_scales.copy_from_host(bank->scales.data(), bank->scales.size() * sizeof(bank->scales[0]),
                                                  l * c.moe.experts * sizeof(bank->scales[0]));
            }
            rank.cache = std::make_unique<ExpertResidency>(c, std::move(banks), stride, columns, rank.device->device);
            rank.cache->set_owner(static_cast<int>(i), static_cast<int>(ranks.size()), pipeline);
            for (int reader : devices) { rank.cache->allow_peer_access(reader); }
            rank.prefill_stream = std::make_unique<ExpertStream>(c.num_hidden_layers, c.moe.experts, stride,
                [this, i, stride](std::uint32_t layer, std::uint32_t expert) -> const std::uint8_t* {
                    if (parallel_expert_owner(layer, expert, parameters.model.config().text.moe.experts, parameters.model.config().text.num_hidden_layers, ranks.size(), pipeline) != int(i)) { return nullptr; }
                    const auto* bank = parameters.layers.at(layer).moe.bank;
                    return reinterpret_cast<const std::uint8_t*>(bank->planes.device_records) + expert * stride;
                }, true);
            rank.vram = open_vram_budget_source(rank.device->device, false);
            VramDemand demand;
            demand.headroom = headroom; demand.graphs = graph_count; demand.frame_bytes = stride;
            demand.max_frames = rank.cache->frame_limit(); demand.elastic = rank.cache->elastic();
            const auto sizing = size_expert_frames(rank.vram->query(), demand);
            rank.free_reserve = sizing.headroom + sizing.reserve;
            const auto allocated = rank.cache->resize(sizing.frames, rank.device->stream, *rank.vram);
            if (allocated.frames == 0) { throw std::runtime_error("parallel execution: no room for peer expert cache"); }
            std::fprintf(stderr, "[parallel] GPU %d: %u expert frames (%.2f GiB); load-time dense shards and workspace allocated\n",
                         rank.device->device, allocated.frames, double(rank.cache->pool_bytes()) / double(1ULL << 30));
            rank.device->synchronize();
        }
    }
    ~Impl() { primary.bind_to_current_thread_noexcept(); if (pipeline_stream) { (void)cudaStreamSynchronize(pipeline_stream); } pipeline_root.reset(); if (pipeline_stream) { (void)cudaStreamDestroy(pipeline_stream); } for (auto& rank : ranks) { rank.reset(); } primary.bind_to_current_thread_noexcept(); }
    bool distributed_broadcast(const Tensor& x) const { return ranks.size() > 2 && x.bytes() >= (1ULL << 20); }
    void start(const Tensor* x = nullptr, const PartitionedWeight* weight = nullptr) {
        primary.bind_to_current_thread(); CUDA_CHECK(cudaEventRecord(ranks.front()->start, primary.stream));
        if (x && distributed_broadcast(*x)) {
            std::array<PeerBroadcastTarget, 3> targets;
            std::size_t count = 0;
            for (std::size_t i = 1; i < ranks.size(); ++i) {
                if (weight && weight->shards[i].weight.n == 0) { continue; }
                auto& r = *ranks[i];
                targets[count++] = {r.input.p, r.device->device, r.device->stream, r.broadcast_ready};
            }
            if (count) { broadcast_peer_bytes(x->data, x->bytes(), {targets.data(), count}, ranks.front()->start); }
        }
    }
    Tensor input(int i, const Tensor& x) {
        auto& r = *ranks[i]; r.device->bind_to_current_thread();
        if (i == 0) { return x; }
        if (distributed_broadcast(x)) { return Tensor(r.input.p, x.dtype, {x.ne[0], x.ne[1]}); }
        CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, ranks.front()->start, 0));
        copy_peer_bytes(r.input.p, x.data, x.bytes(), r.device->stream);
        return Tensor(r.input.p, x.dtype, {x.ne[0], x.ne[1]});
    }
    void done(int i) { CUDA_CHECK(cudaEventRecord(ranks[i]->done, ranks[i]->device->stream)); }
    void wait(int i) {
        primary.bind_to_current_thread();
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(primary.stream, ranks[i]->done, 0)); }
    }
    void project(const Tensor& x, const LinearParameters& linear, Tensor& output, bool swiglu = false) {
        if (!linear.parallel || x.ne[0] != linear.weight.k || x.ne[1] != output.ne[1]) {
            throw std::invalid_argument("parallel projection exceeds its startup plan");
        }
        const auto& shards = linear.parallel->shards;
        // MTP's hidden projection views S residual streams as S*T columns of
        // H features. It occupies the same bytes as an S*H-by-T HC input; a
        // token-column limit would incorrectly reject this legal view.
        // Check every actual transfer/output extent before queuing any work.
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            const auto& shard = shards[i];
            if (shard.weight.n == 0) { continue; }
            const int n = swiglu ? shard.units.count : shard.weight.n;
            Tensor local(nullptr, output.dtype, {n, x.ne[1]});
            if (x.bytes() > ranks[i]->input.bytes || local.bytes() > ranks[i]->output.bytes) {
                throw std::invalid_argument("parallel projection exceeds its startup buffer capacity");
            }
        }
        RestoreDevice restore{primary.device}; start(&x, linear.parallel);
        // Peer work is queued first; all streams join before buffers can be reused.
        for (std::size_t step = 0; step < ranks.size(); ++step) {
            const int i = static_cast<int>((step + 1) % ranks.size());
            auto& r = *ranks[i]; const auto& shard = shards[i];
            if (shard.weight.n == 0) { continue; }
            Tensor local_x = input(i, x);
            const int n = swiglu ? shard.units.count : shard.weight.n;
            Tensor local_y(r.output.p, output.dtype, {n, x.ne[1]});
            r.workspace->reset();
            if (swiglu) { shared_product(local_x, shard.weight, linear.policy, local_y, *r.workspace, r.device->stream); }
            else { parallel_projection(local_x, shard.weight, local_y, linear.policy, *r.workspace, r.device->stream); }
            done(i);
        }
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            if (shards[i].weight.n == 0) { continue; }
            wait(i); const auto& shard = shards[i];
            Tensor local(ranks[i]->output.p, output.dtype, {swiglu ? shard.units.count : shard.weight.n, x.ne[1]});
            if (swiglu) { gather_peer_rows(local, output, shard.units.begin, primary.stream); }
            else {
                int first = 0;
                for (const auto segment : shard.segments) {
                    Tensor part = local.slice(0, first, segment.count);
                    gather_peer_rows(part, output, segment.begin, primary.stream); first += segment.count;
                }
            }
        }
    }
};

ParallelExecution::ParallelExecution(const Parameters& p, DeviceContext& d, std::span<const int> devices,
                                     int columns, std::uint32_t graphs, std::optional<std::uint64_t> headroom,
                                     int max_context, KvCacheStorage storage, int static_columns)
    : impl_(std::make_unique<Impl>(p, d, devices, columns, graphs, headroom, max_context, storage, static_columns)) {}
ParallelExecution::~ParallelExecution() = default;
int ParallelExecution::size() const noexcept { return static_cast<int>(impl_->ranks.size()); }
bool ParallelExecution::project(const Tensor& x, const LinearParameters& p, Tensor& out) {
    if (!p.parallel) { return false; }
    impl_->project(x, p, out); return true;
}
void ParallelExecution::head(const Tensor& x, const LinearParameters& p, Tensor& out) { impl_->project(x, p, out); }
void ParallelExecution::swiglu(const Tensor& x, const LinearParameters& p, Tensor& out) { impl_->project(x, p, out, true); }
void ParallelExecution::attach_caches(ExpertResidency& root) {
    impl_->primary_cache = &root;
    root.set_owner(0, size(), impl_->pipeline);
    for (std::size_t i = 1; i < impl_->ranks.size(); ++i) { auto& r = *impl_->ranks[i]; root.attach_peer(*r.cache, *r.device); }
}

bool ParallelExecution::enter_wide(int columns) {
    auto& p = *impl_;
    if (columns <= p.static_columns) { return false; }
    if (columns > p.columns || p.wide_active) { throw std::logic_error("invalid parallel wide-work scope"); }
    RestoreDevice restore{p.primary.device}; p.start();
    // Acquire all storage before switching the graph-stable narrow buffers.
    // Partial acquisition remains owned and is returned by Program cleanup.
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[0]->start, 0)); }
        if (!r.wide_lease.valid() && !r.wide_fallback.p) {
            auto& cache = i ? *r.cache : *p.primary_cache;
            const auto frames = (r.wide_bytes + cache.frame_stride() - 1) / cache.frame_stride();
            if (frames <= cache.lendable()) {
                const cudaStream_t writers[] = {r.device->stream};
                r.wide_lease = cache.lend(frames, r.device->stream, writers);
            } else {
                std::fprintf(stderr, "[parallel] GPU %d wide scratch: cache loan unavailable; temporary %zu MiB allocation\n",
                             r.device->device, r.wide_bytes >> 20);
                r.wide_fallback = DeviceBuffer(r.wide_bytes);
            }
        }
    }
    for (auto& rank : p.ranks) {
        auto& r = *rank;
        DeviceSpan span = r.wide_lease.valid() ? r.wide_lease.memory : DeviceSpan{r.wide_fallback.p, r.wide_fallback.bytes};
        WorkspaceArena area(span);
        const auto buffers = r.buffers();
        for (std::size_t j = 0; j < buffers.size(); ++j) { r.wide_views[j] = r.wide_sizes[j] ? area.alloc_bytes(r.wide_sizes[j]) : DeviceSpan{}; }
        r.wide_workspace = std::make_unique<WorkspaceArena>(area.alloc_bytes(r.wide_workspace_bytes));
    }
    // No throwing work after this point: either every rank changes views or none does.
    for (auto& rank : p.ranks) {
        auto& r = *rank;
        const auto buffers = r.buffers();
        for (std::size_t j = 0; j < buffers.size(); ++j) { if (r.wide_sizes[j]) { buffers[j]->use(r.wide_views[j]); } }
        r.workspace.swap(r.wide_workspace);
    }
    p.wide_active = true; return true;
}

void ParallelExecution::leave_wide() noexcept {
    auto& p = *impl_;
    for (auto& r : p.ranks) {
        for (auto* buffer : r->buffers()) { buffer->restore(); }
        r->workspace.swap(r->wide_workspace);
    }
    p.wide_active = false;
}

void ParallelExecution::return_wide_work() {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    if (p.wide_active) { throw std::logic_error("parallel wide work still active"); }
    if (std::none_of(p.ranks.begin(), p.ranks.end(), [](const auto& rank) {
            return rank->wide_lease.valid() || rank->wide_fallback.p;
        })) { return; }
    p.start();
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        // The primary has joined every operation that reads peer scratch.
        // Order each cache's new writers after that final reader join.
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[0]->start, 0)); }
        if (r.wide_lease.valid()) {
            auto& cache = i ? *r.cache : *p.primary_cache;
            std::fprintf(stderr, "[parallel] GPU %d wide scratch: %zu MiB, %u frames returned for decode\n",
                         r.device->device, r.wide_bytes >> 20, r.wide_lease.count);
            cache.give_back(r.wide_lease, r.device->stream);
        }
        if (r.wide_fallback.p) { r.device->synchronize(); r.wide_fallback = {}; }
        r.wide_workspace.reset(); p.done(i);
    }
    for (int i = 1; i < size(); ++i) { p.wait(i); }
}

void ParallelExecution::begin_prefill_stream() {
    auto& p = *impl_; RestoreDevice restore{p.primary.device}; p.start();
    const auto& c = p.parameters.model.config().text;
    // A full pair of owned layers guarantees every peer miss is streamed,
    // including experts evicted by the loan itself. Tables occupy one frame.
    const auto frames = 2 * (p.pipeline ? c.moe.experts / 2 : (c.moe.experts + size() - 1) / size()) + 1;
    for (int i = 1; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (r.prefill_lease.valid()) { throw std::logic_error("parallel prefill stream already active"); }
        if (r.cache->lendable() < frames) { continue; }
        CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks.front()->start, 0));
        const cudaStream_t writers[] = {r.prefill_stream->stream()};
        r.prefill_lease = r.cache->lend(frames, r.device->stream, writers);
        r.prefill_streamed_before = r.prefill_stream->stats().streamed;
        r.prefill_stream->begin(r.prefill_lease.memory, r.cache->host_table(), r.device->stream, false, pipeline_first(i, c.num_hidden_layers, size(), p.pipeline), pipeline_first(i, c.num_hidden_layers, size(), p.pipeline) + pipeline_layers(c.num_hidden_layers, size(), p.pipeline));
    }
}

void ParallelExecution::switch_pipeline_stage() {
    auto& p = *impl_; p.primary.bind_to_current_thread();
    std::swap(p.primary.stream, p.pipeline_stream); std::swap(p.ranks[0], p.pipeline_root);
}
void ParallelExecution::synchronize_pipeline() {
    auto& p = *impl_; p.primary.bind_to_current_thread();
    if (p.pipeline_stream) { CUDA_CHECK(cudaStreamSynchronize(p.pipeline_stream)); }
}
void ParallelExecution::restart_pipeline_stream(int stage) {
    auto& p = *impl_; const auto& c = p.parameters.model.config().text;
    for (int i = std::max(1, stage * 2); i < stage * 2 + 2; ++i) {
        auto& r = *p.ranks[i]; if (!r.prefill_lease.valid()) { continue; }
        r.device->bind_to_current_thread(); r.prefill_stream->end();
        r.prefill_stream->begin(r.prefill_lease.memory, r.cache->host_table(), r.device->stream,
            false, stage * (c.num_hidden_layers / 2), (stage + 1) * (c.num_hidden_layers / 2));
    }
    p.primary.bind_to_current_thread();
}
void ParallelExecution::end_prefill_stream() {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    for (int i = 1; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (!r.prefill_lease.valid()) { continue; }
        // Also join copies queued ahead of the last consumed layer when a
        // layer walk is cancelled. Returning the frames must never race DMA.
        CUDA_CHECK(cudaEventRecord(r.done, r.prefill_stream->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, r.done, 0));
        r.prefill_stream->end();
        const auto records = r.prefill_stream->stats().streamed - r.prefill_streamed_before;
        std::fprintf(stderr, "[parallel] GPU %d prefill stream: %llu records (%.2f GiB), %u borrowed frames returned\n",
                     r.device->device, static_cast<unsigned long long>(records),
                     double(records * r.cache->frame_stride()) / double(1ULL << 30), r.prefill_lease.count);
        r.cache->give_back(r.prefill_lease, r.device->stream);
        p.done(i); p.wait(i);
    }
}

bool ParallelExecution::reserve_peer_kv(std::span<const std::size_t> added) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    if (added.size() != p.ranks.size()) { throw std::invalid_argument("parallel KV owner budget mismatch"); }
    for (std::size_t i = 1; i < added.size(); ++i) {
        if (added[i] == 0) { continue; }
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread(); r.device->synchronize();
        const auto free = r.vram->query().device_free;
        const auto spare = free > r.free_reserve ? free - r.free_reserve : 0;
        if (added[i] <= spare) { continue; }
        const auto stride = r.cache->frame_stride();
        const auto give = (added[i] - spare + ExpertResidency::kChunkBytes + stride - 1) / stride;
        const auto frames = r.cache->frames();
        if (give > frames) { return false; }
        const auto result = r.cache->resize(frames - give, r.device->stream, *r.vram);
        if (result.frames > frames - give) { return false; }
    }
    return true;
}

void ParallelExecution::return_peer_kv(std::span<const std::size_t> freed) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    for (std::size_t i = 1; i < freed.size(); ++i) {
        if (!freed[i]) { continue; }
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        const auto frames = r.cache->frames() + freed[i] / r.cache->frame_stride();
        (void)r.cache->resize(std::min<std::size_t>(frames, r.cache->frame_limit()), r.device->stream, *r.vram);
    }
}

Tensor ParallelExecution::attention(const AttentionParameters& a, const Tensor& x,
    const AttentionCall& call, const Tensor& block_tables, const ops::PreparedRope& rope,
    WorkspaceArena& primary_work) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    const auto& config = p.parameters.model.config().text;
    const auto geometry = qsa_geometry(config);
    const int T = call.width * call.batch, D = geometry.head_dim;
    const int Q = geometry.heads / geometry.kv_heads, QW = Q * D;
    const int IH = config.qsa.index_heads, ID = config.qsa.index_head_dim;
    if (geometry.kv_heads != 2 || T > p.columns || call.layer.spaces.host_pages ||
        call.layer.spaces.device_pages != 0x7FFFFFFF) {
        throw std::invalid_argument("parallel attention requires two uniquely owned pool KV heads");
    }
    auto local_geometry = geometry;
    local_geometry.heads = Q; local_geometry.kv_heads = 1;
    auto layer = call.layer;
    if (layer.kv.window.present()) { layer.kv.window.slots = call.slots; }
    const ops::QsaBatch batch{.block_tables = block_tables, .table_rows = call.table_rows,
        .positions = call.positions, .tail_slots = call.slots, .rope_positions = call.rope_positions,
        .block_start_rope = call.block_start_rope, .batch = call.batch, .width = call.width,
        .update_tails = call.update_tails};
    Tensor iq = primary_work.alloc(DType::BF16, {ID, IH, T});
    Tensor ik = call.key_records.data ? call.key_records.view({ID, T}) : primary_work.alloc(DType::BF16, {ID, T});
    std::array<Tensor, 2> queries, keys, values, gates, results;
    // The projection loader packs each owner's semantic row segments in global
    // order. Extract by global range, including the primary-only index rows.
    const auto extract = [&](const Tensor& packed, const ParallelWeightShard& shard,
                             int begin, int count, Tensor& target, cudaStream_t stream) {
        int offset = 0, copied = 0;
        for (const auto segment : shard.segments) {
            const int first = std::max(begin, segment.begin);
            const int last = std::min(begin + count, segment.begin + segment.count);
            if (first < last) {
                auto part = packed.slice(0, offset + first - segment.begin, last - first);
                gather_peer_rows(part, target, first - begin, stream); copied += last - first;
            }
            offset += segment.count;
        }
        if (copied != count) { throw std::logic_error("attention projection has incomplete owner rows"); }
    };
    p.start(&x, a.projection.parallel);
    for (int step = 0; step < size(); ++step) {
        const int owner = (step + 1) % size();
        auto& r = *p.ranks[owner]; const auto& shard = a.projection.parallel->shards[owner];
        // The primary also owns a separately encoded index projection, if any.
        if (!shard.weight.n && (owner || !a.index_projection)) { continue; }
        Tensor input = p.input(owner, x);
        auto& work = *r.state_work; work.reset(); r.workspace->reset();
        const auto s = r.device->stream;
        Tensor projected = work.alloc(DType::BF16, {shard.weight.n, T});
        if (shard.weight.n) { parallel_projection(input, shard.weight, projected, a.projection.policy, *r.workspace, s); }
        for (int head = 0; head < 2; ++head) {
            if (attention_head_owner(call.layer_index, head, 2, size(), p.pipeline, p.fixed_kv) != owner) { continue; }
            Tensor q = work.alloc(DType::BF16, {QW, T});
            Tensor k = work.alloc(DType::BF16, {D, T});
            gates[head] = work.alloc(DType::BF16, {QW, T});
            values[head] = work.alloc(DType::BF16, {D, T});
            extract(projected, shard, head * QW, QW, q, s);
            extract(projected, shard, 2 * QW + head * QW, QW, gates[head], s);
            extract(projected, shard, 4 * QW + head * D, D, k, s);
            extract(projected, shard, 4 * QW + 2 * D + head * D, D, values[head], s);
            queries[head] = work.alloc(DType::BF16, {D, Q, T});
            keys[head] = work.alloc(DType::BF16, {D, 1, T});
            Tensor qnorm = a.query_norm, knorm = a.key_norm;
            if (owner) {
                qnorm = work.alloc(DType::BF16, {D}); knorm = work.alloc(DType::BF16, {D});
                copy_peer_bytes(qnorm.data, a.query_norm.data, qnorm.bytes(), s);
                copy_peer_bytes(knorm.data, a.key_norm.data, knorm.bytes(), s);
            }
            auto qn = queries[head].view({D, Q * T}), kn = keys[head].view({D, T});
            ops::rmsnorm(q.view({D, Q * T}), qnorm, config.rms_norm_eps, true, qn, s);
            ops::rmsnorm(k, knorm, config.rms_norm_eps, true, kn, s);
            ops::rope(call.rope_positions, rope, queries[head], keys[head], r.device->execution_view());
        }
        if (owner == 0) {
            Tensor iqrows = iq.view({ID * IH, T});
            if (a.index_projection) {
                const auto& index = *a.index_projection;
                const auto& index_shard = index.parallel->shards[0];
                Tensor indexed = work.alloc(DType::BF16, {index_shard.weight.n, T});
                parallel_projection(input, index_shard.weight, indexed, index.policy, *r.workspace, s);
                extract(indexed, index_shard, 0, IH * ID, iqrows, s);
                extract(indexed, index_shard, IH * ID, ID, ik, s);
            } else {
                extract(projected, shard, 4 * QW + 4 * D, IH * ID, iqrows, s);
                extract(projected, shard, 4 * QW + 4 * D + IH * ID, ID, ik, s);
            }
        }
    }
    p.primary.bind_to_current_thread();
    ops::qsa_pool_keys(ik, a.index_key_norm, call.tails, layer, batch, geometry, p.primary.stream);
    ops::QsaSelection selection;
    const bool shared_selection = !call.kv_only && T >= 256;
    if (!call.kv_only) { ops::qsa_index_query(iq, a.index_query_norm, call.rope_positions, geometry, p.primary.stream); }
    if (shared_selection) {
        selection = {primary_work.alloc(DType::I32, {geometry.budget / geometry.ratio, T}),
                     primary_work.alloc(DType::I32, {T})};
        const auto scratch = primary_work.alloc_bytes(ops::qsa_selection_workspace_bytes(geometry, T, p.max_context));
        ops::qsa_select_blocks(iq, layer, batch, geometry, p.max_context,
                              scratch.data, scratch.bytes, selection, p.primary.stream);
    }
    p.start();
    for (int head = 0; head < 2; ++head) {
        const int owner = attention_head_owner(call.layer_index, head, 2, size(), p.pipeline, p.fixed_kv);
        auto& r = *p.ranks[owner]; r.device->bind_to_current_thread();
        const auto s = r.device->stream;
        if (owner) { CUDA_CHECK(cudaStreamWaitEvent(s, p.ranks[0]->start, 0)); }
        auto& work = *r.state_work;
        auto local_selection = selection;
        if (shared_selection && owner) {
            local_selection = {work.alloc(DType::I32, {selection.blocks.ne[0], T}), work.alloc(DType::I32, {T})};
            copy_peer_bytes(local_selection.blocks.data, selection.blocks.data, selection.blocks.bytes(), s);
            copy_peer_bytes(local_selection.counts.data, selection.counts.data, selection.counts.bytes(), s);
        }
        auto owned = layer; owned.kv.num_kv_heads = 1;
        owned.kv.k_pages = parallel_kv_head(layer.kv.k_pages, head);
        owned.kv.v_pages = parallel_kv_head(layer.kv.v_pages, head);
        owned.kv.k_scale_pages = parallel_kv_head(layer.kv.k_scale_pages, head);
        owned.kv.v_scale_pages = parallel_kv_head(layer.kv.v_scale_pages, head);
        if (owned.kv.window.present()) {
            auto& w = owned.kv.window;
            w.k_codes = w.k_codes.slice(2, head, 1); w.v_codes = w.v_codes.slice(2, head, 1);
            w.k_scales = w.k_scales.slice(2, head, 1); w.v_scales = w.v_scales.slice(2, head, 1);
            w.tags = w.tags.slice(2, head, 1);
        }
        const bool vq = owned.kv.storage == KvCacheStorage::Vq2 || owned.kv.storage == KvCacheStorage::Q4KeyVq2Value;
        ops::QsaAppend append{keys[head].view({D, 1, call.width, call.batch}),
                              values[head].view({D, 1, call.width, call.batch})};
        if (!vq || call.kv_only) {
            const PagedKVBatchLayerView view{.k_pages = owned.kv.k_pages, .v_pages = owned.kv.v_pages,
                .k_scale_pages = owned.kv.k_scale_pages, .v_scale_pages = owned.kv.v_scale_pages,
                .block_tables = block_tables, .head_dim = D, .num_kv_heads = 1,
                .storage = owned.kv.storage, .window = owned.kv.window};
            ops::kv_cache_append_batch(append.k, append.v, call.positions.view({call.width, call.batch}),
                                       call.table_rows, view, s);
        }
        if (!call.kv_only) {
            results[head] = work.alloc(DType::BF16, {D, Q, T});
            const auto scratch = work.alloc_bytes(ops::qsa_attention_workspace_bytes(local_geometry, T, p.max_context, owned.kv.storage));
            ops::qsa_attention(queries[head], iq, owned, batch, local_geometry, float(1. / std::sqrt(double(D))),
                p.max_context, scratch.data, scratch.bytes, results[head], s, vq ? &append : nullptr,
                shared_selection ? &local_selection : nullptr);
            auto gated = results[head].view({QW, T});
            ops::sigmoid_mul(gates[head], gated, s);
        }
        p.done(owner);
    }
    if (call.kv_only) {
        for (int head = 0; head < 2; ++head) { p.wait(attention_head_owner(call.layer_index, head, 2, size(), p.pipeline, p.fixed_kv)); }
        return {};
    }
    Tensor output = primary_work.alloc(DType::BF16, {int(config.hidden_size), T});
    // Output-row owners pull the gated pieces directly from their KV owners.
    // Separate completion events retain all producer readiness until every reader
    // has consumed the pieces, including the primary's own output projection.
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (!a.output.parallel->shards[i].weight.n) { continue; }
        Tensor input(r.input.p, DType::BF16, {2 * QW, T});
        for (int head = 0; head < 2; ++head) {
            const int owner = attention_head_owner(call.layer_index, head, 2, size(), p.pipeline, p.fixed_kv);
            if (owner != i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[owner]->done, 0)); }
            auto part = results[head].view({QW, T});
            gather_peer_rows(part, input, head * QW, r.device->stream);
        }
        const auto& shard = a.output.parallel->shards[i];
        Tensor local(r.output.p, DType::BF16, {shard.weight.n, T});
        r.workspace->reset();
        parallel_projection(input, shard.weight, local, a.output.policy, *r.workspace, r.device->stream);
        CUDA_CHECK(cudaEventRecord(r.gdn_projected, r.device->stream));
    }
    p.primary.bind_to_current_thread();
    for (int i = 0; i < size(); ++i) {
        if (!a.output.parallel->shards[i].weight.n) { continue; }
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(p.primary.stream, p.ranks[i]->gdn_projected, 0)); }
        const auto& shard = a.output.parallel->shards[i];
        Tensor local(p.ranks[i]->output.p, DType::BF16, {shard.weight.n, T});
        int first = 0;
        for (const auto segment : shard.segments) {
            auto part = local.slice(0, first, segment.count);
            gather_peer_rows(part, output, segment.begin, p.primary.stream); first += segment.count;
        }
    }
    return output;
}

void ParallelExecution::reset_gdn(ParallelGdnState& state, int slot) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device}; p.start();
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[0]->start, 0)); }
        state.shards[i].state->zero_slot(slot, r.device->stream); p.done(i);
    }
    for (int i = 0; i < size(); ++i) { p.wait(i); }
}

void ParallelExecution::commit_gdn(ParallelGdnState& state, int width, std::span<const ops::GdnReplayFoldRow> rows) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device}; p.start();
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[0]->start, 0)); }
        auto& shard = state.shards[i]; auto& fold = shard.folds.at(width);
        if (!fold) { fold.emplace(shard.records.narrowed(width), shard.state->all_layers_view()); }
        fold->execute(rows, r.device->stream); p.done(i);
    }
    for (int i = 0; i < size(); ++i) { p.wait(i); }
}

Tensor ParallelExecution::gdn(const Tensor& x, const GdnParameters& g, std::uint32_t layer,
    const ForwardBatch& batch, ParallelGdnState& states, WorkspaceArena& primary_work) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    const auto& config = p.parameters.model.config().text;
    const int T = x.ne[1], DK = config.gdn.key_head_dim, DV = config.gdn.value_head_dim;
    const int VW = config.gdn.value_width();
    p.start(&x, g.projection.parallel);
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        auto& shard = states.shards[i];
        if (!shard.owns(layer)) { continue; }
        const int local_layer = layer - shard.first_layer;
        Tensor local_x = p.input(i, x);
        auto& work = *r.state_work; work.reset(); const auto s = r.device->stream;
        const int nk = shard.heads.keys.count, nv = shard.heads.values.count, kw = nk * DK, vw = nv * DV, C = 2 * kw + vw;
        GdnReplayRecordLayer records;
        if (batch.verify) { records = shard.records.narrowed(batch.width).layer(local_layer, batch.batch); }
        Tensor qkv = batch.verify ? records.conv.view({C, T}) : work.alloc(DType::BF16, {C, T});
        const auto copy_rows = [&](const Tensor& src, int first, int count, Tensor& dst, int at = 0) {
            auto part = src.slice(0, first, count); gather_peer_rows(part, dst, at, s);
        };
        // Projection rows are stored as local Q, K, V, Z and A, B groups.
        // Keep them on their owner through recurrence instead of gathering to
        // the primary and immediately distributing the same rows back here.
        Tensor full = work.alloc(DType::BF16, {2 * kw + 2 * vw, T});
        Tensor ab = work.alloc(DType::BF16, {2 * nv, T});
        r.workspace->reset();
        parallel_projection(local_x, g.projection.parallel->shards[i].weight, full,
                            g.projection.policy, *r.workspace, s);
        parallel_projection(local_x, g.control.parallel->shards[i].weight, ab,
                            g.control.policy, *r.workspace, s);
        copy_rows(full, 0, C, qkv);
        Tensor z = work.alloc(DType::BF16, {vw, T}); copy_rows(full, C, vw, z);
        Tensor a = work.alloc(DType::BF16, {nv, T}), b = work.alloc(DType::BF16, {nv, T});
        copy_rows(ab, 0, nv, a); copy_rows(ab, nv, nv, b);
        Tensor gate = work.alloc(DType::FP32, {nv, T}), beta = work.alloc(DType::FP32, {nv, T});
        const Tensor log = g.a_log.slice(0, shard.heads.values.begin, nv), bias = g.dt_bias.slice(0, shard.heads.values.begin, nv);
        ops::gdn_gating(a, b, log, bias, gate, beta, s);
        Tensor q = work.alloc(DType::BF16, {kw, T}), k = work.alloc(DType::BF16, {kw, T}), v = work.alloc(DType::BF16, {vw, T});
        Tensor o = work.alloc(DType::BF16, {DV, nv, T});
        const auto conv = g.parallel_convolution.at(i);
        const float scale = float(1. / std::sqrt(double(DK)));
        auto state = shard.state->layer_view(local_layer);
        if (batch.verify) {
            const int W = batch.width, B = batch.batch;
            Tensor conv_out = work.alloc(DType::BF16, {C, W, B});
            ops::causal_conv1d_silu_from_states(qkv.view({C, W, B}), conv, state.conv, batch.slots, conv_out, s);
            Tensor* parts[] = {&q, &k, &v}; ops::split_rows(conv_out.view({C, T}), parts, s);
            Tensor out = o.view({DV, nv, W, B});
            ops::gated_delta_net_replay_record(q.view({DK, nk, W, B}), k.view({DK, nk, W, B}), v.view({DV, nv, W, B}),
                gate.view({nv, W, B}), beta.view({nv, W, B}), scale, state.recurrent, Tensor{}, batch.slots,
                records.key, records.value, records.gate, out, s);
        } else if (batch.width > 1) {
            if (batch.batch != 1 || batch.host_slots.empty()) { throw std::invalid_argument("parallel GDN prefill requires one sequence"); }
            auto history = shard.state->conv_slot(local_layer, batch.host_slots.front());
            ops::causal_conv1d_silu_split(qkv, conv, history, history, q, k, v, s);
            auto recurrence = shard.state->recurrent_slot(local_layer, batch.host_slots.front());
            ops::gated_delta_net(q.view({DK, nk, T}), k.view({DK, nk, T}), v.view({DV, nv, T}), gate, beta,
                scale, true, work, recurrence, recurrence, o, r.device->execution_view());
        } else {
            Tensor conv_out = work.alloc(DType::BF16, {C, 1, T});
            ops::causal_conv1d_silu_snapshot(qkv.view({C, 1, T}), conv, state.conv, Tensor{}, batch.slots, batch.slots, conv_out, s);
            Tensor* parts[] = {&q, &k, &v}; ops::split_rows(conv_out.view({C, T}), parts, s);
            Tensor out = o.view({DV, nv, 1, T});
            ops::gated_delta_net_batch_update(q.view({DK, nk, 1, T}), k.view({DK, nk, 1, T}), v.view({DV, nv, 1, T}),
                gate.view({nv, 1, T}), beta.view({nv, 1, T}), scale, true, state.recurrent,
                batch.slots, batch.slots, out, s);
        }
        Tensor normed = work.alloc(DType::BF16, {DV, nv * T});
        r.gdn_output = normed;
        ops::gated_rmsnorm(o.view({DV, nv * T}), g.norm, z.view({DV, nv * T}),
            config.gdn.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
            config.rms_norm_eps, normed, r.device->execution_view());
        p.done(i);
    }
    // Every row owner needs the complete recurrent output for its local dense
    // dot products. Pull the disjoint pieces directly from their owners.
    Tensor output = primary_work.alloc(DType::BF16, {int(config.hidden_size), T});
    for (int i = 0; i < size(); ++i) {
        auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
        if (!states.shards[i].owns(layer)) { continue; }
        Tensor input(r.input.p, DType::BF16, {VW, T});
        for (int owner = 0; owner < size(); ++owner) {
            if (!states.shards[owner].owns(layer)) { continue; }
            if (owner != i) { CUDA_CHECK(cudaStreamWaitEvent(r.device->stream, p.ranks[owner]->done, 0)); }
            const auto head = states.shards[owner].heads.values;
            Tensor local(p.ranks[owner]->gdn_output.data, DType::BF16, {head.count * DV, T});
            gather_peer_rows(local, input, head.begin * DV, r.device->stream);
        }
        const auto& weight = g.output.parallel->shards[i].weight;
        Tensor local(r.output.p, DType::BF16, {weight.n, T});
        r.workspace->reset();
        parallel_projection(input, weight, local, g.output.policy, *r.workspace, r.device->stream);
        // Keep readiness of recurrent pieces distinct from projection completion:
        // later receivers must not accidentally wait for an already re-recorded event.
        CUDA_CHECK(cudaEventRecord(r.gdn_projected, r.device->stream));
    }
    p.primary.bind_to_current_thread();
    for (int i = 0; i < size(); ++i) {
        if (!states.shards[i].owns(layer)) { continue; }
        if (i) { CUDA_CHECK(cudaStreamWaitEvent(p.primary.stream, p.ranks[i]->gdn_projected, 0)); }
        const auto& shard = g.output.parallel->shards[i];
        if (!shard.weight.n) { continue; }
        Tensor local(p.ranks[i]->output.p, DType::BF16, {shard.weight.n, T});
        int first = 0;
        for (const auto segment : shard.segments) {
            Tensor part = local.slice(0, first, segment.count);
            gather_peer_rows(part, output, segment.begin, p.primary.stream); first += segment.count;
        }
    }
    return output;
}

void ParallelExecution::mix(const Tensor& residual, const HyperConnectionParameters& h, int streams, int rank,
                            float epsilon, Tensor& output, Tensor* inject, WorkspaceArena& work) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    const int T = residual.ne[1];
    if (h.down.parallel->shards[0].weight.n == 0 || !ops::hyper_connection_fused_supported(h.down.weight, h.up.weight, output.ne[0], streams, rank, T)) {
        auto scope = work.scope();
        Tensor normalized = work.alloc(DType::BF16, {residual.ne[0], T});
        ops::hyper_connection_norm(residual, h.norm, streams, epsilon, normalized, p.primary.stream);
        Tensor z = work.alloc(DType::BF16, {h.down.weight.n, T}); p.project(normalized, h.down, z);
        std::array<Tensor, 4> local_norms, local_outputs;
        // Down already broadcast the normalized residual. Preserve only the
        // features owned by each up shard before reusing the input buffers.
        // Collapse those streams locally, so only H rows cross back, not S*H.
        for (int i = 0; i < size(); ++i) {
            auto& r = *p.ranks[i]; r.device->bind_to_current_thread();
            auto& scratch = *r.state_work; scratch.reset();
            const auto& up = h.up.parallel->shards[i];
            if (!up.weight.n) { continue; }
            local_norms[i] = scratch.alloc(DType::BF16, {streams * up.units.count, T});
            local_outputs[i] = scratch.alloc(DType::BF16, {up.units.count, T});
            Tensor full(i ? r.input.p : normalized.data, DType::BF16, {normalized.ne[0], T});
            int at = 0;
            for (const auto segment : up.segments) {
                const auto part = full.slice(0, segment.begin, segment.count);
                gather_peer_rows(part, local_norms[i], at, r.device->stream);
                at += segment.count;
            }
        }
        p.primary.bind_to_current_thread();
        Tensor m = work.alloc(DType::BF16, {rank, T});
        ops::hyper_connection_gates(z, rank, streams, m, inject, p.primary.stream);
        p.start(&m, h.up.parallel);
        for (int i = 0; i < size(); ++i) {
            auto& r = *p.ranks[i]; const auto& up = h.up.parallel->shards[i];
            if (!up.weight.n) { continue; }
            Tensor local_m = p.input(i, m);
            Tensor u(r.output.p, DType::BF16, {up.weight.n, T});
            r.workspace->reset();
            parallel_projection(local_m, up.weight, u, h.up.policy, *r.workspace, r.device->stream);
            ops::hyper_connection_collapse(u, local_norms[i], streams, local_outputs[i], r.device->stream);
            p.done(i);
        }
        for (int i = 0; i < size(); ++i) {
            if (!h.up.parallel->shards[i].weight.n) { continue; }
            p.wait(i); const auto units = h.up.parallel->shards[i].units;
            gather_peer_rows(local_outputs[i], output, units.begin, p.primary.stream);
        }
        return;
    }
    const int rows = h.down.weight.n;
    auto& root = *p.ranks.front();
    // A separate root buffer gathers FP32 down partials without overwriting
    // the root's local partials while peer rows are being read.
    p.start();
    for (int i = 1; i <= size(); ++i) {
        const int r = i % size(); auto& peer = *p.ranks[r];
        if (!h.down.parallel->shards[r].weight.n) { continue; }
        Tensor local = p.input(r, residual);
        Tensor norm = h.norm;
        if (r) {
            // Stage the small shared vector once; per-CTA P2P norm loads are
            // high-latency on PCIe even though the matrix weights are local.
            copy_peer_bytes(peer.hc_norm.p, h.norm.data, h.norm.bytes(), peer.device->stream);
            norm = Tensor(peer.hc_norm.p, h.norm.dtype, {h.norm.ne[0]});
        }
        ops::hyper_connection_down_partition(local, norm, h.down.parallel->shards[r].weight,
                                              epsilon, peer.hc_partials.p, peer.hc_inverse.p, peer.device->stream);
        p.done(r);
    }
    for (int i = 0; i < size(); ++i) {
        if (!h.down.parallel->shards[i].weight.n) { continue; }
        p.wait(i); auto& peer = *p.ranks[i];
        const auto segment = h.down.parallel->shards[i].segments.front();
        for (int s = 0; s < streams; ++s) {
            auto* destination = static_cast<float*>(root.hc_gathered.p) + (s * rows + segment.begin) * T;
            const auto* source = static_cast<const float*>(peer.hc_partials.p) + s * segment.count * T;
            copy_peer_bytes(destination, source, std::size_t(segment.count) * T * 4, p.primary.stream);
        }
    }
    p.start();
    for (int i = 1; i <= size(); ++i) {
        const int r = i % size(); auto& peer = *p.ranks[r]; peer.device->bind_to_current_thread();
        if (!h.up.parallel->shards[r].weight.n) { continue; }
        if (r) { CUDA_CHECK(cudaStreamWaitEvent(peer.device->stream, root.start, 0)); }
        copy_peer_bytes(peer.hc_partials.p, root.hc_gathered.p, std::size_t(streams) * rows * T * 4, peer.device->stream);
        const auto& up = h.up.parallel->shards[r];
        Tensor local_residual(r ? peer.input.p : residual.data, DType::BF16, {residual.ne[0], T});
        Tensor local_output(peer.output.p, DType::BF16, {up.units.count, T});
        Tensor norm = r ? Tensor(peer.hc_norm.p, h.norm.dtype, {h.norm.ne[0]}) : h.norm;
        ops::hyper_connection_up_partition(local_residual, norm, up.weight, rows, rank, up.units.begin,
                                            peer.hc_partials.p, peer.hc_inverse.p, local_output, r ? nullptr : inject,
                                            peer.device->stream);
        p.done(r);
    }
    for (int i = 0; i < size(); ++i) {
        if (!h.up.parallel->shards[i].weight.n) { continue; }
            p.wait(i); const auto units = h.up.parallel->shards[i].units;
        Tensor local(p.ranks[i]->output.p, DType::BF16, {units.count, T});
        gather_peer_rows(local, output, units.begin, p.primary.stream);
    }
}

void ParallelExecution::begin_experts(std::uint32_t layer, const Tensor& x, const ops::MoeRouting& routing,
                                     bool streamed, bool release_stream) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device};
    const auto& c = p.parameters.model.config().text;
    const int T = x.ne[1], K = c.moe.top_k, E = c.moe.experts;
    if (T > p.columns) { throw std::invalid_argument("parallel experts exceed the startup plan"); }
    p.start();
    for (int i = 1; i < size(); ++i) {
        auto& r = *p.ranks[i];
        if (!pipeline_owns(layer, i, c.num_hidden_layers, size(), p.pipeline)) { continue; }
        r.device->bind_to_current_thread(); auto s = r.device->stream;
        CUDA_CHECK(cudaStreamWaitEvent(s, p.ranks.front()->start, 0));
        copy_peer_bytes(r.expert_input.p, x.data, x.bytes(), s);
        copy_peer_bytes(r.expert_ids.p, routing.ids.data, routing.ids.bytes(), s);
        auto* weights = static_cast<std::byte*>(r.expert_ids.p) + routing.ids.bytes();
        if (T > 8 && K >= 2 * size()) { copy_peer_bytes(weights, routing.weights.data, routing.weights.bytes(), s); }
        Tensor local_x(r.expert_input.p, DType::BF16, {x.ne[0], T}), output(r.expert_outputs.p, DType::BF16, {x.ne[0], K * T});
        ops::MoeRouting local; local.ids = Tensor(r.expert_ids.p, DType::I32, {K, T});
        local.weights = Tensor(weights, DType::FP32, {K, T});
        auto dispatch = ops::carve_moe_dispatch(r.expert_dispatch.p, E, K * T);
        ops::moe_dispatch(local, E, dispatch, r.cache->route_log() + layer * r.cache->route_stride(), s, i, size(), r.cache->owners(layer));
        const auto* bank = p.parameters.layers.at(layer).moe.bank;
        ops::MoeExpertSource source;
        source.frame_base = r.cache->frame_base(); source.frames = r.cache->table(layer);
        source.host_records = reinterpret_cast<const std::uint8_t*>(bank->planes.device_records);
        source.record_stride = bank->planes.record_stride;
        source.scales = static_cast<const ops::offloaded_moe::ExpertScales*>(r.expert_scales.p) + layer * E;
        source.activation = bank->activation; source.staging_base = static_cast<std::uint8_t*>(r.staging.p); source.staging_slots = Impl::staging_slots;
        const bool prefetched = streamed && r.prefill_stream->active();
        if (prefetched) {
            r.prefill_stream->before_experts(layer, s);
            source.prefetched = r.prefill_stream->slots(layer);
            source.prefetch_base = r.prefill_stream->base();
        }
        const int owned = p.pipeline ? E / 2 : (E + size() - 1 - i) / size();
        ops::moe_experts(local_x, dispatch, source, K, std::min(owned, K * T), r.expert_work.p, output, s);
        if (prefetched && release_stream) { r.prefill_stream->after_experts(layer, s); }
        Tensor partial(r.expert_partials.p, DType::FP32, {x.ne[0], T});
        if (T > 8 && K >= 2 * size()) { ops::moe_combine_owner(output, local, i, size(), partial, s, r.cache->owners(layer)); }
        CUDA_CHECK(cudaEventRecord(r.experts_done, s));
    }
}
const std::int32_t* ParallelExecution::expert_owners(std::uint32_t layer) const { return impl_->primary_cache->owners(layer); }
void ParallelExecution::combine_experts(std::uint32_t layer, const ops::MoeRouting& routing,
    const Tensor& outputs, const Tensor& shared, Tensor& y) {
    auto& p = *impl_; p.primary.bind_to_current_thread();
    const int H = y.ne[0], T = y.ne[1], E = p.parameters.model.config().text.moe.experts;
    const auto owns = [&](int rank) { return pipeline_owns(layer, rank, p.parameters.model.config().text.num_hidden_layers, size(), p.pipeline); };
    if (T <= 8 || routing.ids.ne[0] < 2 * size()) {
        Tensor gathered = outputs;
        for (int i = 1; i < size(); ++i) {
            if (!owns(i)) { continue; }
            auto& r = *p.ranks[i]; CUDA_CHECK(cudaStreamWaitEvent(p.primary.stream, r.experts_done, 0));
            Tensor peer(r.expert_outputs.p, DType::BF16, {outputs.ne[0], outputs.ne[1]});
            if (p.pipeline) { gather_peer_expert_range(peer, routing.ids, gathered, (i % 2) * (E / 2), E / 2, p.primary.stream); }
            else { gather_peer_experts(peer, routing.ids, gathered, i, size(), p.primary.stream); }
        }
        ops::moe_combine(gathered, routing, shared, y, p.primary.stream);
        return;
    }
    // Each stage has its own root scratch/stream. Inactive owners contribute
    // exact zeros without waiting on the other stage's producer events.
    Tensor root(p.ranks.front()->output.p, DType::FP32, {H, T});
    ops::moe_combine_owner(outputs, routing, 0, size(), root, p.primary.stream, p.primary_cache->owners(layer));
    Tensor partials(outputs.data, DType::FP32, {H, T, size()});
    copy_peer_bytes(partials.data, root.data, root.bytes(), p.primary.stream);
    for (int i = 1; i < size(); ++i) {
        auto destination = partials.slice(2, i, 1);
        if (!owns(i)) { CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), p.primary.stream)); continue; }
        auto& r = *p.ranks[i]; CUDA_CHECK(cudaStreamWaitEvent(p.primary.stream, r.experts_done, 0));
        copy_peer_bytes(destination.data, r.expert_partials.p, destination.bytes(), p.primary.stream);
    }
    ops::moe_combine_partials(partials, routing, shared, y, p.primary.stream);
}
void ParallelExecution::resident_experts(const Tensor& x, const ops::MoeRouting& routing, const MtpParameters& m, Tensor& output) {
    auto& p = *impl_; RestoreDevice restore{p.primary.device}; p.start(&x);
    const int T = x.ne[1], K = routing.ids.ne[0];
    const auto& c = p.parameters.model.config().text;
    for (int i = 1; i <= size(); ++i) {
        const int r = i % size(); auto& peer = *p.ranks[r];
        Tensor local_x = p.input(r, x);
        Tensor ids = routing.ids;
        if (r) {
            copy_peer_bytes(peer.expert_ids.p, routing.ids.data, routing.ids.bytes(), peer.device->stream);
            ids = Tensor(peer.expert_ids.p, DType::I32, {K, T});
        }
        const auto& gate = m.parallel_experts_gate_up->shards[r];
        Tensor local(peer.expert_outputs.p, DType::BF16, {x.ne[0], K * T});
        ops::resident_moe_experts(local_x, ids, gate.weight, m.parallel_experts_down->shards[r].weight,
                                  c.moe.experts, c.moe.intermediate, peer.resident_work.p, peer.resident_work.bytes,
                                  local, peer.device->stream, gate.units.begin, gate.units.count);
        p.done(r);
    }
    for (int i = 0; i < size(); ++i) {
        p.wait(i); const auto units = m.parallel_experts_gate_up->shards[i].units;
        Tensor local(p.ranks[i]->expert_outputs.p, DType::BF16, {x.ne[0], K * T});
        gather_peer_expert_range(local, routing.ids, output, units.begin, units.count, p.primary.stream);
    }
}
} // namespace infernix::models::qwen4_exp::execution

namespace infernix::models::qwen4_exp::execution {
bool ParallelExecution::pipeline() const { return impl_->pipeline; }
}
