// Production artifact-to-GPU shards, exact encoded ownership, and distributed
// projection/SwiGLU math against independent FP64 decoded-weight oracles.
#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/peer_copy.h"
#include "models/qwen4_exp/parallel_weights.h"
#include "models/qwen4_exp/head_partition.h"
#include "models/qwen4_exp/execution/parallel_projection.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/hyper_connection.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

using namespace infernix;
namespace q = infernix::models::qwen4_exp;
namespace qw = infernix::test::quantized_weight;
namespace t = infernix::test;
using infernix::test::artifact_fixture::Fixture;
using infernix::test::artifact_fixture::Json;
namespace {
void require(bool ok, const char* what) { if (!ok) { throw std::runtime_error(what); } }
qw::PackedWeight make_weight(QType codec, int n, int k, std::uint32_t seed) {
    if (codec != QType::BF16) {
        return qw::make_patterned_weight(codec, n, k, seed,
            {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    }
    qw::PackedWeight w;
    w.weight.qtype = QType::BF16; w.weight.layout = QuantLayout::Contiguous; w.weight.ndim = 2;
    w.weight.n = w.weight.shape[0] = w.weight.padded_shape[0] = n;
    w.weight.k = w.weight.shape[1] = w.weight.padded_shape[1] = k;
    w.payload.resize(std::size_t(n) * k * 2);
    for (std::size_t i = 0; i < w.payload.size() / 2; ++i) {
        seed = seed * 1664525U + 1013904223U;
        const auto bits = t::f32_to_bf16(float(int(seed >> 16) - 32768) / (32768 * std::sqrt(float(k))));
        w.payload[i * 2] = bits & 255; w.payload[i * 2 + 1] = bits >> 8;
    }
    return w;
}
struct World {
    std::vector<int> ids;
    std::vector<std::unique_ptr<DeviceContext>> devices;
    std::vector<cudaEvent_t> events;
    cudaEvent_t start = nullptr;
    explicit World(int count) {
        for (int i = 0; i < count; ++i) {
            ids.push_back(i); devices.push_back(std::make_unique<DeviceContext>(i));
            cudaEvent_t e; CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); events.push_back(e);
        }
        for (int i = 1; i < count; ++i) for (int direction = 0; direction < 2; ++direction) {
            const int from = direction ? 0 : i, to = direction ? i : 0;
            CUDA_CHECK(cudaSetDevice(from));
            const auto result = cudaDeviceEnablePeerAccess(to, 0);
            if (result == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); } else { CUDA_CHECK(result); }
        }
        devices[0]->bind_to_current_thread(); CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
    }
    ~World() {
        for (std::size_t i = 0; i < devices.size(); ++i) { devices[i]->bind_to_current_thread_noexcept(); (void)cudaEventDestroy(events[i]); }
        devices[0]->bind_to_current_thread_noexcept(); (void)cudaEventDestroy(start);
    }
    template<class Compute, class Gather> void launch(Compute compute, Gather gather) {
        auto& root = *devices[0]; root.bind_to_current_thread(); CUDA_CHECK(cudaEventRecord(start, root.stream));
        for (std::size_t i = 0; i < devices.size(); ++i) {
            devices[i]->bind_to_current_thread();
            if (i) { CUDA_CHECK(cudaStreamWaitEvent(devices[i]->stream, start, 0)); }
            compute(i, devices[i]->stream); CUDA_CHECK(cudaEventRecord(events[i], devices[i]->stream));
        }
        root.bind_to_current_thread();
        for (std::size_t i = 0; i < devices.size(); ++i) {
            if (i) { CUDA_CHECK(cudaStreamWaitEvent(root.stream, events[i], 0)); }
            gather(i, root.stream);
        }
    }
};
struct Loaded {
    Fixture fixture;
    std::unique_ptr<artifact::Reader> reader;
    artifact::MaterializedArtifact backing;
    std::unique_ptr<q::ParallelWeightStorage> storage;
    const q::PartitionedWeight* weight;
    Loaded(World& world, const qw::PackedWeight& host, q::WeightPartition kind, int groups,
           std::vector<std::vector<RowPartition>> rank_rows = {}) {
        fixture.payload.resize(host.payload.size());
        std::memcpy(fixture.payload.data(), host.payload.data(), host.payload.size());
        const char* format = host.weight.qtype == QType::BF16 ? "bf16" : host.weight.qtype == QType::Q8_G32_FP16 ? "q8_g32_fp16" : "q4_g64_fp16";
        fixture.root["components"] = {{"text", {{"config", Json::object()}, {"resources", Json::object()}}}};
        fixture.root["objects"] = Json::array({{{"id", "matrix"}, {"kind", "tensor"}, {"format", format},
            {"layout", host.weight.qtype == QType::BF16 ? "contiguous_le_v1" : "row_split_k128_v1"},
            {"shape", {host.weight.n, host.weight.k}}, {"offset", 0}, {"bytes", host.payload.size()}}});
        fixture.root["bindings"] = {{"matrix", {{"object", "matrix"}}}};
        fixture.root["uses"] = Json::array();
        // A file split inside a row verifies that direct shard reads retain framing semantics.
        const auto split = std::min<std::size_t>(511, host.payload.size());
        fixture.root["files"] = Json::array({{{"path", nullptr}, {"payload_bytes", split}},
            {{"path", "weights-second.bin"}, {"payload_bytes", host.payload.size() - split}}});
        fixture.write();
        reader = std::make_unique<artifact::Reader>(fixture.entry);
        artifact::Binder binder(*reader);
        (void)binder.parameter("matrix", {std::uint64_t(host.weight.n), std::uint64_t(host.weight.k)}, artifact::Residency::Streamed);
        world.devices[0]->bind_to_current_thread();
        backing = artifact::materialize(*reader, std::move(binder).finish(), *world.devices[0]);
        const auto object = reader->find("matrix");
        const q::ParallelWeightPlan plan{object, kind, groups, std::move(rank_rows)};
        storage = std::make_unique<q::ParallelWeightStorage>(*reader, backing, std::span(&plan, 1), world.ids, 0);
        weight = storage->find(&backing.streamed_parent(object));
        require(weight && weight->logical.qdata == nullptr && !backing.has_device(object), "full dense matrix was retained");
        const auto& geometry = reader->geometry(object);
        const std::size_t row_bytes = host.weight.qtype == QType::BF16 ? host.weight.k * 2 : geometry.code_bytes_per_row;
        std::vector<int> coverage(host.weight.n);
        std::uint64_t allocated = 0;
        for (const auto& shard : weight->shards) {
            if (!shard.weight.n) {
                require(!shard.weight.qdata && !shard.codes.bytes && !shard.scales.bytes, "nonowner allocated weight bytes");
                continue;
            }
            CUDA_CHECK(cudaSetDevice(shard.device));
            cudaPointerAttributes attr{}; CUDA_CHECK(cudaPointerGetAttributes(&attr, shard.weight.qdata));
            require(attr.type == cudaMemoryTypeDevice && attr.device == shard.device, "shard is on another device");
            std::vector<std::uint8_t> codes(shard.codes.bytes), scales(shard.scales.bytes);
            shard.codes.copy_to_host(codes.data(), codes.size());
            if (!scales.empty()) { shard.scales.copy_to_host(scales.data(), scales.size()); }
            std::size_t local = 0;
            for (const auto segment : shard.segments) for (int r = 0; r < segment.count; ++r, ++local) {
                const auto row = segment.begin + r; ++coverage[row];
                require(std::memcmp(codes.data() + local * row_bytes, host.payload.data() + row * row_bytes, row_bytes) == 0,
                        "shard code rows differ from artifact");
                if (!scales.empty()) {
                    require(std::memcmp(scales.data() + local * geometry.scale_bytes_per_row,
                        host.payload.data() + geometry.scale_offset + row * geometry.scale_bytes_per_row,
                        geometry.scale_bytes_per_row) == 0, "shard scale rows differ from artifact");
                }
            }
            allocated += storage->bytes(shard.device);
        }
        require(std::all_of(coverage.begin(), coverage.end(), [](int n) { return n == 1; }), "rows overlap or are missing");
        require(allocated == std::uint64_t(host.weight.n) * (row_bytes + geometry.scale_bytes_per_row), "hidden dense replica allocation");
        world.devices[0]->bind_to_current_thread();
    }
};

void projection(World& world, QType codec, int N, int K, int T, DType dtype, bool paired = false,
                bool stream_features = false, int semantic_heads = 0, bool pipeline = false, bool fixed = false) {
    auto host = make_weight(codec, N, K, 2917);
    std::vector<std::vector<RowPartition>> rows;
    if (semantic_heads == 1 || semantic_heads == 4) {
        rows.resize(world.ids.size());
        for (int head = 0; head < 2; ++head) {
            rows[q::attention_head_owner(semantic_heads == 4 ? 7 : 1, head, 2, world.ids.size(), pipeline, fixed)].push_back({head * N / 2, N / 2});
        }
    } else if (semantic_heads == 2) {
        rows.resize(world.ids.size());
        for (int rank = 0; rank < int(rows.size()); ++rank) {
            if (!q::pipeline_owns(0, rank, 36, rows.size(), pipeline)) { continue; }
            const auto h = q::gdn_head_partition(16, 48, rank, rows.size(), pipeline).values;
            rows[rank] = {{h.begin, h.count}, {48 + h.begin, h.count}};
        }
    } else if (semantic_heads == 3) {
        rows.resize(world.ids.size());
        for (int rank = 0; rank < int(rows.size()); ++rank) {
            if (!q::pipeline_owns(0, rank, 36, rows.size(), pipeline)) { continue; }
            const auto h = q::gdn_head_partition(16, 48, rank, rows.size(), pipeline);
            const int k = h.keys.begin * 128, nk = h.keys.count * 128;
            const int v = h.values.begin * 128, nv = h.values.count * 128;
            rows[rank] = {{k, nk}, {2048 + k, nk}, {4096 + v, nv}, {10240 + v, nv}};
        }
    }
    Loaded loaded(world, host, paired ? q::WeightPartition::Paired :
        stream_features ? q::WeightPartition::StreamFeatures : q::WeightPartition::Rows,
        paired ? 2 : stream_features ? 4 : 1, std::move(rows));
    const int out_n = paired ? N / 2 : N;
    std::vector<std::uint16_t> values(std::size_t(K) * T);
    std::uint32_t random = 5327;
    for (auto& value : values) { random = random * 1664525U + 1013904223U; value = t::f32_to_bf16(float(int(random >> 16) - 32768) / 32768); }
    std::vector<DeviceBuffer> input, output;
    std::vector<std::unique_ptr<WorkspaceArena>> work;
    for (std::size_t i = 0; i < world.devices.size(); ++i) {
        world.devices[i]->bind_to_current_thread();
        input.emplace_back(values.size() * 2); input.back().copy_from_host(values.data(), values.size() * 2);
        const auto& shard = loaded.weight->shards[i];
        output.emplace_back(std::size_t(shard.weight.n) * T * 4); work.push_back(std::make_unique<WorkspaceArena>(8 << 20));
    }
    world.devices[0]->bind_to_current_thread(); DeviceBuffer combined(std::size_t(out_n) * T * 4);
    Tensor all(combined.p, dtype, {out_n, T});
    const auto call = [&] {
        world.launch([&](int i, cudaStream_t stream) {
            const auto& shard = loaded.weight->shards[i];
            if (!shard.weight.n) { return; }
            Tensor x(input[i].p, DType::BF16, {K, T}), y(output[i].p, dtype, {paired ? shard.units.count : shard.weight.n, T});
            work[i]->reset();
            if (paired) { ops::linear_swiglu(x, shard.weight, y, ops::LinearPolicy::A16Only, *work[i], stream); }
            else { q::execution::parallel_projection(x, shard.weight, y, ops::LinearPolicy::A16Only, *work[i], stream); }
        }, [&](int i, cudaStream_t stream) {
            const auto& shard = loaded.weight->shards[i];
            if (!shard.weight.n) { return; }
            Tensor y(output[i].p, dtype, {paired ? shard.units.count : shard.weight.n, T});
            if (paired) { gather_peer_rows(y, all, shard.units.begin, stream); }
            else {
                int first = 0;
                for (const auto segment : shard.segments) {
                    Tensor part = y.slice(0, first, segment.count);
                    gather_peer_rows(part, all, segment.begin, stream);
                    first += segment.count;
                }
            }
        });
    };
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    call(); world.devices[0]->synchronize();
    std::vector<std::byte> eager(all.bytes()); combined.copy_to_host(eager.data(), eager.size());
    definition.capture(world.devices[0]->stream, call); graph.instantiate(definition);
    for (int replay = 0; replay < 3; ++replay) { graph.launch(world.devices[0]->stream); }
    world.devices[0]->synchronize();
    std::vector<std::byte> captured(all.bytes()); combined.copy_to_host(captured.data(), captured.size());
    require(eager == captured, "projection eager/graph results differ");
    std::vector<double> actual, reference;
    // Wide MTP profiles sample columns through the final boundary as well as
    // rows; ordinary profiles retain the full column oracle.
    for (int r = 0; r < out_n; r += std::max(1, out_n / 67))
      for (int col = 0; col < T; col += T > 512 && col + 1 < T ? std::min(127, T - 1 - col) : 1) {
        double sum = 0, up = 0, magnitude = 0;
        for (int k = 0; k < K; ++k) {
            const double x = t::bf16_to_f32(values[std::size_t(col) * K + k]);
            const double value = qw::logical_weight_fp64(host, r, k) * x;
            sum += value; magnitude += std::abs(value);
            if (paired) { up += qw::logical_weight_fp64(host, r + out_n, k) * x; }
        }
        if (paired) { sum = sum / (1 + std::exp(-sum)) * up; }
        double got;
        if (dtype == DType::BF16) { std::uint16_t bits; std::memcpy(&bits, captured.data() + (std::size_t(col) * out_n + r) * 2, 2); got = t::bf16_to_f32(bits); }
        else { float value; std::memcpy(&value, captured.data() + (std::size_t(col) * out_n + r) * 4, 4); got = value;
            require(std::abs(got - sum) <= magnitude * K * std::ldexp(1., -24) + 1e-30, "FP32 projection exceeds accumulation bound"); }
        actual.push_back(got); reference.push_back(sum);
    }
    const std::string tag = "TP" + std::to_string(world.ids.size()) + (paired ? " SwiGLU " : " projection ") +
        std::to_string(N) + "x" + std::to_string(K) + " T=" + std::to_string(T) + " codec=" + std::to_string(int(codec));
    if (dtype == DType::BF16) { require(t::verify_reduction(tag, actual, reference, {1. / 256, 1. / 256, 2. / 256}) == 0, "BF16 projection exceeds oracle criterion"); }
    std::printf("PASS %s: direct owned shards, exact encoded rows, FP64 oracle and graph replay\n", tag.c_str());
}
void wide_collapse(World& world) {
    constexpr int H = 2560, S = 4, T = 65;
    std::vector<std::uint16_t> norm(H * S * T), logits(norm.size());
    for (std::size_t j = 0; j < norm.size(); ++j) {
        norm[j] = t::f32_to_bf16(float(int(j * 17 % 101) - 50) / 29);
        logits[j] = t::f32_to_bf16(float(int(j * 31 % 127) - 63) / 17);
    }
    std::vector<DeviceBuffer> nd, ud, packed_n, packed_u, result;
    for (std::size_t i = 0; i < world.ids.size(); ++i) {
        world.devices[i]->bind_to_current_thread();
        const auto part = partition_rows(H, i, world.ids.size(), 16);
        nd.emplace_back(norm.size() * 2); nd.back().copy_from_host(norm.data(), norm.size() * 2);
        ud.emplace_back(logits.size() * 2); ud.back().copy_from_host(logits.data(), logits.size() * 2);
        packed_n.emplace_back(S * part.count * T * 2); packed_u.emplace_back(packed_n.back().bytes);
        result.emplace_back(part.count * T * 2);
    }
    world.devices[0]->bind_to_current_thread(); DeviceBuffer combined(H * T * 2);
    Tensor all(combined.p, DType::BF16, {H, T});
    const auto call = [&] {
        world.launch([&](int i, cudaStream_t stream) {
            const auto p = partition_rows(H, i, world.ids.size(), 16);
            Tensor n(nd[i].p, DType::BF16, {S * H, T}), u(ud[i].p, DType::BF16, {S * H, T});
            Tensor pn(packed_n[i].p, DType::BF16, {S * p.count, T}), pu(packed_u[i].p, DType::BF16, {S * p.count, T});
            for (int s = 0; s < S; ++s) {
                const auto ns = n.slice(0, s * H + p.begin, p.count), us = u.slice(0, s * H + p.begin, p.count);
                gather_peer_rows(ns, pn, s * p.count, stream); gather_peer_rows(us, pu, s * p.count, stream);
            }
            Tensor out(result[i].p, DType::BF16, {p.count, T});
            ops::hyper_connection_collapse(pu, pn, S, out, stream);
        }, [&](int i, cudaStream_t stream) {
            const auto p = partition_rows(H, i, world.ids.size(), 16);
            Tensor out(result[i].p, DType::BF16, {p.count, T}); gather_peer_rows(out, all, p.begin, stream);
        });
    };
    call(); world.devices[0]->synchronize(); std::vector<std::uint16_t> eager(H * T), got(H * T);
    combined.copy_to_host(eager.data(), combined.bytes);
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    definition.capture(world.devices[0]->stream, call); graph.instantiate(definition); graph.launch(world.devices[0]->stream);
    world.devices[0]->synchronize(); combined.copy_to_host(got.data(), combined.bytes);
    require(got == eager, "wide owner HC collapse graph/eager mismatch");
    std::vector<double> actual, reference;
    for (int col = 0; col < T; ++col) for (int h = 0; h < H; ++h) {
        double sum = 0;
        for (int s = 0; s < S; ++s) {
            const auto j = col * S * H + s * H + h;
            sum += double(t::bf16_to_f32(norm[j])) / (1 + std::exp(-double(t::bf16_to_f32(logits[j]))));
        }
        reference.push_back(sum / S); actual.push_back(t::bf16_to_f32(got[col * H + h]));
    }
    require(t::verify_reduction("owner-local wide HC collapse", actual, reference, {1. / 256, 1. / 256, 2. / 256}) == 0,
            "wide owner HC collapse exceeds FP64 oracle");
}

void hyper_connection(World& world, QType codec, int T, bool injection) {
    constexpr int H = 2560, S = 4, W = S * H, rank = 320;
    const int rows = rank + (injection ? S : 0);
    auto down = make_weight(codec, rows, W, 7331);
    auto up = make_weight(codec, W, rank, 7993);
    Loaded d(world, down, q::WeightPartition::Rows, 1), u(world, up, q::WeightPartition::StreamFeatures, S);
    std::vector<std::uint16_t> residual(W * T), norm(W);
    std::uint32_t state = 4441;
    const auto random = [&]() { state = state * 1664525U + 1013904223U; return float(int(state >> 16) - 32768) / 32768; };
    for (auto& v : residual) { v = t::f32_to_bf16(random()); }
    for (auto& v : norm) { v = t::f32_to_bf16(random() / 4); }
    std::vector<DeviceBuffer> rd, nd, partial, inv, local_out;
    for (std::size_t i = 0; i < world.ids.size(); ++i) {
        world.devices[i]->bind_to_current_thread();
        rd.emplace_back(residual.size() * 2); rd.back().copy_from_host(residual.data(), residual.size() * 2);
        nd.emplace_back(norm.size() * 2); nd.back().copy_from_host(norm.data(), norm.size() * 2);
        partial.emplace_back(S * rows * T * 4); inv.emplace_back(S * T * 4);
        local_out.emplace_back(u.weight->shards[i].units.count * T * 2);
    }
    world.devices[0]->bind_to_current_thread();
    DeviceBuffer all_partial(S * rows * T * 4), all_out(H * T * 2), all_inject(S * T * 4);
    Tensor out(all_out.p, DType::BF16, {H, T}), inject(all_inject.p, DType::FP32, {S, T});
    const auto call = [&] {
        world.launch([&](int i, cudaStream_t stream) {
            if (i) { copy_peer_bytes(nd[i].p, nd[0].p, norm.size() * 2, stream); }
            Tensor r(rd[i].p, DType::BF16, {W, T}), n(nd[i].p, DType::BF16, {W});
            ops::hyper_connection_down_partition(r, n, d.weight->shards[i].weight, 1e-6f, partial[i].p, inv[i].p, stream);
        }, [&](int i, cudaStream_t stream) {
            const auto segment = d.weight->shards[i].segments.front();
            for (int s = 0; s < S; ++s) {
                auto* dst = static_cast<float*>(all_partial.p) + (s * rows + segment.begin) * T;
                auto* src = static_cast<float*>(partial[i].p) + s * segment.count * T;
                copy_peer_bytes(dst, src, segment.count * T * 4, stream);
            }
        });
        world.launch([&](int i, cudaStream_t stream) {
            copy_peer_bytes(partial[i].p, all_partial.p, S * rows * T * 4, stream);
            const auto& shard = u.weight->shards[i];
            Tensor r(rd[i].p, DType::BF16, {W, T}), n(nd[i].p, DType::BF16, {W});
            Tensor y(local_out[i].p, DType::BF16, {shard.units.count, T});
            ops::hyper_connection_up_partition(r, n, shard.weight, rows, rank, shard.units.begin,
                                               partial[i].p, inv[i].p, y, i == 0 && injection ? &inject : nullptr, stream);
        }, [&](int i, cudaStream_t stream) {
            const auto units = u.weight->shards[i].units;
            Tensor y(local_out[i].p, DType::BF16, {units.count, T}); gather_peer_rows(y, out, units.begin, stream);
        });
    };
    call(); world.devices[0]->synchronize();
    std::vector<std::uint16_t> eager(H * T), got(H * T);
    all_out.copy_to_host(eager.data(), eager.size() * 2);
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    definition.capture(world.devices[0]->stream, call); graph.instantiate(definition);
    graph.launch(world.devices[0]->stream); graph.launch(world.devices[0]->stream); world.devices[0]->synchronize();
    all_out.copy_to_host(got.data(), got.size() * 2); require(eager == got, "HC graph/eager mismatch");
    std::vector<float> injections(S * T); if (injection) { all_inject.copy_to_host(injections.data(), injections.size() * 4); }
    std::vector<double> reference(H * T), normalized(W), z(rows), mix(rank);
    for (int col = 0; col < T; ++col) {
        for (int s = 0; s < S; ++s) {
            double ss = 0;
            for (int h = 0; h < H; ++h) { const double x = t::bf16_to_f32(residual[col * W + s * H + h]); ss += x * x; }
            const double inverse = 1 / std::sqrt(ss / H + double(1e-6f));
            for (int h = 0; h < H; ++h) { const int row = s * H + h;
                normalized[row] = t::bf16_to_f32(residual[col * W + row]) * inverse * (1 + double(t::bf16_to_f32(norm[row]))); }
        }
        for (int row = 0; row < rows; ++row) {
            double sum = 0; for (int k = 0; k < W; ++k) { sum += qw::logical_weight_fp64(down, row, k) * normalized[k]; }
            z[row] = sum / S;
            if (row < rank) { mix[row] = z[row] / (1 + std::exp(-z[row])); }
            else { const double expected = 2 / (1 + std::exp(-z[row]));
                require(std::abs(injections[col * S + row - rank] - expected) <= 1e-5 * std::abs(expected) + 1e-6, "HC inject oracle mismatch"); }
        }
        for (int h = 0; h < H; ++h) {
            double value = 0;
            for (int s = 0; s < S; ++s) {
                double dot = 0; const int row = s * H + h;
                for (int k = 0; k < rank; ++k) { dot += qw::logical_weight_fp64(up, row, k) * mix[k]; }
                value += normalized[row] / (1 + std::exp(-dot));
            }
            reference[col * H + h] = value / S;
        }
    }
    double ss = 0; for (double v : reference) { ss += v * v; }
    const double rms = std::sqrt(ss / reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require(std::abs(t::bf16_to_f32(got[i]) - reference[i]) <= std::abs(reference[i]) / 256 + rms / 4096,
                "HC output oracle mismatch");
    }
    std::printf("PASS TP HC codec=%d T=%d inject=%d: owned stream/row shards, FP64 oracle, graph replay\n", int(codec), T, injection);
}
}
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return 77; }
        count = std::min(count, 4); World world(count);
        wide_collapse(world);
        if (count == 4) for (auto codec : {QType::BF16, QType::Q8_G32_FP16}) {
            projection(world, codec, 512, 2560, 5, DType::BF16, false, false, 1, false, true);
            projection(world, codec, 512, 2560, 2048, DType::BF16, false, false, 4, false, true);
            projection(world, codec, 512, 2560, 5, DType::BF16, false, false, 1, true);
            projection(world, codec, 512, 2560, 5, DType::BF16, false, false, 4, true);
            projection(world, codec, 96, 2560, 2048, DType::BF16, false, false, 2, true);
            projection(world, codec, 16384, 2560, 17, DType::BF16, false, false, 3, true);
        }
        for (auto codec : {QType::BF16, QType::Q8_G32_FP16}) {
            projection(world, codec, 512, 2560, 5, DType::BF16, false, false, true);
            projection(world, codec, 512, 2560, 5, DType::BF16, false, false, 4);
            projection(world, codec, 96, 2560, 17, DType::BF16, false, false, 2);
            projection(world, codec, 16384, 2560, 1, DType::BF16, false, false, 3);
            projection(world, codec, 16384, 2560, 17, DType::BF16, false, false, 3);
            hyper_connection(world, codec, 1, true);
            hyper_connection(world, codec, 8, false);
            projection(world, codec, 48, 2560, 1, DType::BF16);
            projection(world, codec, 513, 2560, 8, DType::FP32);
            projection(world, codec, 13952, 2560, 17, DType::BF16);
            projection(world, codec, 2560, 640, 65, DType::BF16);
            // Four residual streams flattened by MTP's 512-cell prefill.
            projection(world, codec, 2560, 2560, 2048, DType::BF16);
            // Wide HC uses composed BF16 boundaries and noncontiguous stream-row shards.
            projection(world, codec, 324, 10240, 17, DType::BF16);
            projection(world, codec, 10240, 320, 17, DType::BF16, false, true);
            if (codec == QType::BF16) {
                projection(world, codec, 96, 2560, 2048, DType::BF16, false, false, 2);
                // 48 local rows selects the large-shared-memory route on every GPU.
                projection(world, codec, 192, 2560, 2048, DType::BF16);
                projection(world, codec, 324, 10240, 257, DType::BF16);
                projection(world, codec, 10240, 320, 257, DType::BF16, false, true);
                // Compact K and uneven row tails exercise both MMA predicates.
                projection(world, codec, 100, 72, 129, DType::BF16);
            }
            projection(world, codec, 1280, 2560, 8, DType::BF16, true);
            if (codec == QType::Q8_G32_FP16) { projection(world, codec, 1280, 2560, 65, DType::BF16, true); }
        }
        // Expert and HC stream layouts must also own every code/scale row once.
        for (auto codec : {QType::BF16, QType::Q8_G32_FP16}) {
            auto host = make_weight(codec, 10240, 320, 3911);
            Loaded check(world, host, q::WeightPartition::StreamFeatures, 4);
        }
        for (auto codec : {QType::Q4_G64_FP16, QType::Q8_G32_FP16}) {
            auto host = qw::make_patterned_weight(codec, 12 * 1280, 2560, 4121, {});
            Loaded check(world, host, q::WeightPartition::Experts, 12);
        }
        // TP3 has uneven 208/224-feature SwiGLU shards and tail HC row shards.
        // Exercise each advertised rank count, including all three Q8 schedules.
        for (int ranks = 2; ranks < count; ++ranks) {
            World subset(ranks);
            for (auto codec : {QType::BF16, QType::Q8_G32_FP16}) {
                projection(subset, codec, 96, 2560, 17, DType::BF16, false, false, 2);
                projection(subset, codec, 16384, 2560, 5, DType::BF16, false, false, 3);
                hyper_connection(subset, codec, 1, true);
                projection(subset, codec, 1280, 2560, 8, DType::BF16, true);
                if (codec == QType::Q8_G32_FP16) {
                    projection(subset, codec, 1280, 2560, 17, DType::BF16, true);
                    projection(subset, codec, 1280, 2560, 65, DType::BF16, true);
                }
            }
        }
        std::puts("PASS parallel weight storage and projection qualification"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
