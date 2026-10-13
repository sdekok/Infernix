// Exact transport and ownership tests. These exercise the production peer gather,
// dispatch and multi-device stream-capture operations without a model download.
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/parallel_partition.h"
#include "core/peer_copy.h"
#include "core/vmm_arena.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace infernix;

namespace {
void require(bool good, const char* what) { if (!good) { throw std::runtime_error(what); } }

struct Rank {
    DeviceContext device;
    DeviceBuffer input, output, ids, dispatch;
    cudaEvent_t done = nullptr;
    explicit Rank(int id) : device(id) { CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); }
    ~Rank() { device.bind_to_current_thread_noexcept(); if (done) { (void)cudaEventDestroy(done); } }
};

void test_rows(std::vector<std::unique_ptr<Rank>>& ranks, int columns = 17) {
    auto& root = ranks.front()->device;
    root.bind_to_current_thread();
    constexpr int rows = 324;
    DeviceBuffer output(rows * columns * 4);
    Tensor out(output.p, DType::FP32, {rows, columns});
    cudaEvent_t start = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
    const int count = static_cast<int>(ranks.size());
    for (int r = 0; r < count; ++r) {
        auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
        const auto p = partition_rows(rows, r, count);
        rank.input = DeviceBuffer(std::size_t(p.count) * columns * 4);
    }
    auto body = [&] {
        root.bind_to_current_thread();
        CUDA_CHECK(cudaEventRecord(start, root.stream));
        for (int r = 1; r < count; ++r) {
            auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
            CUDA_CHECK(cudaStreamWaitEvent(rank.device.stream, start, 0));
            // A device operation is essential: otherwise capture could collapse
            // an empty peer branch and fail to test a multi-device graph.
            CUDA_CHECK(cudaMemsetAsync(rank.output.p, 0, 4, rank.device.stream));
            CUDA_CHECK(cudaEventRecord(rank.done, rank.device.stream));
        }
        root.bind_to_current_thread();
        for (int r = 0; r < count; ++r) {
            auto& rank = *ranks[r];
            if (r) { CUDA_CHECK(cudaStreamWaitEvent(root.stream, rank.done, 0)); }
            const auto p = partition_rows(rows, r, count);
            Tensor peer(rank.input.p, DType::FP32, {p.count, columns});
            gather_peer_rows(peer, out, p.begin, root.stream);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    for (int pass = 0; pass < 4; ++pass) {
        for (int r = 0; r < count; ++r) {
            auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
            const auto p = partition_rows(rows, r, count);
            std::vector<float> values(std::size_t(p.count) * columns);
            for (int t = 0; t < columns; ++t) for (int n = 0; n < p.count; ++n) {
                values[std::size_t(t) * p.count + n] = float(pass * 100000 + t * rows + p.begin + n);
            }
            rank.input.copy_from_host(values.data(), values.size() * 4);
        }
        root.bind_to_current_thread();
        if (pass == 0) { body(); }
        else {
            if (!graph.ready()) { definition.capture(root.stream, body); graph.instantiate(definition); }
            graph.launch(root.stream);
        }
        root.synchronize();
        std::vector<float> got(rows * columns);
        output.copy_to_host(got.data(), got.size() * 4);
        for (std::size_t i = 0; i < got.size(); ++i) { require(got[i] == float(pass * 100000 + i), "peer row gather mismatch"); }
    }
    CUDA_CHECK(cudaEventDestroy(start));
}

// Direct all-gather follows the GDN owner-to-output-projection dependency DAG.
void test_allgather(std::vector<std::unique_ptr<Rank>>& ranks, int columns = 5, int owners = 0, int first_owner = 0) {
    const int count = ranks.size(), rows = 6144;
    if (!owners) { owners = count; }
    const auto owned = [&](int rank) {
        const int logical = (rank - first_owner + count) % count;
        return logical < owners ? partition_rows(rows, logical, owners) : RowPartition{0, 0};
    };
    auto& root = ranks[0]->device;
    root.bind_to_current_thread();
    cudaEvent_t start;
    CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDisableTiming));
    std::vector<cudaEvent_t> joined(count);
    for (int r = 0; r < count; ++r) {
        auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
        const auto part = owned(r);
        rank.input = DeviceBuffer(part.count * columns * 4);
        rank.output = DeviceBuffer(rows * columns * 4);
        CUDA_CHECK(cudaEventCreateWithFlags(&joined[r], cudaEventDisableTiming));
    }
    const auto call = [&] {
        root.bind_to_current_thread(); CUDA_CHECK(cudaEventRecord(start, root.stream));
        for (int r = 0; r < count; ++r) {
            auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
            if (r) { CUDA_CHECK(cudaStreamWaitEvent(rank.device.stream, start, 0)); }
            CUDA_CHECK(cudaMemsetAsync(rank.output.p, 0, rank.output.bytes, rank.device.stream));
            CUDA_CHECK(cudaEventRecord(rank.done, rank.device.stream));
        }
        for (int r = 0; r < count; ++r) {
            auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
            Tensor out(rank.output.p, DType::FP32, {rows, columns});
            for (int owner = 0; owner < count; ++owner) {
                if (owner != r) { CUDA_CHECK(cudaStreamWaitEvent(rank.device.stream, ranks[owner]->done, 0)); }
                const auto part = owned(owner);
                if (!part.count) { continue; }
                Tensor input(ranks[owner]->input.p, DType::FP32, {part.count, columns});
                gather_peer_rows(input, out, part.begin, rank.device.stream);
            }
            CUDA_CHECK(cudaEventRecord(joined[r], rank.device.stream));
        }
        root.bind_to_current_thread();
        for (int r = 1; r < count; ++r) { CUDA_CHECK(cudaStreamWaitEvent(root.stream, joined[r], 0)); }
    };
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    for (int pass = 0; pass < 3; ++pass) {
        for (int r = 0; r < count; ++r) {
            auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
            const auto part = owned(r);
            std::vector<float> input(part.count * columns);
            for (int t = 0; t < columns; ++t) for (int n = 0; n < part.count; ++n) {
                input[t * part.count + n] = float(pass * 100000 + t * rows + part.begin + n);
            }
            if (!input.empty()) { rank.input.copy_from_host(input.data(), input.size() * 4); }
        }
        root.bind_to_current_thread();
        if (!pass) { call(); }
        else {
            if (!graph.ready()) { definition.capture(root.stream, call); graph.instantiate(definition); }
            graph.launch(root.stream);
        }
        root.synchronize();
        for (auto& rank : ranks) {
            rank->device.bind_to_current_thread();
            std::vector<float> output(rows * columns); rank->output.copy_to_host(output.data(), output.size() * 4);
            for (int i = 0; i < rows * columns; ++i) { require(output[i] == float(pass * 100000 + i), "direct all-gather mismatch"); }
        }
    }
    for (int r = 0; r < count; ++r) { ranks[r]->device.bind_to_current_thread(); CUDA_CHECK(cudaEventDestroy(joined[r])); }
    root.bind_to_current_thread(); CUDA_CHECK(cudaEventDestroy(start));
}

void test_bf16_rows(std::vector<std::unique_ptr<Rank>>& ranks) {
    constexpr int rows = 1280, pitch = 4096, columns = 257, first = 768;
    auto& root = ranks.front()->device; root.bind_to_current_thread();
    DeviceBuffer source(std::size_t(pitch) * columns * 2);
    std::vector<std::uint16_t> input(std::size_t(pitch) * columns);
    for (std::size_t j = 0; j < input.size(); ++j) { input[j] = std::uint16_t(j * 7919 + 31); }
    source.copy_from_host(input.data(), input.size() * 2);
    for (auto& rank : ranks) {
        rank->device.bind_to_current_thread(); DeviceBuffer output(source.bytes);
        Tensor src(source.p, DType::BF16, {pitch, columns}), dst(output.p, DType::BF16, {pitch, columns});
        const auto body = [&] {
            CUDA_CHECK(cudaMemsetAsync(output.p, 0, output.bytes, rank->device.stream));
            auto part = src.slice(0, 256, rows);
            gather_peer_rows(part, dst, first, rank->device.stream);
        };
        DecodeGraphDefinition definition; DecodeGraphExecutable graph;
        for (int pass = 0; pass < 2; ++pass) {
            if (!pass) { body(); }
            else { definition.capture(rank->device.stream, body); graph.instantiate(definition); graph.launch(rank->device.stream); }
            rank->device.synchronize(); std::vector<std::uint16_t> got(input.size());
            output.copy_to_host(got.data(), output.bytes);
            for (int t = 0; t < columns; ++t) for (int r = 0; r < pitch; ++r) {
                const auto expected = r >= first && r < first + rows ? input[t * pitch + r - first + 256] : 0;
                require(got[t * pitch + r] == expected, "wide BF16 strided gather or untouched row mismatch");
            }
        }
    }
    root.bind_to_current_thread();
}

void test_broadcast(std::vector<std::unique_ptr<Rank>>& ranks) {
    auto& root = ranks.front()->device; root.bind_to_current_thread();
    // Uneven word-aligned pieces, large enough to use DMA; vary the source
    // between replays to detect stale readiness and source-lifetime mistakes.
    constexpr std::size_t words = (4ULL << 20) + 3;
    DeviceBuffer source(words * 4);
    cudaEvent_t ready; CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    std::vector<PeerBroadcastTarget> targets;
    for (std::size_t i = 1; i < ranks.size(); ++i) {
        auto& r = *ranks[i]; r.device.bind_to_current_thread(); r.output = DeviceBuffer(words * 4);
        r.input = DeviceBuffer(words * 4);
        cudaEvent_t piece; CUDA_CHECK(cudaEventCreateWithFlags(&piece, cudaEventDisableTiming));
        targets.push_back({r.output.p, r.device.device, r.device.stream, piece});
    }
    for (std::size_t count = 1; count <= targets.size(); ++count) {
        DecodeGraphDefinition definition; DecodeGraphExecutable graph;
        const auto body = [&] {
            root.bind_to_current_thread(); CUDA_CHECK(cudaEventRecord(ready, root.stream));
            broadcast_peer_bytes(source.p, source.bytes, {targets.data(), count}, ready);
            for (std::size_t i = 1; i <= count; ++i) {
                auto& r = *ranks[i]; r.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(r.input.p, r.output.p, r.output.bytes, cudaMemcpyDeviceToDevice, r.device.stream));
                // Consumers may reuse their input before other devices finish
                // unrelated compute; all broadcast readers must already be done.
                CUDA_CHECK(cudaMemsetAsync(r.output.p, 0, r.output.bytes, r.device.stream));
                CUDA_CHECK(cudaEventRecord(r.done, r.device.stream));
                root.bind_to_current_thread(); CUDA_CHECK(cudaStreamWaitEvent(root.stream, r.done, 0));
            }
        };
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<std::uint32_t> expected(words);
            for (std::size_t j = 0; j < words; ++j) { expected[j] = std::uint32_t(j * 7919 + pass * 413); }
            root.bind_to_current_thread(); source.copy_from_host(expected.data(), source.bytes);
            if (!pass) { body(); }
            else {
                if (!graph.ready()) { definition.capture(root.stream, body); graph.instantiate(definition); }
                graph.launch(root.stream);
            }
            root.synchronize();
            for (std::size_t i = 1; i <= count; ++i) {
                auto& r = *ranks[i]; r.device.bind_to_current_thread(); std::vector<std::uint32_t> got(words);
                r.input.copy_to_host(got.data(), r.input.bytes);
                require(got == expected, "distributed broadcast differs from source");
            }
        }
    }
    for (const auto& target : targets) { CUDA_CHECK(cudaSetDevice(target.device)); CUDA_CHECK(cudaEventDestroy(target.piece_ready)); }
    root.bind_to_current_thread(); CUDA_CHECK(cudaEventDestroy(ready));
}

void test_peer_arena(std::vector<std::unique_ptr<Rank>>& ranks, std::size_t owner) {
    auto& root = ranks.at(owner)->device; root.bind_to_current_thread();
    constexpr std::size_t chunk = 2ULL << 20;
    VmmArena arena(root.device, 2 * chunk, chunk);
    require(arena.map_chunk(), "initial arena map failed");
    for (std::size_t i = 0; i < ranks.size(); ++i) { arena.allow_access(ranks[i]->device.device); }
    for (int pass = 0; pass < 3; ++pass) {
        root.bind_to_current_thread();
        if (pass == 1) { require(arena.map_chunk(), "arena growth failed"); }
        if (pass == 2) { arena.unmap_chunk(); require(arena.map_chunk(), "arena remap failed"); }
        const std::size_t offset = pass ? chunk : 0;
        auto* source = static_cast<std::uint32_t*>(arena.base()) + offset / 4;
        std::vector<std::uint32_t> expected(20480);
        for (std::size_t j = 0; j < expected.size(); ++j) { expected[j] = pass * 10000 + j; }
        CUDA_CHECK(cudaMemcpy(source, expected.data(), expected.size() * 4, cudaMemcpyHostToDevice));
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            auto& rank = *ranks[i]; rank.device.bind_to_current_thread();
            const auto copy = [&] { copy_peer_bytes(rank.output.p, source, expected.size() * 4, rank.device.stream); };
            copy(); rank.device.synchronize();
            DecodeGraphDefinition definition; DecodeGraphExecutable graph;
            definition.capture(rank.device.stream, copy); graph.instantiate(definition);
            graph.launch(rank.device.stream); rank.device.synchronize();
            std::vector<std::uint32_t> got(expected.size()); rank.output.copy_to_host(got.data(), got.size() * 4);
            require(got == expected, "borrowed arena peer access mismatch");
        }
    }
    root.bind_to_current_thread();
}

void test_experts(std::vector<std::unique_ptr<Rank>>& ranks) {
    const int count = static_cast<int>(ranks.size());
    auto& root = ranks.front()->device;
    constexpr int experts = 512, hidden = 2560, top_k = 10;
    for (int columns : {1, 8, 129, 257}) {
        const int entries = top_k * columns;
        root.bind_to_current_thread();
        DeviceBuffer ids(entries * 4), combined(std::size_t(hidden) * entries * 2);
        Tensor output(combined.p, DType::BF16, {hidden, entries});
        Tensor id_view(ids.p, DType::I32, {top_k, columns});
        for (int pattern = 0; pattern < 2; ++pattern) {
            std::vector<std::int32_t> selected(entries);
            for (int i = 0; i < entries; ++i) { selected[i] = pattern ? (i * count) % experts : (i * 17) % experts; }
            root.bind_to_current_thread(); ids.copy_from_host(selected.data(), entries * 4);
            combined.fill(0xFF);
            for (int r = 0; r < count; ++r) {
                auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
                rank.ids = DeviceBuffer(entries * 4);
                rank.ids.copy_from_host(selected.data(), entries * 4);
                rank.dispatch = DeviceBuffer(ops::moe_dispatch_bytes(experts, entries));
                auto dispatch = ops::carve_moe_dispatch(rank.dispatch.p, experts, entries);
                ops::MoeRouting routing;
                routing.ids = Tensor(rank.ids.p, DType::I32, {top_k, columns});
                ops::moe_dispatch(routing, experts, dispatch, nullptr, rank.device.stream, r, count);
                rank.device.synchronize();
                std::vector<int> counts(experts), offsets(experts + 1), jobs(experts), positions(entries);
                int job_count = -1;
                CUDA_CHECK(cudaMemcpy(counts.data(), dispatch.counts, experts * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(offsets.data(), dispatch.offsets, (experts + 1) * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&job_count, dispatch.job_count, 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(jobs.data(), dispatch.jobs, experts * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(positions.data(), dispatch.entries, entries * 4, cudaMemcpyDeviceToHost));
                int total = 0, expected_jobs = 0;
                for (int e = 0; e < experts; ++e) {
                    const int expected = e % count == r ? static_cast<int>(std::count(selected.begin(), selected.end(), e)) : 0;
                    require(counts[e] == expected && offsets[e] == total, "expert ownership/count mismatch");
                    if (expected) { require(jobs[expected_jobs++] == e, "expert job order mismatch"); }
                    std::vector<int> found;
                    for (int n = 0; n < expected; ++n) { found.push_back(positions[total + n]); }
                    std::sort(found.begin(), found.end());
                    std::vector<int> oracle;
                    for (int n = 0; n < entries; ++n) if (selected[n] == e && e % count == r) { oracle.push_back(n); }
                    require(found == oracle, "expert entry permutation mismatch");
                    total += expected;
                }
                require(offsets.back() == total && job_count == expected_jobs, "expert dispatch terminal count mismatch");
                rank.output = DeviceBuffer(std::size_t(hidden) * entries * 2);
                std::vector<std::uint16_t> values(std::size_t(hidden) * entries, 0xFFFF);
                for (int e = 0; e < entries; ++e) if (selected[e] % count == r) {
                    for (int h = 0; h < hidden; ++h) { values[std::size_t(e) * hidden + h] = std::uint16_t((e * 13 + h) % 65535); }
                }
                rank.output.copy_from_host(values.data(), values.size() * 2);
                root.bind_to_current_thread();
                Tensor peer(rank.output.p, DType::BF16, {hidden, entries});
                gather_peer_experts(peer, id_view, output, r, count, root.stream);
                root.synchronize();
            }
            std::vector<std::uint16_t> got(std::size_t(hidden) * entries);
            combined.copy_to_host(got.data(), got.size() * 2);
            for (int e = 0; e < entries; ++e) for (int h = 0; h < hidden; ++h) {
                require(got[std::size_t(e) * hidden + h] == std::uint16_t((e * 13 + h) % 65535), "expert output slot mismatch");
            }
            root.bind_to_current_thread(); combined.fill(0xFF);
            for (int r = 0; r < count; ++r) {
                auto& rank = *ranks[r]; rank.device.bind_to_current_thread();
                Tensor peer(rank.output.p, DType::BF16, {hidden, entries});
                Tensor local_ids(rank.ids.p, DType::I32, {top_k, columns});
                gather_peer_experts(peer, local_ids, output, r, count, rank.device.stream);
                CUDA_CHECK(cudaEventRecord(rank.done, rank.device.stream));
            }
            root.bind_to_current_thread();
            for (int r = 1; r < count; ++r) { CUDA_CHECK(cudaStreamWaitEvent(root.stream, ranks[r]->done, 0)); }
            root.synchronize(); combined.copy_to_host(got.data(), got.size() * 2);
            for (int e = 0; e < entries; ++e) for (int h = 0; h < hidden; ++h) {
                require(got[std::size_t(e) * hidden + h] == std::uint16_t((e * 13 + h) % 65535), "owner-pushed expert output mismatch");
            }
        }
    }
}
}

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return 77; }
        count = std::min(count, 4);
        std::vector<std::unique_ptr<Rank>> ranks;
        for (int r = 0; r < count; ++r) {
            auto rank = std::make_unique<Rank>(r);
            rank->output = DeviceBuffer(4);
            ranks.push_back(std::move(rank));
        }
        for (int r = 1; r < count; ++r) for (int reverse = 0; reverse < 2; ++reverse) {
            int from = reverse ? 0 : r, to = reverse ? r : 0, supported = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&supported, from, to));
            require(supported, "required P2P link is unavailable");
            CUDA_CHECK(cudaSetDevice(from));
            const auto status = cudaDeviceEnablePeerAccess(to, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); }
            else { CUDA_CHECK(status); }
        }
        for (int from = 0; from < count; ++from) for (int to = 0; to < count; ++to) {
            if (from == to) { continue; }
            CUDA_CHECK(cudaSetDevice(from));
            const auto status = cudaDeviceEnablePeerAccess(to, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); } else { CUDA_CHECK(status); }
        }
        test_rows(ranks);
        test_rows(ranks, 2048);
        test_allgather(ranks);
        test_allgather(ranks, 2048);
        if (count == 4) {
            for (int first : {0, 2}) {
                test_allgather(ranks, 3, 2, first);
                test_allgather(ranks, 2048, 2, first);
            }
        }
        test_bf16_rows(ranks);
        test_broadcast(ranks);
        for (std::size_t owner = 0; owner < ranks.size(); ++owner) { test_peer_arena(ranks, owner); }
        test_experts(ranks);
        std::printf("PASS: %d GPUs, exact row gather, eager/captured replay, expert dispatch and output gather\n", count);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
