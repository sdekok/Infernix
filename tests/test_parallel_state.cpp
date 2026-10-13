// Physical head ownership, KV page transfers, and canonical lane snapshots.
#include "core/cuda_vmm.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/vmm_arena.h"
#include "models/qwen4_exp/parallel_kv.h"
#include "models/qwen4_exp/execution/parallel_state.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <numeric>

using namespace infernix;
namespace q = infernix::models::qwen4_exp;
namespace px = q::prefix;
namespace {
void require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
q::TextConfig config() {
    q::TextConfig c;
    c.hidden_size = 2560; c.num_hidden_layers = 48;
    for (int l = 0; l < 48; ++l) { c.layer_types.push_back(l % 4 == 3 ? q::MixerKind::Attention : q::MixerKind::Gdn); }
    c.attention_layers = 12; c.gdn_layers = 36;
    c.attention = {24, 2, 256};
    c.gdn.key_heads = 16; c.gdn.value_heads = 48;
    c.gdn.key_head_dim = c.gdn.value_head_dim = 128; c.gdn.conv_kernel = 4;
    c.hc.streams = 4; c.qsa.index_head_dim = 128; c.qsa.compress_ratio = 4;
    c.ple.layer = 0; c.ple.conv_kernel = 4; c.ple.conv_dilation = 3;
    return c;
}
void pattern(void* data, std::size_t size) {
    auto* bytes = static_cast<std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) { bytes[i] = (i * 71 + i / 251 + 17) & 255; }
}
void physical_owner(void* address, int owner) {
    CUmemGenericAllocationHandle handle{};
    vmm::check(cuMemRetainAllocationHandle(&handle, address), "retain allocation");
    CUmemAllocationProp prop{};
    const auto result = cuMemGetAllocationPropertiesFromHandle(&prop, handle);
    vmm::check(cuMemRelease(handle), "release retained allocation");
    vmm::check(result, "allocation properties");
    require(prop.location.id == owner, "KV physical owner differs from projection owner");
}
void kv_pages(DeviceContext& root, std::span<const int> devices, KvCacheStorage codec, bool pipeline = false, bool fixed = false) {
    const auto c = config(); const auto geometry = px::kv_page_geometry(c, codec, true, true);
    LayoutBuilder builder;
    const auto layout = plan_device_kv_page_pool(builder, {129, geometry.geometry});
    const auto bytes = builder.finish(256);
    const auto owners = q::parallel_kv_owners(layout, bytes, geometry.planes_per_layer, devices.size(), pipeline, fixed);
    VmmRange memory(devices[0], bytes, q::kParallelStateChunk);
    for (std::size_t i = 0; i < owners.size(); ++i) {
        require(memory.map(i, devices[owners[i]], devices), "KV physical allocation failed");
        physical_owner(static_cast<std::byte*>(memory.base()) + i * q::kParallelStateChunk, devices[owners[i]]);
    }
    std::size_t total = 0; for (int device : devices) { total += memory.mapped_bytes(device); }
    require(total == memory.mapped_bytes(), "KV physical bytes counted more than once");
    DeviceKVPagePool pool({memory.base(), bytes}, layout);
    for (std::size_t p = 0; p < layout.planes.size(); ++p) {
        const auto plane = pool.plane(p);
        require(plane.ne[2] == 129, "padding changed the logical KV capacity");
        for (int head = 0; head < plane.ne[3]; ++head) {
            require(q::parallel_kv_head(plane, head).is_contiguous(), "owned KV head is not contiguous");
        }
    }
    auto reserved = pool.reserve(3); require(reserved.has_value(), "reserve pages");
    std::vector<DeviceKVPageLease> leases; leases.reserve(3); pool.materialize(*reserved, 3, leases);
    const auto host = plan_host_kv_page_layout(geometry.geometry);
    PinnedHostBuffer input(host.page_stride), output(host.page_stride);
    pattern(input.data(), host.page_stride);
    const auto source = leases[0].handle(), destination = leases[1].handle();
    const auto* in = static_cast<const std::byte*>(input.data());
    auto* out = static_cast<std::byte*>(output.data());
    pool.copy_from_host_records(std::span(&in, 1), {}, std::span(&source, 1), host, root.stream);
    (void)pool.copy_page(source, destination, root.stream);
    pool.copy_to_host_records(std::span(&destination, 1), std::span(&out, 1), {}, host, root.stream);
    root.synchronize();
    require(std::memcmp(in, out, host.page_stride) == 0, "distributed KV page round-trip differs");
    std::printf("PASS TP%zu codec=%d: unique KV owners and exact host/device page transfer\n", devices.size(), int(codec));
}
void snapshots(DeviceContext& root, std::span<const int> devices, bool pipeline = false, bool fixed = false) {
    const auto c = config(); constexpr int lanes = 3;
    q::execution::ParallelGdnState gdn(c, devices, lanes, 5, pipeline);
    const auto geometry = px::kv_window_geometry(c, KvCacheStorage::Vq2, true, lanes, true);
    VmmRange window(devices[0], geometry.bytes(), q::kParallelStateChunk);
    for (unsigned l = 0; l < geometry.kv_layers; ++l) for (int plane = 0; plane < 5; ++plane) {
        for (int head = 0; head < 2; ++head) {
            const auto first = (geometry.plane_offset(l, plane) + head * geometry.head_stride(plane)) / q::kParallelStateChunk;
            const int owner = q::attention_head_owner(l, head, 2, devices.size(), pipeline, fixed);
            for (std::size_t i = 0; i < geometry.head_stride(plane) / q::kParallelStateChunk; ++i) {
                require(window.map(first + i, devices[owner], devices), "window allocation failed");
            }
            const auto view = px::kv_window_view(geometry, window.base(), l);
            require(view.tags.slice(2, head, 1).is_contiguous(), "owned window head is not contiguous");
        }
    }
    const auto layout = px::plan_state_image(px::state_image_spec(c, true, KvCacheStorage::Vq2));
    DeviceBuffer ple(layout.spec.ple_bytes * lanes), tails(layout.spec.qsa_tail_bytes * (c.attention_layers + 1) * lanes), saved(layout.spec.mtp_saved_bytes * lanes);
    const px::LaneStateBuffers buffers{.gdn_shards = gdn.shards, .lanes = lanes,
        .ple = static_cast<std::byte*>(ple.p), .tails = static_cast<std::byte*>(tails.p),
        .mtp_saved = static_cast<std::byte*>(saved.p), .window = static_cast<std::byte*>(window.base()), .window_geometry = geometry};
    px::LaneStateImage image(layout, buffers);
    PinnedHostBuffer input(layout.image_bytes), output(layout.image_bytes);
    pattern(input.data(), layout.image_bytes);
    const auto* in = static_cast<const std::byte*>(input.data()); auto* out = static_cast<std::byte*>(output.data());
    const px::HostImageConst source{std::span(&in, 1), layout.image_bytes};
    const px::HostImage destination{std::span(&out, 1), layout.image_bytes};
    for (unsigned group = 0; group < layout.groups(); ++group) { image.copy_group_from_host(group, source, 1, root.stream); }
    root.synchronize();
    DeviceBuffer packed(layout.image_bytes);
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    const auto capture = [&] {
        image.copy_lane_to_packed(1, static_cast<std::byte*>(packed.p), root.stream);
        for (unsigned group = 0; group < layout.groups(); ++group) {
            image.copy_group_from_packed(group, static_cast<std::byte*>(packed.p), 2, root.stream);
        }
    };
    capture(); root.synchronize();
    definition.capture(root.stream, capture); graph.instantiate(definition); graph.launch(root.stream); root.synchronize();
    image.copy_lane_to_host(2, destination, root.stream); root.synchronize();
    for (const auto part : layout.parts) {
        if (part.kind == px::StateImagePart::Kind::Header) { continue; }
        require(std::memcmp(in + part.offset, out + part.offset, part.bytes) == 0, "distributed state snapshot differs from canonical image");
    }
    for (const auto& shard : gdn.shards) {
        cudaPointerAttributes attributes{}; CUDA_CHECK(cudaPointerGetAttributes(&attributes, shard.backing.p));
        require(attributes.device == shard.device, "GDN state has the wrong physical owner");
    }
    std::printf("PASS TP%zu: GDN/window ownership, host snapshots and captured device round-trip\n", devices.size());
}
}
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return 77; }
        count = std::min(count, 4);
        std::vector<int> devices(count); std::iota(devices.begin(), devices.end(), 0);
        for (int device : devices) {
            std::printf("GPU %d VMM minimum allocation: %zu bytes\n", device, vmm::granularity(device));
        }
        for (int a : devices) for (int b : devices) if (a != b) {
            CUDA_CHECK(cudaSetDevice(a)); const auto e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); } else { CUDA_CHECK(e); }
        }
        DeviceContext root(0);
        for (int n = 2; n <= count; ++n) {
            const std::span ids(devices.data(), n);
            for (const auto codec : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                 KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                 KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
                kv_pages(root, ids, codec);
            }
            snapshots(root, ids);
            if (n == 4) {
                kv_pages(root, ids, KvCacheStorage::Int8Group64, true); snapshots(root, ids, true);
                kv_pages(root, ids, KvCacheStorage::Int8Group64, false, true); snapshots(root, ids, false, true);
            }
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
