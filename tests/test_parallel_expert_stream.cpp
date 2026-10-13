// Exact streamed records and exclusion of resident, remote-owned and CPU jobs.
#include "core/device.h"
#include "models/qwen4_exp/execution/expert_stream.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace infernix;
namespace q = infernix::models::qwen4_exp::execution;
void require(bool good, const char* message) { if (!good) { throw std::runtime_error(message); } }
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { return 77; }
        for (int gpu = 0; gpu < std::min(count, 4); ++gpu) {
            DeviceContext device(gpu);
            constexpr unsigned L = 3, E = 16, stride = 256;
            PinnedHostBuffer bank(L * E * stride);
            auto* bytes = static_cast<std::uint8_t*>(bank.data());
            for (unsigned l = 0; l < L; ++l) for (unsigned e = 0; e < E; ++e) {
                std::memset(bytes + (l * E + e) * stride, l * E + e, stride);
            }
            void* mapped = nullptr; CUDA_CHECK(cudaHostGetDevicePointer(&mapped, bank.data(), 0));
            for (unsigned owners : {1U, 2U, 3U, 4U}) for (bool gated : {false, true}) {
                const unsigned owner = gpu % owners;
                q::ExpertStream stream(L, E, stride, [=](unsigned l, unsigned e) -> const std::uint8_t* {
                    return e % owners == owner ? static_cast<const std::uint8_t*>(mapped) + (l * E + e) * stride : nullptr;
                }, true);
                DeviceBuffer ring(16 * 1024);
                std::vector<std::int32_t> frames(L * E, -1), columns(E, 2);
                std::vector<std::uint8_t> cpu(E, 0);
                for (unsigned l = 0; l < L; ++l) { frames[l * E + owner] = 0; }
                if (gated) { cpu[owners + owner] = 1; }
                stream.begin({ring.p, ring.bytes}, frames.data(), device.stream, gated);
                unsigned expected_count = 0;
                for (unsigned l = 0; l < L; ++l) {
                    if (gated) { stream.plan_layer(l, columns, cpu); }
                    // A layer walk reuses the same prefetched records for every
                    // chunk, releasing the half only after the final consumer.
                    for (int chunk = 0; chunk < 3; ++chunk) {
                        stream.before_experts(l, device.stream); device.synchronize();
                        std::vector<std::int32_t> slots(E);
                        CUDA_CHECK(cudaMemcpy(slots.data(), stream.slots(l), E * 4, cudaMemcpyDeviceToHost));
                        for (unsigned e = 0; e < E; ++e) {
                            const bool expected = e % owners == owner && e != owner && (!gated || cpu[e] == 0);
                            require((slots[e] >= 0) == expected, "streamed unowned/resident/CPU expert");
                            if (!expected) { continue; }
                            if (chunk == 0) { ++expected_count; }
                            std::vector<std::uint8_t> got(stride);
                            CUDA_CHECK(cudaMemcpy(got.data(), stream.base() + slots[e] * stride, stride, cudaMemcpyDeviceToHost));
                            for (auto value : got) { require(value == l * E + e, "stream record bytes changed"); }
                        }
                    }
                    stream.after_experts(l, device.stream);
                }
                device.synchronize(); stream.end();
                require(stream.stats().streamed == expected_count, "stream byte accounting mismatch");
            }
        }
        std::puts("PASS owned expert prefetch records and gated exclusions"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
