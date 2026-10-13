#pragma once

#include "core/device.h"
#include <mutex>
#include <vector>

namespace infernix {
// Function attributes are per CUDA device context. Keep the shared-memory
// reservation monotonic across Programs/threads that use the same kernel.
template <auto Kernel>
void prepare_cuda_shared(int bytes, bool prefer_shared = false) {
    struct Attributes { int bytes = 0; bool prefer_shared = false; };
    static std::mutex mutex;
    static std::vector<Attributes> devices;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard lock(mutex);
    if (devices.size() <= static_cast<std::size_t>(device)) { devices.resize(device + 1); }
    auto& attributes = devices[device];
    if (bytes > attributes.bytes) {
        CUDA_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        attributes.bytes = bytes;
    }
    if (prefer_shared && !attributes.prefer_shared) {
        CUDA_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
        attributes.prefer_shared = true;
    }
}
} // namespace infernix
