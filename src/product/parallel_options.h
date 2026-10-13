#pragma once

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace infernix::product {
inline std::vector<int> parse_parallel_devices(std::string_view text) {
    std::vector<int> devices;
    for (;;) {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        int value = -1;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value < 0 ||
            std::find(devices.begin(), devices.end(), value) != devices.end()) {
            throw std::invalid_argument("--parallel-devices expects 2-4 distinct nonnegative CUDA ordinals separated by commas");
        }
        devices.push_back(value);
        if (devices.size() > 4) { throw std::invalid_argument("--parallel-devices supports at most four GPUs"); }
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    if (devices.size() < 2) { throw std::invalid_argument("--parallel-devices needs at least two GPUs; omit it for one GPU"); }
    return devices;
}
} // namespace infernix::product
