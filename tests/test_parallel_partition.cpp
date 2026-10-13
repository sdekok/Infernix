#include "core/parallel_partition.h"
#include "models/qwen4_exp/head_partition.h"
#include "product/parallel_options.h"

#include <climits>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace infernix;

void check(bool result, const char* label) {
    if (!result) { throw std::runtime_error(label); }
}
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input accepted");
}
int main() {
    try {
        namespace q = infernix::models::qwen4_exp;
        for (bool pipeline : {false, true}) for (int ranks : {2, 3, 4}) {
            if (pipeline && ranks != 4) { continue; }
            for (int layer = 0; layer < 36; ++layer) {
            int keys = 0, values = 0;
            for (int rank = 0; rank < ranks; ++rank) {
                if (!q::pipeline_owns(layer, rank, 36, ranks, pipeline)) { continue; }
                const auto heads = q::gdn_head_partition(16, 48, rank, ranks, pipeline);
                check(heads.keys.begin == keys && heads.values.begin == values, "GDN head gap or overlap");
                check(heads.values.count == 3 * heads.keys.count, "GDN group split");
                keys += heads.keys.count; values += heads.values.count;
            }
            check(keys == 16 && values == 48, "GDN coverage");
            }
            std::vector<int> owned(ranks);
            for (int layer = 0; layer < 12; ++layer) {
                const int a = q::attention_head_owner(layer, 0, 2, ranks, pipeline);
                const int b = q::attention_head_owner(layer, 1, 2, ranks, pipeline);
                check(a != b, "attention heads share an owner"); ++owned[a]; ++owned[b];
            }
            for (int n : owned) { check(n == 24 / ranks, "attention owner rotation is unbalanced"); }
        }
        for (int layer = 0; layer <= 12; ++layer) for (int head = 0; head < 2; ++head) {
            check(q::attention_head_owner(layer, head, 2, 4, false, true) == head, "fixed KV owner changed across layers");
        }
        std::vector<int> experts_owned(4);
        for (int l = 0; l < 48; ++l) for (int e = 0; e < 512; ++e) {
            const int owner = q::parallel_expert_owner(l, e, 512, 48, 4, true);
            check(owner >= 0 && owner < 4 && q::pipeline_owns(l, owner, 48, 4, true), "expert escapes its pipeline stage");
            ++experts_owned[owner];
        }
        for (int n : experts_owned) { check(n == 6144, "unbalanced pipeline expert capacity"); }
        for (int rows : {1, 16, 48, 96, 320, 324, 640, 1280, 2560, 13952, 16384, 248320, INT_MAX}) {
            for (int ranks : {1, 2, 3, 4}) {
                std::int64_t cursor = 0;
                int smallest = INT_MAX, largest = 0;
                for (int rank = 0; rank < ranks; ++rank) {
                    auto p = partition_rows(rows, rank, ranks);
                    check(p.begin == cursor && p.count >= 0, "gap or overlap between shards");
                    check(p.begin == rows || p.begin % 16 == 0, "unaligned shard");
                    cursor += p.count;
                    smallest = std::min(smallest, p.count); largest = std::max(largest, p.count);
                }
                check(cursor == rows, "shards do not cover the matrix");
                check(largest - smallest <= 31, "unbalanced row groups");
            }
        }
        for (auto text : {"", "0", "0,0", "-1,2", "0,", ",1", "0,,1", "0, 1", "0,1x", "0,1,2,3,4", "0,2147483648"}) {
            rejects([&] { product::parse_parallel_devices(text); });
        }
        check(product::parse_parallel_devices("3,1,2,0") == std::vector<int>({3,1,2,0}), "device order lost");
        const std::vector<int> devices{3,1,2,0};
        validate_parallel_devices(devices, 3);
        rejects([&] { validate_parallel_devices(devices, 0); });
        rejects([] { partition_rows(0, 0, 4); });
        rejects([] { partition_rows(16, 4, 4); });
        rejects([] { partition_rows(16, 0, 0); });
        std::cout << "parallel partition and option tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
