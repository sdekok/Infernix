// Stored BF16 expert outputs and FP32 routing weights against an independent
// FP64 weighted-sum oracle. Unowned slots are NaNs and must never be read.
#include "core/device.h"
#include "core/decode_graph.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/op_tester.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
using namespace infernix;
namespace t = infernix::test;
void require(bool pass, const char* reason) { if (!pass) { throw std::runtime_error(reason); } }
void run(DeviceContext& device, int H, int T, int R, bool skew, bool mapped = false, bool alias = false, int stage = -1) {
    constexpr int K = 10;
    const std::size_t cells = std::size_t(H) * T;
    std::vector<std::int32_t> ids(K*T), owner_map(512);
    for (int e=0;e<512;++e) { owner_map[e] = stage >= 0 ? stage * 2 + e / 256 : mapped ? (e / 7 + 1) % R : e % R; }
    std::vector<float> weights(K*T), gates(T);
    std::vector<std::uint16_t> values(cells*K), shared(cells);
    std::uint32_t random = 91723;
    const auto draw = [&]() { random = random * 1664525U + 1013904223U; return float(int(random >> 16) - 32768) / 32768; };
    for (int i = 0; i < K*T; ++i) { ids[i] = skew ? (i % 3 ? 0 : (R-1)) : (i * 7 + i / K) % 512; weights[i] = std::abs(draw()) / K; }
    for (auto& x : gates) { x = std::abs(draw()); }
    for (auto& x : values) { x = t::f32_to_bf16(draw() * 8); }
    for (auto& x : shared) { x = t::f32_to_bf16(draw()); }
    DeviceBuffer alias_output(cells*K*2), root_partial(cells*4);
    DeviceBuffer dm(owner_map.size()*4), di(ids.size()*4), dw(weights.size()*4), dg(gates.size()*4), ds(cells*2), dp(cells*R*4), dy(cells*2);
    dm.copy_from_host(owner_map.data(),dm.bytes);
    di.copy_from_host(ids.data(), di.bytes); dw.copy_from_host(weights.data(), dw.bytes);
    dg.copy_from_host(gates.data(), dg.bytes); ds.copy_from_host(shared.data(), ds.bytes);
    std::vector<DeviceBuffer> inputs;
    for (int r = 0; r < R; ++r) {
        auto local = values;
        for (int entry = 0; entry < K*T; ++entry) if (owner_map[ids[entry]] != r) {
            std::fill_n(local.begin() + std::size_t(entry)*H, H, std::uint16_t{0x7fc1});
        }
        inputs.emplace_back(local.size()*2); inputs.back().copy_from_host(local.data(), local.size()*2);
    }
    ops::MoeRouting routing{Tensor(di.p,DType::I32,{K,T}), Tensor(dw.p,DType::FP32,{K,T}), Tensor(dg.p,DType::FP32,{T})};
    Tensor partials(alias ? alias_output.p : dp.p,DType::FP32,{H,T,R}), ts(ds.p,DType::BF16,{H,T}), y(dy.p,DType::BF16,{H,T});
    const auto call = [&] {
        if (alias) { CUDA_CHECK(cudaMemcpyAsync(alias_output.p, inputs[0].p, inputs[0].bytes, cudaMemcpyDeviceToDevice, device.stream)); }
        for (int r = 0; r < R; ++r) {
            Tensor output(alias && r == 0 ? alias_output.p : inputs[r].p,DType::BF16,{H,K*T});
            auto p = alias && r == 0 ? Tensor(root_partial.p,DType::FP32,{H,T}) : partials.slice(2,r,1).view({H,T});
            ops::moe_combine_owner(output,routing,r,R,p,device.stream,mapped ? static_cast<const std::int32_t*>(dm.p) : nullptr);
        }
        if (alias) { CUDA_CHECK(cudaMemcpyAsync(partials.data, root_partial.p, root_partial.bytes, cudaMemcpyDeviceToDevice, device.stream)); }
        ops::moe_combine_partials(partials,routing,ts,y,device.stream);
    };
    call(); device.synchronize();
    std::vector<std::uint16_t> eager(cells), captured(cells); std::vector<float> actual(cells*R), replay(cells*R);
    dy.copy_to_host(eager.data(),dy.bytes); CUDA_CHECK(cudaMemcpy(actual.data(),partials.data,cells*R*4,cudaMemcpyDeviceToHost));
    DecodeGraphDefinition definition; DecodeGraphExecutable graph;
    definition.capture(device.stream,call); graph.instantiate(definition); graph.launch(device.stream); device.synchronize();
    dy.copy_to_host(captured.data(),dy.bytes); CUDA_CHECK(cudaMemcpy(replay.data(),partials.data,cells*R*4,cudaMemcpyDeviceToHost));
    require(eager==captured && actual==replay,"owner combine captured/eager differ");
    for (int token = 0; token < T; ++token) for (int d = 0; d < H; ++d) {
        double reference=0, magnitude=0;
        for (int r = 0; r < R; ++r) {
            double sum=0, mag=0;
            for (int k=0;k<K;++k) { const auto e=std::size_t(token)*K+k; if(owner_map[ids[e]]!=r)continue;
                const double term=double(weights[e])*t::bf16_to_f32(values[e*H+d]); sum+=term; mag+=std::abs(term); }
            const float got=actual[std::size_t(r)*cells+std::size_t(token)*H+d];
            require(std::isfinite(got) && std::abs(double(got)-sum)<=std::ldexp(double(2*K+2),-24)*mag+1e-35,"owner partial exceeds FP64 accumulation bound or reads poison");
            reference+=sum; magnitude+=mag;
        }
        const auto at=std::size_t(token)*H+d;
        const double term=double(gates[token])*t::bf16_to_f32(shared[at]); reference+=term; magnitude+=std::abs(term);
        const double got=t::bf16_to_f32(captured[at]);
        const double arithmetic=std::ldexp(double(2*K+R+4),-24)*magnitude;
        require(std::isfinite(got) && std::abs(got-reference)<=std::abs(reference)/256+arithmetic*(1+1./256)+std::ldexp(1.,-133),"combined output exceeds FP64/BF16 bound");
    }
    std::printf("PASS GPU%d H=%d T=%d R=%d skew=%d alias=%d: FP64 partials/full sum, poisoned unowned slots, captured replay\n",device.device,H,T,R,int(skew),int(alias));
}
int main() { try {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1)return 77;
    for(int device=0;device<std::min(count,4);++device) { DeviceContext gpu(device);
        for(int ranks:{2,3,4})for(bool skew:{false,true})run(gpu,257,3,ranks,skew);
        run(gpu,257,31,4,true,true);
        run(gpu,257,31,4,true,true,true);
        for (int stage : {0,1}) { run(gpu,2560,31,4,false,true,true,stage); }
        if(device==0) {
            for(int columns:{1,8,31,256,2048})for(bool skew:{false,true})run(gpu,2560,columns,4,skew);
            for(int columns:{9,512,2048})for(bool skew:{false,true})run(gpu,2560,columns,4,skew,false,true);
        }
    }
    return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;} }
