# Experimental Flash-Next TP/EP build and tests

This checkout adds `--parallel-devices 0,1,2,3` to the CLI, server and inference
benchmark. Read the [implementation and validation record](../../docs/maintainer/multi-gpu-5060-ti-design.md)
and [model requirements](../../docs/qwen3_8-flash-next.md#experimental-multi-gpu)
before using a real artifact. The current target is Linux with SM120 GPUs and
working P2P between every selected device pair.

## Isolated build

From the repository root on the GPU host:

```bash
sudo podman build -t localhost/infernix-tp4-build -f tools/tp4/Containerfile tools/tp4
sudo podman run --rm --network none --security-opt label=disable \
  -v "$PWD:/work" localhost/infernix-tp4-build -lc '
    cmake -S /work -B /work/build-tp4 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DINFERNIX_BUILD_BENCHMARKS=ON
    cmake --build /work/build-tp4 -j 4 --target \
      infernix infernix-serve infernix_bench \
      infernix_parallel_partition_test infernix_parallel_transport_test \
      infernix_parallel_weights_test infernix_resident_moe_test infernix_hyper_connection_test \
      infernix_parallel_state_test infernix_parallel_qsa_test infernix_parallel_gdn_test \
      infernix_gdn_replay_fold_test infernix_gated_delta_net_replay_record_test \
      infernix_qwen4_exp_prefix_state_test infernix_kv_cache_append_test \
      infernix_kv_cache_test infernix_host_context_arena_test \
      infernix_gdn_gating_test infernix_causal_conv1d_silu_test \
      infernix_offloaded_moe_layer_test infernix_offloaded_moe_wide_test \
      infernix_parallel_expert_stream_test infernix_qsa_select_test infernix_linear_bf16_a16_test
  '
```

The image defaults to NVIDIA's CUDA development image. An existing local CUDA
image can be selected with `--build-arg BASE_IMAGE=...`. On the target server,
`localhost/vllm031:pcie-prob-async-20261011` was used as the cached base. The test
image adds development packages and missing NVTX/profiler headers; it does not
modify the running vLLM image or host toolchain. Four build jobs avoid competing
with the live service on the six-core CPU. No GPU access is needed for building.

## GPU checks

The test runner explicitly takes the **systemd unit** it is permitted to stop.
Use a unit rather than its transient Podman container name: systemd can recreate
auto-removed containers, and can otherwise restart the service during tests.

```bash
bash tools/tp4/run-tests.sh vllm.service
```

It stops that unit, runs the tests with NVIDIA CDI device access, removes its
own test container, and restarts the unit if it was previously active. The exit
status reports test failure even when recovery succeeds. Check the service's
normal readiness endpoint after its model has reloaded. The log is
`build-tp4/tp4-tests.log`. Unrelated services are left running.

The tests cover parser/row partitions, exact cross-GPU transfers, multi-device
capture/replay, artifact row/scale ownership, FP64 projection and fused-HC
oracles, shared-expert fused projections and MTP expert partitions. They use
small synthetic artifacts and require no model download. They do not establish
whole-model performance, context capacity or speculative acceptance.

The initial 2026-10-12 state-sharding run passed all fourteen GPU-suite targets on the
target server, including selected two-/three-GPU uneven partitions, wide-prefill
gathers, physically sharded KV/GDN state, and one-KV-head attention oracles.
CLI/server options and benchmark-support tests passed separately without GPU
access. Real-artifact testing subsequently added GDN gating/convolution shard
cases, same-process multi-device expert/QSA checks and the wide-expert oracle
to the runner. The expanded run passed all nineteen targets in 222.20 s.
The final tuning build, including owned expert streaming and VMM peer-access
regressions, passed all twenty selected targets in 220.80 s.
The later MTP projection extent correction passed the expanded wide-projection
suite and real 512-token MTP prefill/generation; see the validation record for
the affected follow-up and comparative measurements.

## Real-model gate

A compatible `.infernix`/`.ninfer` artifact and its `.ngram` volume are
required. Start with one lane, a short prompt and 131,072-token INT8 capacity, then compare
single-GPU and four-GPU generation, MTP, prefix reuse and memory usage with the
same artifact and settings. A model that fits the original 32 GB target may not
fit one 16 GB card for a baseline; do not mislabel a changed configuration as an
equivalent comparison. Report PCIe/cache costs as well as any throughput gains.

The isolated target-server checkout is `/home/stephen/dev/InfernixTp4-test`.
The editable source checkout is `/home/stephen/dev/InfernixTp4/infernix` on the
local development machine. No full model was used for the synthetic checks.
The Dense8 artifact is now installed at `/home/stephen/models/infernix-dense8`
on the target host, with all three parts and the shared n-gram volume verified
against their Hugging Face SHA-256 hashes. Short generation and MTP smoke
results, failures fixed, measured costs and remaining gaps are recorded in the
[validation record](../../docs/maintainer/multi-gpu-5060-ti-design.md).

For an already downloaded artifact, the smoke runner records generation and
per-GPU/host memory in `tp4-results/`, limits each run to 30 minutes, and restores
the named service on exit. Pass the first part and the n-gram volume explicitly;
all sibling artifact parts must be in the same directory. The default run uses
four GPUs, INT8 KV, context 131072, greedy sampling and CUDA graphs:

```bash
bash tools/tp4/run-model-smoke.sh vllm.service /models/model-00001-of-00003.infernix /models/model.ngram
# Exercise the same prompt with the resident MTP expert shards:
bash tools/tp4/run-model-smoke.sh vllm.service /models/model-00001-of-00003.infernix /models/model.ngram \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Additional CLI flags follow the three required arguments. The runner mounts
the artifact directories read-only and disables expert-state persistence.
Check the service readiness endpoint after its restart; `systemctl start` alone
does not establish that its model has finished loading.

Set `TP4_PREFLIGHT_REGEX` on the smoke runner to run selected CTest checks in
the same service interruption before loading the model. For example,
`TP4_PREFLIGHT_REGEX="^infernix_parallel_(qsa_devices|offloaded_moe)_test$"`.

## System performance comparisons

`run-perf.sh` uses the public benchmark with one loaded Engine, two discarded
warmup requests and three measured repetitions. Its default workload is 512
corpus prompt tokens plus 256 decode tokens. All invocations enforce INT8 KV
and 131,072-token capacity after extra options; actual occupied context is the
workload length. Physical KV grows on demand, reclaiming expert-cache frames.

```bash
bash tools/tp4/run-perf.sh vllm.service /models/model-00001-of-00003.infernix /models/model.ngram baseline
bash tools/tp4/run-perf.sh vllm.service /models/model-00001-of-00003.infernix /models/model.ngram mtp3 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Reports, command options and sampled memory/power/clocks go into the named
`tp4-results/` directory. `TP4_PREFLIGHT_REGEX` runs affected CTest checks before
loading. `TP4_PROFILE=1` captures one measured repetition using the target's
Nsight Systems 2026.1.3 installation. Node tracing failed on the current
multi-device graph; use `--no-cuda-graph` for the attribution capture. Instrumented
eager timings must not be presented as uninstrumented graph performance.

Set `TP4_WORKLOAD=8192,256` to replace the default workload; a semicolon-separated
list runs multiple workloads in the same Engine and therefore shares expert-cache
warming between them. Use separate invocations for matched standalone comparisons.
`TP4_BENCH_BINARY=/work/path/to/saved/infernix_bench` selects an explicit saved
baseline. Keep runner files unchanged while an invocation is active.

On the tested four-5060-Ti Dense8 system, the balanced long-prompt setting
uses chunk 2048. Add MTP3 and the optimized draft head for speculative decoding:

```bash
TP4_WORKLOAD=8192,256 bash tools/tp4/run-perf.sh vllm.service \
  /models/model-00001-of-00003.infernix /models/model.ngram combined \
  --prefill-chunk 2048 --spec mtp --draft-tokens 3 --lm-head-draft
```

This keeps the same 131,072-token INT8 capacity. On the final selectable-layout
build, the measured rates are:

| Workload | TP4 rotating KV | TP4 fixed KV | TP2+PP2 |
|---|---:|---:|---:|
| 8K + 256, ordinary decode | 1152 / 44.8 | 1156 / 45.2 | 1401 / 45.4 |
| 64K + 256, ordinary decode | 1216 / 35.4 | 1214 / 38.0 | 1799 / 34.9 |
| 8K + 256, MTP3 | 1140 / 110.5 | 1144 / 110.9 | 1347 / 104.5 |

Each cell is prefill / decode tokens per second. Add `--fixed-kv-owners` for the
fixed-KV TP4 layout: its 64K ordinary decode improves 7.07%, but prefill is 0.10%
lower and KV pressure concentrates on the first two GPUs. Its MTP gains are small.
The default keeps rotating KV owners.

Add `--pipeline-parallel` to select TP2+PP2. It improves long-prompt prefill but
uses additional primary workspace: matched 64K ordinary decode is 1.51% slower,
and matched 8K MTP decode is 5.40% slower. Tokens match in these comparisons;
8K MTP acceptance is identical at 97.93%. The pipeline's 512-token MTP case instead
changes output and acceptance: prefill is 33.8% slower, speculative decode 14.2%
faster, and the full request 0.53% slower. These rates are workload-specific.
Pipeline chunk 4096 loses 2.81% prefill and 5.00% ordinary decode versus 2048 at
64K, with a different CPU-autotune split; keep chunk 2048 for this rig.

Selective scratch loans recover about 0.5–0.6 GiB per GPU at chunk 2048 and
1.0–1.1 GiB at chunk 4096 compared with retaining all wide scratch. Expert caches
use the available memory, so this recovery need not lower `nvidia-smi` usage.
It can instead leave more room for resident experts or growing KV. Mathematical
Op checks pass; whole-model quality parity is unqualified. The validation record
retains the earlier adverse results, including the pre-campaign short-MTP decode
regression from 79.89 to 66.64 tok/s and 11.07% longer request time after changed
rounding/routing. Updated capacity and serving evidence is recorded there separately.

The concurrent expert-result-write experiment remains saved, disabled, in
[`experiments/`](experiments/README.md) for a fresh comparison after other
bottlenecks change. Its final-code 64K retest, with two alternating fresh launches
per variant, improved prefill 0.976% and request time 0.848%, with decode 0.030%
lower and identical cache/memory diagnostics. It remains a candidate because
final-code shorter prompts, MTP and chunk 4096 have not been compared; the older
negative 8K comparison used earlier code.

For a physical-capacity probe, provide a corpus containing at least 131,000 token
IDs and run `TP4_WORKLOAD=131000,32` with
`--corpus /work/path/to/corpus.ids --warmup 0 -r 1`. This occupies almost all the
131,072-token address space and exercises growth beyond startup backing. A
repeated corpus can establish allocation/execution capacity but does not qualify
long-context model quality. Small synthetic Op checks do not allocate a model KV
cache; all real-model performance invocations enforce the 128K INT8 capacity.

The final capacity probes added `--spec mtp --draft-tokens 3 --lm-head-draft` and
passed all three layouts with 131,000 prompt tokens plus 33 outputs, including
physical KV growth and shrink. Cold decode was 23.4 / 24.3 / 17.8 tok/s for rotating
TP4 / fixed-KV TP4 / pipeline; these single cold runs do not predict warmed speed.
Basic OpenAI chat serving also passed prefix reuse, two concurrent lanes, long
prefill cancellation and recovery for all three layouts, using a requested 256K
shared INT8 pool. Pipeline's cached/concurrent responses were slower in that
functional probe. See the validation record for scope, timings and limitations.
