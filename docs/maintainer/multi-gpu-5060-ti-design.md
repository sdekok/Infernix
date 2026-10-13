# Qwen4Exp tensor and expert parallelism

Status: weight and state sharding are implemented; twenty selected GPU checks,
Dense8 CLI runs with CUDA graphs and MTP3, controlled short/8K benchmarks and a
131,000-token prompt capacity probe have passed on the four
RTX 5060 Ti cards. KV/recurrent state follows projection head ownership
without persistent replicas. The earlier
prototype split selected projection computation but retained full dense weights on
GPU 0. That prototype did not meet the requirement and has been replaced by
load-time, GPU-owned shards. The checks below establish component numerical
behavior and specified model execution/capacity, not general model quality or
full serving qualification.

## Requirement and completion criteria

The target is four RTX 5060 Ti 16 GB cards on Fedora, Ryzen 7600X, 96 GB RAM,
PCIe 5 x4 per GPU, with bidirectional P2P. The user explicitly requires complete
partitioning of the weights assigned to each GPU, including dense projections,
fused shared-expert projections, hyper-connections, vocabulary/proposal heads and
MTP. A full dense copy on the primary GPU is not an acceptable implementation.

The user permits stopping the existing `vllm` Podman container only during GPU
tests and requires restarting it afterward, including after failed tests. Build
and CPU work leave it running. The unit to stop/restart is `vllm.service`, which
owns the transient Podman container.

Completion requires:

- CLI, server and benchmark routes select the same explicit GPU list.
- Encoded weight rows load directly into their owning GPU. No full primary or
  persistent host copy of the partitioned dense matrices exists.
- Every text/MTP matrix consumer uses its partitioned operands. Small vectors,
  and the original host token embedding may retain their native ownership.
- KV and GDN state must have explicit, nonduplicated ownership matching the
  projection head groups. Attention and recurrence execute on those owners.
  Prefill, decode, MTP verification/commit, reset, prefix save/restore and cache
  growth must all respect that ownership. Component qualification and full-model
  validation must be reported separately.
- GPU tests establish exact encoding/ownership/transport and numerical behavior
  against independent oracles, including graph replay and zero-work owners.
- Build and affected single-GPU regression checks pass.
- Real-artifact fitting, generation, speculation, cache reuse and throughput
  require separate qualification with an explicitly selected artifact.

The user selected
[`wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix`](https://huggingface.co/wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix)
for the first real-model test. The download is pinned to revision
`a17aa78166b6661214a85152df8fed5f27b38742`: three artifact parts totaling
74,867,265,280 bytes and the 52,429,058,048-byte shared n-gram volume. The initial
workload is a short greedy text prompt, context 4096, INT8 KV, prefill chunk 512,
one CLI lane, CUDA graphs and an initially empty expert cache; then the same
prompt with MTP3 and the optimized proposal head. These are integration checks,
not a controlled throughput comparison or a numerical oracle. Record load/run
failures as well as successful output and memory usage.

## Ownership

The selected implementation distributes immutable weights and mutable state.

`ParallelWeightStorage` belongs to immutable Model data. The loader selects
streamed artifact parents for distributed matrices and reads only each shard's
code and scale rows into that GPU's final buffers through bounded temporary
staging. Stored quantization codes/scales are not converted. Logical full-matrix
metadata has null pointers, so a primary-only consumer cannot silently read a
replica. `Parameters` carries explicit references to the partitioned weights.

The finite partition layouts are:

| Matrix | Local ownership |
| --- | --- |
| GDN input/control | Whole key-head groups and their three value heads |
| QSA query/gate/key/value | Two whole KV/query groups; owners rotate by layer |
| QSA index query/key | One primary backing, shared by both attention owners |
| GDN/QSA output, PLE, router, head | Output rows |
| Shared expert gate/up | Matching intermediate rows of both matrices |
| HC down including injection rows | Output rows, retaining FP32 stream partials |
| HC up | Matching feature interval from every residual stream |
| MTP expert banks | Complete, contiguous expert intervals |
| Main offloaded expert banks | Expert `id % device_count`; separate owner caches |

Every GPU retains full input columns for its local dense dot products. Results
are gathered in their original row positions. Row partitions need no floating
point all-reduce. Wide main-expert calls use owner-local FP32 weighted partials and a final primary
FP32 sum/BF16 cast; narrow calls retain the ordered slot combine. The primary retains routing,
sampling and scheduler. Norm/scale vectors retain their small primary backing; fused HC stages its
normalization vector once into each peer's scratch instead of repeatedly reading it over P2P. The token embedding
continues to use the original pinned-host representation. Vision uses the
existing offload path and is not a tensor-parallel vision implementation.

`ParallelExecution` belongs to Program data and owns contexts, streams, events,
workspaces, graph-stable scratch and peer expert caches. It borrows immutable
weight shards. Native CUDA event fork/join and P2P copies keep execution in one
process and allow multi-device CUDA Graphs. There is no NCCL or multiprocessing
service layer. P2P access to the primary is required; distributed VMM planes
currently grant access to every selected GPU, so the target requires all-pairs P2P.

Growing KV planes use head-major virtual ranges with 2 MiB physical chunks on
their owners. Exact recent-key windows have the same ownership. The shared pooled
index keys and page tables remain on the primary. There is one K/V payload per
logical head, including MTP; no primary shadow cache. Head-major backing pads each head to the allocation boundary while preserving
logical pool capacity. Exact-window planes also pad each owner independently;
this padding costs memory, especially for small scale/tag planes.

GDN owns one recurrence, convolution-history slice and speculative record slice
per head group on each GPU. Native prefill, decode and verification run there;
accepted speculative records fold into the corresponding local states. Prefix
images pack/unpack those disjoint pieces into the canonical host image. Graphs
use kernel copies for distributed device snapshots. KV growth releases expert
frames on affected owners, and shrinking KV returns capacity to those caches.

Attention keeps its dense Q/gate/K/V rows on the owning GPU through Q/K
RMSNorm, RoPE, append, attention and output gating. GDN projections remain
local through recurrence; each output-row owner gathers the recurrent head
outputs directly from the other owners. Attention output-projection row owners gather the gated head pieces directly
from their KV owners.
These transfers and activation workspaces are a measurable cost to investigate,
not eliminated by distributing the persistent state. Small convolution parameters
retain their original backing plus prepared local channel slices.

## Numerical behavior

Whole-row sharding keeps each dot product on one GPU. Smaller matrix dimensions
can select different existing linear kernels; bitwise dense equivalence is not
assumed. BF16 outputs and FP32 routing/head outputs retain their public types.

The shared expert partitions gate/up together to retain fused FP32 activation
arithmetic at narrow widths. Wider BF16 calls retain the original composed
BF16 boundaries. HC's narrow fused path retains FP32 normalized values,
per-stream down partials, gates and up arithmetic. Down partials gather without
arithmetic; each up owner computes the original four-stream collapse for its
features. Wide HC calls retain the composed BF16 intermediates.

Main experts retain the NVFP4 offload kernels and ownership-filtered dispatch.
MTP experts retain the resident Q4/Q8 kernels with local expert numbering and
skip entries owned by another GPU. MTP and narrow main-expert calls gather by original routing slot before the
ordered combine. Wide main-expert calls (T>8,K>=2*rank_count) reduce locally in
FP32, then combine owners in rank order and cast to BF16 once. This changes
rounding; the independent FP64/poison/alias/graph oracle qualifies its semantics.

## Current restrictions and unresolved risks

- Linux SM120, 2–4 explicit GPUs, first GPU equal to `--device`.
- BF16/Q8 text matrices (recipes A/B); row-split Q4/Q8 MTP/proposal matrices.
  Recipe C's FP8 dense matrices need their own qualified shard layouts/routes.
- Vision, when enabled, requires its existing offload path (`auto` or `on`).
  Resident vision with parallel execution is explicitly rejected during planning.
- The full main expert bank must fit in host RAM. SSD tiering is rejected before
  materialization; the existing SSD tier is not silently shared across owners.
- Per-owner state/KV and primary scratch still constrain maximum context and concurrency.
  A collection of four 16 GB devices is not a single 64 GB allocator.
- Peer expert caches start from available memory and resize for KV growth/shrink;
  adapting peers to unrelated external VRAM pressure needs further qualification.
  The primary retains the upstream elastic resource mechanisms.
- The primary's CPU-assist and prefill stream remain; peer misses use GPU
  staging. Primary prefetch excludes other owners' experts. The controlled
  measurements below cover one lane; concurrent-request PCIe contention remains
  unmeasured. CPU prefetches of
  full dense matrices cannot be retained after those matrices are sharded.
- Weight statistics include all owned shards, and request cache-use counters sum
  the disjoint owners. Runtime memory categories remain incomplete/primary-only;
  they are not evidence of total four-GPU memory consumption.
- P2P transport, additional launches, gathers and duplicated activations can
  offset compute gains. Four cards' aggregate VRAM/bandwidth alone predicts no
  throughput result on this PCIe topology.

## Verification record

Hardware inspection established four RTX 5060 Ti devices, 16 GB each, SM120,
PCIe 5 x4 links and P2P support. A GPU-free build container uses CUDA 13.0 and
GCC 13 on the Fedora host. The first image lacked NVTX and profiler development
headers; dependency repairs are isolated to the build image. No host CUDA or
vLLM installation was replaced.

On 2026-10-12, CLI, server and inference benchmark builds passed. The focused
five-test GPU suite passed on all four RTX 5060 Ti cards: partition/parser,
transport, owned weight storage/projections/HC, resident MTP experts and the
existing single-GPU HC regression suite. The weight tests check artifact code
and scale bytes, disjoint row coverage, actual CUDA allocation ownership,
absence of a full dense backing, independently decoded FP64 numerical oracles,
and eager versus multi-device CUDA Graph replay.

The final GPU run passed all five targets in 13.59 seconds, including wide HC
BF16 projection boundaries and segmented gathers, and two-/three-GPU narrow HC
and paired SwiGLU checks (uneven 208/224-row partitions and Q8 T=8/17/65 routes).
The CLI-options, serving-options and benchmark-support CPU tests also passed.
CLI, server and benchmark were rebuilt after the final loader/runtime changes.
`git diff --check` passed. The isolated checkout lacks Git metadata, so these
test binaries report `build unknown`; this does not identify a published release.

Earlier runs exposed two fixture errors (unsupported BF16 fixture generation
and a resident-expert scale offset) and a runtime peer-copy call rejected during
graph capture. The fixtures were corrected; runtime TP transfers now use P2P
copy kernels. The subsequent five-test run passed. An initial recovery command
also targeted the auto-removed Podman container; systemd recreated the service.
The runner now stops and restores `vllm.service`, and subsequent recovery was
verified. No unrelated service was stopped.

The first distributed-state run passed 10 of 11 targets, including owned-head
QSA for all seven codecs, GDN FP64 checks, speculative record/fold, prior TP
weight/transport and single-GPU prefix/HC/expert regressions. The physical-state
test stopped before transfers because the cards reject 64 KiB VMM chunks. The
layout now uses 2 MiB chunks and padded head strides. The corrected run passed
all 14 targets in 123.01 seconds. This includes two-/three-/four-GPU physical
ownership for all seven KV codecs, a 129-page logical pool with aligned backing,
byte-exact host/device page transfers and canonical GDN/window snapshots,
captured snapshot restore, zero-work attention owners, uneven GDN control rows,
one-head append codecs, owned-head QSA FP64 oracles, GDN FP64 oracles across
the recurrent/chunked boundary, speculative record/fold, and existing KV, host
context, prefix, HC and resident-expert regressions. CLI, serving and inference
benchmark targets were rebuilt successfully. All four GPUs reported a 2 MiB
minimum VMM allocation. The test runner restored `vllm.service`; its health
endpoint returned HTTP 200 after reloading. `git diff --check` and shell syntax
validation of the recovery script passed.

The VMM correction adds per-head/plane alignment padding; it is not free
aggregate VRAM. At that stage, projection activations still gathered on the primary before the
local state operations, and complete-model PCIe/cache measurements were pending.
The subsequent system-tuning section records the GDN locality changes and
whole-model evidence. Dynamic KV/expert-cache resizing still needs longer-context
qualification.

### First Dense8 artifact runs, 2026-10-12

The selected artifact and n-gram volume are installed under
`/home/stephen/models/infernix-dense8` on the target host. All four LFS SHA-256
hashes match the pinned Hugging Face metadata. Download/assembly/checking took
place while vLLM remained available.

The first CLI run loaded successfully: 63.3 GiB of routed experts fit in pinned
RAM, and distributed dense weights occupied 1.22/1.18/1.18/1.18 GiB. The root
allocated 4,930 expert frames (12.7 GiB); each peer allocated 5,091 frames
(13.12 GiB). Startup took 55.9 s on this cold run. Generation failed before the
first token because the GDN gating wrapper/kernel still assumed 48 heads.
The gating implementation now accepts the finite whole-head shard sizes, and
the fused convolution split gained the corresponding row profiles (the latter
was found while tracing the same GDN path). Existing FP64/output-history
criteria were preserved. The two affected Op suites passed in 5.91 s, including
new shard cases and original full-shape cases.

The second load took 36.9 s with the same allocation figures, but the first
offloaded expert launch failed with CUDA `invalid argument`. Its shared-memory
configuration was process-static instead of per device. Expert and QSA kernel
attributes are now configured per device, with monotonic shared-memory limits
across Program threads. New regressions exercise the same kernels on successive
GPUs within one process. Those two multi-device tests and the wide-expert oracle
passed in 90.67 s. The final expanded regression run is recorded below.

The final affected GPU suite passed **19/19 tests in 222.20 s**, after rebuilding
all selected targets against the final source. It includes the earlier fourteen
targets, GDN gating/convolution shard oracles, multi-device QSA/expert execution
and the wide-expert oracle. CLI, server and benchmark products built. Shell
syntax checks and `git diff --check` passed.

The next model attempt emitted `6 times 7 is` and stalled. An initial
`--no-cuda-graph` attempt also stalled: the existing Flash-Next adapter had not
forwarded that option. Wiring it into Program execution and graph budgeting
made a true eager run complete (`6 times 7 is 42.`). Replacing 504 programmatic
graph edges with full dependencies did **not** fix the graph stall; that
experiment was removed. Temporary host diagnostics then located the wait
inside `cudaGraphLaunch`, before the launch thread could read n-gram rows and
release the mapped gate that the GPU was waiting on. A single persistent
Program-owned host worker now reads/publishes those rows independently, for
both plain decode and MTP verification. Each round joins before its staging
buffers or captured references can change; I/O errors still release the gate,
and observer callbacks remain on the execution thread. Original CUDA graph/PDL
ordering is retained. All stalled test containers were terminated and the
runner restarted vLLM; it remained running during rebuilds.

Successful runs used greedy sampling, one CLI lane, INT8 KV, context 4096,
prefill chunk 512 and `--expert-state off` (cold expert caches). The arithmetic
prompt was `What is 6 times 7? Answer in one short sentence.` The counting prompt
was `Count from 1 to 20, separated by commas. Do not include anything else.`

| Workload | Mode | Generated tokens | Prefill | Decode | Reported decode rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Arithmetic | Eager, no MTP | 10 | 2.3 s | 636 ms | 14.2 tok/s |
| Arithmetic | CUDA graphs, no MTP | 10 | 2.3 s | 736 ms | 12.2 tok/s |
| Counting | CUDA graphs, no MTP | 70 | 2.4 s | 3.5 s | 19.6 tok/s |
| Counting | CUDA graphs, MTP3 + proposal head | 70 | 2.4 s | 1.8 s | 39.2 tok/s |

Both arithmetic runs produced the same answer. The graph run was slower in
this tiny sample, which includes graph startup. The counting MTP run produced
all integers 1 through 20, took 19 speculative rounds and accepted 50/50 drafts
(3.63 committed tokens/round). This is an easy repetitive prompt, not a claim
of general MTP acceptance or throughput. There is no single-GPU comparison.
The plain and MTP counting runs have exactly matching sequences of all 70
generated token IDs, including the stop token. The reported decode rate doubled
on this one prompt; total generation time was 6.0 s plain versus 4.2 s MTP,
excluding model loading. This is not a representative performance benchmark.

The successful logs are in the target checkout's `tp4-results/` directories:
`20261012T192841Z-272289` (eager arithmetic),
`20261012T193747Z-280075` (graph arithmetic),
`20261012T193910Z-281395` (MTP counting), and
`20261012T194308Z-284347` (final 19-test preflight and plain counting).
After the final run, `vllm.service` was active and its port 8030 health endpoint
returned HTTP 200. No test container was left running.

Two-second samples observed plain-mode GPU peaks of
15,433/15,391/15,391/15,391 MiB, with 16.62 GiB minimum available host RAM.
MTP peaks were 15,447/15,439/15,413/15,413 MiB, with 16.10 GiB available host RAM.
MTP dense shards total 6.27 GiB (1.60/1.56/1.55/1.55 GiB); it leaves 11.9 GiB
for the root expert cache and 12.75 GiB on each peer. Sampled peaks may miss
shorter transients. The CLI's generic weight/sequence/workspace and graph-memory
fields still report zeros for this architecture; use loader/ledger diagnostics
and external per-device measurements, not those fields, for this result.

At this initial smoke-test stage, prefix reuse through the real serving route,
long-context growth, maximum context/concurrency, rejection-heavy MTP and
representative whole-system performance remained unverified. Later capacity
and controlled performance results are recorded below. Short plausible outputs supplement the Op
oracles; they do not replace numerical qualification.

## State-distribution decision

The user selected two local KV/attention owners per layer, rotating GPU pairs
across layers. This is the implemented layout. Keep four-query-owner attention with
P2P reads as a follow-up experiment, not a second implementation in this change.
Compare both at the same artifact, KV format, context, concurrency and expert
budget, then compare their independently tuned cache budgets. Record per-GPU KV
and expert-cache bytes, expert hit/miss rates, PCIe traffic, prefill throughput
and decode latency/throughput. Equal total nonduplicated KV bytes do not by
themselves imply extra expert capacity: the placement and ownership of useful
expert frames also matter. No advantage is established for either placement.

The model has 24 query heads and two KV heads per attention layer, including
MTP. Ordinary head partitioning cannot assign a distinct whole KV head to each
of four GPUs. vLLM's `QKVParallelLinear` explicitly replicates KV heads when TP
size exceeds the KV-head count; that route does not satisfy the user's explicit
no-duplication requirement. llama.cpp documents a tensor split for weights and
KV, without establishing the absence of replication for this particular geometry.

Sources inspected 2026-10-12:
[vLLM QKVParallelLinear](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/linear/#vllm.model_executor.layers.linear.QKVParallelLinear),
[llama.cpp multi-GPU modes](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).

The selected layout and the deferred experiment are:

- Two local KV/attention owners per layer, rotating pairs across layers to
  balance aggregate memory. Each owner executes its twelve query heads against
  its single local KV head. GDN and expert work can still use four GPUs.
- Four query owners per layer, with two unique KV owners. Nonowners access
  the corresponding KV through P2P. Include a fixed KV-owner pair as a
  placement variant, so the other cards can devote more VRAM to experts. This avoids persistent KV replicas but
  introduces remote cache reads on PCIe and needs explicit head addressing.

The implementation replaces arbitrary fused-QKV row cuts with semantic
query/gate/key/value head partitions. Both attention owners use the same shared
indexer inputs and independently perform the same selection. Splitting individual heads' feature dimensions is a
different design requiring cross-GPU normalization and attention-score sums;
neither whole-head option claims that implementation.

Existing state ownership is embedded in Program allocation, the KV plane pool,
GDN state and replay-record layouts, lane images, prefix transfers and KV growth.
Qualification must include pointer ownership and exact byte coverage, numerical
attention/recurrent oracles, graph replay, and prefix/MTP state transitions.
The earlier five-test GPU result does not qualify distributed state; the new
state, owned-head attention, recurrence and replay checks are recorded separately.

## System tuning, 128K KV capacity

The next user-authorized phase profiles and tunes this four-card system. Every
real-model performance comparison reserves at least 131,072 tokens of INT8 or
FP8 KV capacity, including warmups and profiler captures. Reserved capacity is
reported separately from occupied context. The initial comparable workload is
512 corpus prompt tokens plus 256 decode tokens, two discarded warmup requests
and three measured repetitions in one loaded Engine, one lane, greedy sampling,
INT8 KV and CUDA graphs. A separate one-repetition Nsight capture attributes
execution; its instrumented throughput is not a performance result.

Compare the current TP4 implementation against owner-local GDN projection/state
execution first; record each repetition, prefill/decode and wall time, per-GPU
memory and expert-cache behavior. Preserve arithmetic/cast/state boundaries and
run the affected independent oracles. Consider TP2 + PP2 if the measured costs
justify changing layer placement/scheduling; it is not implemented by merely
selecting two devices. Report attempted alternatives, regressions and remaining
uncertainty as well as retained improvements. Restore vLLM after each test session.

The baseline `pp512+tg256` run (two warmups, three measured repetitions) passed
with explicit 131,072-token INT8 capacity: prefill 414.53 +/- 0.27 tok/s, decode
26.6769 +/- 0.0074 tok/s, request wall 10.8443 +/- 0.0036 s. Decode repetitions
were 9.59928, 9.59563 and 9.59409 s. Only 768 positions were occupied; the 128K
address space grows physical KV backing from expert-cache capacity. This is
not a filled-128K-context throughput measurement. Logs:
`tp4-results/20261012T204529Z-baseline128k-312889` on the target.

Nsight Systems 2026.1.3 node tracing caused a launch failure while priming the
multi-device graph; no trace was produced and vLLM was restored. An eager-path
capture completed (`20261012T204902Z-eager-profile128k-315756`). This instrumented
eager trace is attribution evidence, not a graph throughput comparison. It shows
2.63 million ordinary kernel launches in one measured request, with HC down
partials consuming 36.7% and row-copy kernels 22.6% of summed GPU kernel duration
(overlapping durations are not wall-time fractions). HC down partials averaged
8.42 microseconds on GPU 0, 90.44 on GPU 1, 90.36 on GPU 2 and 111.01 on GPU 3.
Their normalization vectors were read remotely from GPU 0 inside every CTA.
Ngram diagnostics showed no exposed gate wait in the unprofiled graph baseline.
The PCIe topology reports PHB between every NVIDIA GPU pair; all four cards
attach to CPU root ports, without a preferred directly switched pair.

Sequential graph-mode changes on the same pp512+tg256 workload:

| Variant | Prefill tok/s (mean +/- SD) | Decode tok/s (mean +/- SD) | Mean request wall |
| --- | ---: | ---: | ---: |
| Original TP4 | 414.53 +/- 0.27 | 26.677 +/- 0.007 | 10.844 s |
| Owner-local GDN projections | 451.21 +/- 0.65 | 28.614 +/- 0.006 | 10.094 s |
| Plus HC norm staging | 451.55 +/- 0.82 | 47.738 +/- 0.046 | 6.508 s |
| Plus right-sized copy grids | 451.79 +/- 0.57 | 47.988 +/- 0.047 | 6.479 s |

GDN retains each owner's Q/K/V/Z and A/B projection outputs through its state
operations, preserving BF16 boundaries. HC stages a 20 KiB normalization vector
into per-rank scratch for each fused call; it does not duplicate dense matrices.
Copy grids now use only as many CTAs as their element count needs, up to the
original 256-CTA cap. The grid gain is small (0.52% decode in these runs).
All seven affected GDN/weight/state checks passed (46.42 s), including added
whole-Q/K/V/Z shard FP64 checks at 2/3/4 ranks and capture replay. HC checks
passed (5.08 s), then exact transport/weight/state checks passed (8.56 s) after
the grid change. Per-GPU request cache diagnostics now cover every owner, and
public aggregate cache counts sum their disjoint expert uses. In the final
measured grid repetition, owner hit rates were 96.66/96.92/97.07/97.04%.
The original primary-specific detailed diagnostic remains explicitly about
that cache; CPU-assisted experts only belong to the primary path.

The direct GDN recurrent-output all-gather experiment passed its exact eager
and captured transfer checks and completed three measurements: 464.37 prefill
and 48.5055 decode tok/s, 6.3915 s mean wall. It reuses state scratch and adds no
persistent activation allocation. A runner was inadvertently updated in place
while this experiment was active; the benchmark finished and wrote its full
three-repetition report, then the shell parsed a stray trailing command and
returned 127. Service restoration worked. The replacement runner passes syntax
checks; subsequent runner edits occur only when it is inactive.

The original saved executable failed pp8192 prefill with an illegal GPU memory
access; the tuned executable reproduced it. Synchronous launches narrowed the
failure to a peer KV-append read. Long-prefill layer walks lend primary expert
VMM frames for activations and call metadata; those frames lacked peer access.
The arena now grants registered peers access to existing chunks and every new
or remapped chunk. Physical backing remains on its original GPU. A regression
checks exact peer reads before/after growth and remapping, including graph replay.
The original 8K run has no valid throughput result, so it is not a comparison
baseline. The post-fix real-model result is recorded below.

The corrected 8K baseline (all GDN/HC/transport tuning plus the VMM repair,
before owned-prefetch filtering) completed standalone pp8192+tg256, two warmups
and three measurements: 410.096 prefill tok/s, 44.335 decode tok/s and 25.763 s
request wall. Its primary prefetch stream still copied nonresident experts
assigned to peer GPUs: about 54.05 GiB per warmed request through 991 lent frames.
The stream now excludes nonowned records and sizes its loan for only the primary
owner's experts. An exact synthetic check covers record bytes and resident,
remote-owner and CPU-job exclusions, with one through four owners and both bulk
and gated streaming.

An exploratory combined pp512+tg256;pp8192+tg256 run completed with the owned
stream. Short decode was 48.496 +/- 0.066 tok/s (6.393 s request wall), versus
26.677 +/- 0.007 before this tuning phase. The following 8K workload reached
495.772 prefill and 44.391 decode tok/s and streamed about 3.68 GiB per request.
That 8K result has five extra preceding short requests warming its expert cache;
it is not the matched comparison against the standalone 8K baseline. A separate
standalone run uses the original two-warmup/three-measurement protocol.
Two-second GPU peaks in the combined run were 15421/15423/15423/15423 MiB;
minimum available host RAM was 15.76 GiB. These are sampled whole-device totals,
not exact allocation categories. All runs use 131072-token INT8 capacity.

The direct GDN exchange retains TP4, pulls recurrent pieces from their actual
owners into each output-projection owner, and gathers only final projected rows
on the primary. It does not remove every primary activation dependency. TP2+PP2
has not been implemented or measured; the measured avoidable costs supported
these local changes first. No claim that TP4 outperforms a tuned pipeline layout
is established. The all-PHB topology does not identify a preferred switched pair.

The matched standalone owned-stream run (`20261012T211804Z-ownedstream8k128k-342423`)
completed pp8192+tg256 with two warmups and three measurements: 431.789 +/-
0.617 prefill tok/s, 44.328 +/- 0.051 decode tok/s and 24.760 s request wall.
Against the post-VMM-fix baseline, prefill rose 5.29% and request wall fell 3.89%;
decode was 0.015% lower, within the observed repetition dispersion but with no
measured decode gain. Request walls were 24.7998/24.7449/24.7350 s. Final warmed
primary streaming was 6.59 GiB through 223 lent frames, versus 54.05 GiB through
991 frames before filtering. Final owner hit rates were 97.38/97.06/97.08/97.05%.
The mixed-workload 495.77 tok/s result must not substitute for this matched result.
One gated read wait reached 60 microseconds; storage waits were not universally zero.

The final source uses a dedicated GDN output-projection completion event,
separate from recurrent-piece readiness and expert completion. CLI, server,
benchmark and all affected GPU executables built with CUDA 13.0/GCC 13.3;
all twenty selected GPU checks passed in 220.80 s. This includes the original
single-device Op regressions, distributed FP64 checks, exact all-gather and
VMM remapping, expert-stream exclusions, KV append/cache, GDN speculative
records/fold and canonical prefix-state snapshots. Full real serving/prefix
reuse remains separate from these component checks.

The capacity probe uses a 131,000-token prompt and 32 decode steps (33 output
tokens including the prefill result), zero warmups and one measurement. Its
131,072-token corpus is two concatenated copies of the built-in 65,536-token
fixture. This is an allocation/execution probe, not a retrieval or long-context
quality evaluation; its cold-cache timing is not compared with warmed short
requests. Admission grew the pool to all 2,048 pages with 1,730 MiB physical KV
backing across owners, reducing the primary cache from 4,893 to 4,732 frames.

A further system knob is prefill chunk width. Compare 1024 columns against the
final 512-column standalone pp8192+tg256 result above, with the same two warmups,
three repetitions, 131072 INT8 capacity and no speculation. Larger chunks may
amortize fixed transfer/launch costs but consume additional workspace and reduce
expert residency. Retain a changed default only if whole-request results support
it; report any decode/cache or memory cost. If 1024 materially improves prefill,
a 2048-column check can distinguish whether that benefit continues.

Speculative tuning compares MTP draft limits 1 and 3 with the optimized proposal
head on standalone pp512+tg256, two warmups and three measured repetitions,
chunk 512 and 131072 INT8 capacity. Compare committed output tokens/s and whole
request wall against tuned plain decode; report proposal acceptance and the
extra dense/workspace memory that reduces expert residency. Acceptance on this
single corpus does not establish a general best draft length or model quality.

The near-full capacity run passed (`20261012T212123Z-capacity128k-345536`):
131,000 prompt tokens and all 33 requested output tokens, including 32 decode
steps. Prefill took 512.550 s (255.585 tok/s), decode 2.050 s (15.607 tok/s),
request wall 514.825 s. This one cold request is materially slower than the
short warmed measurements and is not a context-only scaling comparison. Owner
cache hit counters over the request were 14.25/13.08/12.30/12.43%, with 4732/
4925/4925/4925 expert frames after growth. Primary streaming moved 30.96 GiB;
ngram reads took 287 ms with no exposed GPU gate waits. The KV pool returned
to 512 backed pages after release. Full real-model KV growth, long prefill,
near-capacity decode and shrink now have execution evidence; retrieval quality,
real serving/prefix reuse and multi-request throughput do not.
The capacity session's two-second peaks were 15403/15415/15415/15415 MiB,
with 15.58 GiB minimum available host RAM. Whole-device memory stayed near
15 GiB during the occupied-context request; KV growth reclaimed cache capacity
rather than adding its full size on top of the original allocation.

At chunk 1024 (`20261012T213437Z-chunk1024-128k-356108`), standalone 8K prefill
reached 496.264 +/- 0.367 tok/s and decode 44.427 +/- 0.056 tok/s; request wall
was 22.282 s versus 24.760 s at chunk 512. Prefill improved 14.93% and wall fell
10.01%, while decode rose 0.22%. Cache capacity fell to 4800/4963/4963/4963
frames (from 4893/5059/5059/5059); final hit rates were 96.08/95.82/95.85/95.82%.
The generated continuation/routing can differ with chunk-dependent floating-point
paths; this is a same-prompt whole-request comparison, not token-identical work.
A 2048-column experiment follows because the 1024 gain is material.

At chunk 2048 (`20261012T213749Z-chunk2048-128k-358196`), standalone 8K prefill
reached 558.581 +/- 0.338 tok/s and decode 44.440 +/- 0.065 tok/s; request wall
was 20.439 s. Relative to chunk 512, prefill rose 29.36% and wall fell 17.45%;
relative to chunk 1024, prefill rose 12.56% and wall fell 8.27%. Cache capacity
was 4615/4768/4768/4768 frames, with final hits 95.56/95.07/95.25/95.18%.
Chunk 2048 is the best tested 8K prefill setting, not an established optimum or
proof of equal performance on other prompts. The suite retains chunk 512 as its
comparison default; explicit `--prefill-chunk 2048` selects the larger setting.
The near-full occupied-context probe used chunk 512; chunk 1024/2048 tests
reserved 131072 slots but occupied only their 8K prompt plus generation.

The first 512-token MTP3 benchmark (`20261012T214049Z-tuned-mtp3-128k-360264`)
failed during its first warmup: `parallel projection exceeds its startup plan`.
Short earlier MTP prompts had not reached this bound. MTP reshapes its four
residual streams into S*T columns for the hidden projection; the TP guard used
ordinary token-column count rather than the actual buffer extent. Existing HC
input/output buffers already cover those bytes. The corrected route checks each
rank's input/output byte capacity before queuing work, without increasing VRAM
allocation or changing arithmetic. Additional BF16/Q8 FP64-oracle checks cover
2560-by-2560 projection at 2048 columns, including eager/captured replay and
sampled columns through the last boundary. Service restoration succeeded after
the failed run. The twenty-test result above predates this guard correction;
its affected follow-up checks and real MTP reruns are recorded below.

The expanded projection qualification passed in 5.18 s after the MTP extent
correction. The corrected MTP3 run (`20261012T214541Z-fixed-mtp3-128k-363161`)
then completed all warmups and measured repetitions: 465.695 +/- 0.165 prefill
and 79.886 +/- 0.023 committed decode tok/s, 4.32049 s request wall. The three
measurements proposed 552 tokens and accepted 420 (76.09%), with 288 speculative
rounds and 60 plain fallback steps. The configured draft limit is 3; the engine
adapts actual draft length, so not every round proposes three tokens. Relative
to tuned plain pp512+tg256 at 48.496 tok/s and 6.393 s wall, committed decode
rose 64.73% and request wall fell 32.41% on this corpus. MTP's dense shards are
1.60/1.56/1.55/1.55 GiB; expert capacities are 4594/4893/4908/4908 frames.
A one-draft comparison follows under the same protocol.

MTP1 (`20261012T214709Z-tuned-mtp1-128k-364229`) completed at 462.929 +/- 0.066
prefill and 70.997 +/- 0.027 decode tok/s, 4.73123 s wall. It accepted 321/366
proposals (87.70%), with 366 speculative rounds and 81 fallback steps across
three measurements. Despite higher acceptance per proposal, it was 11.13%
slower in committed decode than MTP3 and took 9.51% longer per request. Its
prefill rate was 0.59% below MTP3 in this run; these are measured differences,
not claims of universal ordering. Cache capacities differed by at most one
frame per owner. Select draft limit 3 for the final combined experiment.

The final combined check is standalone pp8192+tg256, chunk 2048 and MTP3 with
the optimized head, the same two warmups/three repetitions and 131072 INT8
capacity. Compare directly with the measured plain chunk-2048 result; do not
multiply gains from separate chunk and short-prompt speculation experiments.

The combined run (`20261012T214904Z-combined-mtp3-2048-128k-365991`) passed:
554.734 +/- 0.044 prefill and 111.083 +/- 0.072 committed decode tok/s,
17.08330 s request wall (17.08447/17.08272/17.08260 s). All 570 proposals across
198 speculative rounds were accepted, with no fallback steps. This favorable
100%-acceptance continuation must not stand in for general MTP performance;
the shorter corpus test accepted 76.09% and decoded at 79.886 tok/s.
Compared with plain chunk-2048 8K inference, combined prefill was 0.69% slower,
decode 149.96% faster, and whole-request wall 16.42% lower. Compared with tuned
plain chunk-512 inference, whole-request wall fell 31.00%. Speculation and chunk
changes are measured together here, not inferred by multiplying separate gains.

Combined expert capacities were 4303/4589/4604/4604 frames (about 11.1/11.9/
11.9/11.9 GiB of cache backing), and final request hits were 94.45/94.17/94.31/
94.28%. Two-second whole-GPU peaks were 15469/15457/15417/15417 MiB, with
15.53 GiB minimum available host RAM. The larger scratch and MTP weights cost
expert residency; total memory is not reduced by the throughput changes. Separate
MTP3 and MTP1 short-run peaks were 15481/15425/15451/15451 and 15449/15415/15439/
15439 MiB, respectively. Generic category counters remain incomplete/primary-only;
these sampled device totals and explicit loader/cache diagnostics are the resource
evidence, not the report's zero-valued weight/workspace fields.

Suggested tested settings for this artifact/system are TP4, INT8 KV, one lane,
MTP with a draft limit of 3 and the optimized proposal head; chunk 2048 is the
best tested 8K prefill setting. Short-workload MTP comparisons used chunk 512.
The full occupied-128K probe used plain decoding and chunk 512. No measurement
establishes fully occupied 128K MTP performance, arbitrary-prompt acceptance,
multi-lane throughput, or a TP2+PP2 comparison. Existing QSA activation gathering
and shared indexer traffic still pass through the primary. Further tuning can
investigate these after the measured GDN and HC fixes; no claim that all primary
communication or every performance bottleneck has been eliminated is made.

Completion: CLI/server/benchmark builds and affected GPU checks passed, shell
runners pass syntax checks, and `git diff --check` is clean. Sources and these
instructions are synchronized to `/home/stephen/dev/InfernixTp4-test`; reports
are retained in both checkouts' `tp4-results` locations (the local report root
is `/home/stephen/dev/InfernixTp4/tp4-results`). No commit or PR was created.
After the final run on 2026-10-12, `vllm.service` was active and
`http://127.0.0.1:8030/health` returned HTTP 200. Only the usual service containers
remained; no Infernix test container was left running.

## Prefill tuning follow-up (2026-10-12)

The requested deliverable is higher prefill throughput on the four 5060 Ti cards,
while retaining owned dense/KV/recurrent shards, the artifact's precision and at
least 131072 INT8 KV slots in every real-model run. The user's Strata reference
is approximately 1500–2000 prompt tok/s with IQ3_S at approximately 64K tokens;
it is a target, not a matched quality/format baseline. Inspect Strata's mechanisms
as well as Infernix's timeline. Evaluate prefill, decode, request wall time,
cache/scratch costs, exact transport and numerical correctness. Acceptance requires
measured end-to-end improvement and passing affected oracles; disclose any remaining
gap to the target and any decode/resource regression.

The matched short baseline is the standalone 8192-prompt/256-decode, chunk-2048
run above: 558.580644 prompt tok/s, 44.440101 decode tok/s and 20.438898 s wall.
Use two discarded warmups and three measured repetitions, with mean and sample
standard deviation; context reuse is disabled. Also measure 64K separately.
`tp4-results/prefill-start-binaries/infernix_bench` preserves the initial binary
on the server. Profiling uses eager execution and is attribution evidence, not
an uninstrumented throughput result.

The initial eager 8192+1 trace (`20261012T221552Z-prefill-profile128k-377754`)
has a 17.475 s GPU timeline. Each peer spent 5.76–5.79 s in raw-word copies and
4.40–4.41 s in expert staging. The primary spent 2.014 s in BF16 row copies,
1.258 s gathering expert outputs and 1.032 s in scalar BF16 GEMM. These times
can overlap; they are not additive fractions of wall time. The one-token decode
workload barely warms the expert caches (about 3% hits), unlike the matched
256-token decode benchmark. Both cold expert traffic and communication matter.

The large-copy DMA candidate passed expanded exact eager/captured multi-GPU
transport tests (3.12 s). On the matched workload it measured 595.010776 ±
0.491920 prompt tok/s, 44.423561 ± 0.067497 decode tok/s, and 19.543251 s wall:
prefill +6.52%, decode -0.037% (reported, not dismissed), wall -4.38%.

Strata source inspected at `a0116b2782a5b82622358c04a06496485d12b2d6`:
[its layer-split documentation](https://github.com/Niko1221/Strata/blob/a0116b2782a5b82622358c04a06496485d12b2d6/docs/MULTI_GPU.md)
and [prefill implementation](https://github.com/Niko1221/Strata/blob/a0116b2782a5b82622358c04a06496485d12b2d6/src/prefill/prefill.cpp).
Its layer pipeline overlaps successive prompt chunks across GPUs, reducing
activation handoffs, and its prefill streams nonresident experts ahead of the
layer router. Its documented layout also replicates dense weights per card;
any future TP2+PP2 comparison here must retain explicit layer ownership to
preserve the user’s no-duplication requirement. Infernix already has layer-major expert reuse on the primary;
this follow-up extends its own ExpertStream mechanism to peer owners. No Strata
source is copied into Infernix.

The BF16 generic path now uses predicated tensor-core tiles for aligned compact
matrices; scalar handling remains for unaligned K/storage. This is the same
represented BF16-input/FP32-accumulation/BF16-output math, but accumulation order
can change rounding and downstream routing. Expanded shard FP64 oracles,
including wide HC, MTP, control projections and K/row tails, passed. The matched
BF16 candidate (`20261012T222439Z-prefill-bf16-128k-385428`) measured
675.603996 ± 0.642384 prompt tok/s and 44.438650 ± 0.054641 decode tok/s.
Two initial template-argument build errors were corrected before that run.

Peer owners now use the existing ExpertStream, borrowing two layers of owned
records plus a table frame (257 frames, approximately 0.66 GiB per peer at TP4)
from their own cache during a streamed chunk/span. The records are reused for
all chunks visiting that layer; the final chunk releases the ring half. Return
joins pending copies, including on walk cancellation. Narrow calls retain
staging. Explicit per-peer streamed-record/byte diagnostics accompany the loan
return. This adds no permanent expert copy or duplicate KV/recurrent state.
The matched peer-stream candidate (`20261012T222841Z-prefill-peerstream-128k-388791`)
measured 746.996419 ± 0.359633 prompt tok/s, 44.452732 ± 0.049999 decode tok/s,
and 16.737724 s wall. Its route/hit counts matched the preceding BF16 run.
Exact owned-record exclusions/reuse were checked on all four devices.

Wide activation broadcasts scatter distinct pieces from the primary, then
exchange pieces among active receivers. Only nonempty weight owners participate;
a join of all readers protects input-buffer reuse. The first candidate was
stopped (container SIGKILL, status 137) after review identified the missing
sparse-owner join, and its performance was discarded. A corrected-owner
exploratory run (`20261012T223424Z-prefill-broadcast-fixed-128k-393666`) reached
928.174979 prompt tok/s and 44.459145 decode tok/s, with matching route/hit counts;
its timing precedes the final explicit reader-reuse barrier and is not the final
qualified result. Exact tests now overwrite each received buffer immediately
after saving its result, in eager and captured execution, to check that barrier.

HC now preserves each up-projection owner's normalized feature groups in its
existing scratch, applies the same BF16 up-projection boundary and local stream
collapse, then returns only H combined rows rather than S*H gates. No persistent
allocation is added. The final broadcast barrier plus local-collapse candidate
(`20261012T223934Z-prefill-localcollapse-128k-397001`) passed exact transport,
FP64 shard/local-collapse, broad BF16 linear and HC tests (four targets, 104.29 s).
It measured 1010.415735 ± 0.317938 prompt tok/s, 44.434807 ± 0.058967 decode
tok/s and 13.881469 s wall on the matched 8K workload. Decode is 0.012% below
the original 44.440101 baseline; this small adverse measurement is retained.

The next experiment returns disjoint expert-output slots from the producing
owners concurrently with the primary's expert work, joining before the original
ordered top-k combination. Exact transport tests cover both root-pull and
owner-push for balanced and skewed routing. Accept only with a matched end-to-end
gain; narrow decode keeps the existing pull path.

That owner-push experiment (`20261012T224426Z-prefill-ownerpush-128k-411703`)
passed the expanded transport test (3.34 s), but measured 995.739641 ± 0.668607
prompt tok/s, 44.455352 ± 0.060150 decode tok/s and 13.997951 s wall. Prefill
was 1.45% lower versus local collapse and request wall increased 0.84%; retain
the root gather for the active comparison. These are three repetitions within
one launch per candidate, not a characterization of between-launch variation.
At the user's request, keep concurrent writes as a future candidate after other
bottlenecks change. The reproducible patch and retest protocol live in
[`tools/tp4/experiments`](../../tools/tp4/experiments/README.md); exact transport
coverage remains in the normal suite. This is a deferred candidate, not a
conclusion that concurrent writes cannot win on this system.

For the long comparison use standalone 65536 prompt tokens plus 256 decode
tokens, the same fixed corpus, chunk 2048, two discarded warmups, three measured
repetitions, CUDA graphs and 131072 INT8 capacity. Compare the saved initial
binary and accepted optimized binary under that protocol. Chunk 4096 is a
separate tuning experiment: report workspace/cache and decode tradeoffs as well
as prefill, without treating it as a same-configuration comparison.

A further trace finding motivates shared QSA selection: at 8K, wide block
scoring totals 0.1848/0.1922/0.1937 s on GPUs 1/2/3 versus 0.00666 s on GPU 0
(384 launches each). Each attention owner reads the same primary pooled-key
plane and repeats the same head-independent selection. The candidate computes
selection once next to that backing, then transfers only block IDs/counts to
the two KV owners. Its API exposes that semantic selection for reuse; KV remains
owned, and both source and destination selections use existing transient scratch.
Calls below 256 columns retain their current path. Qualification uses the
existing exact-score/top-k and independent FP64 selection oracle, and the
selected-attention route against FP64 across storage/shape boundaries. Require
unchanged routed-expert counts on the matched 8K case and measured end-to-end
improvement before retaining this candidate.

An attempted BF16 regression fixture used a two-byte activation offset, based
on an incorrect assumption that public Linear accepted unaligned activations.
The preflight (`20261012T230134Z-prefill-selection-128k-423274`) correctly rejected
it with `linear: x/out must be non-null and 16-byte aligned`; the model benchmark
did not run. Remove that unsupported fixture and its redundant internal check;
the public alignment contract already protects the template. This was a test
mistake, not a model-path regression. That same preflight passed one-head QSA
(72.93 s), QSA across four devices (16.40 s), expert streaming (0.53 s), and the
exact-score/top-k plus FP64 selection suite (5.45 s).
The restored BF16 suite passed on rerun in 90.48 s before the model benchmark.

The initial-binary long baseline (`20261012T224733Z-baseline64k-128k-414148`)
completed all two warmups and three measured 65536+256 requests: 544.051806 ±
0.102735 prefill tok/s, 35.084603 ± 0.006038 ordinary decode tok/s and
127.854090 ± 0.024027 s total. Prefill took 120.438/120.483/120.456 s across
measurements. Physical KV grew to 1088 pages (1034 MiB across the owner pools)
and shrank after each request; final cache capacities were 4499/4676/4676/4676
frames before that shrink, with 90.63/95.80/95.68/95.31% request hit rates.
Two-second device-memory peaks were 15433/15413/15413/15413 MiB; minimum
available host RAM was 15.58 GiB. The report's final KV payload is the shrunken
post-request pool, not peak occupied-context memory.

The qualified shared-selection run
(`20261012T230657Z-prefill-selection-qualified-128k-432076`) measured
1056.053343 ± 0.150727 prefill tok/s, 44.445507 ± 0.044404 ordinary decode
tok/s and 13.529459 ± 0.007325 s wall at 8192+256, chunk 2048. All aggregate
routed-expert counts and per-request peer hit counts match local collapse;
the primary's measured hit counts match too. Retain this candidate: prefill
improved 4.52% versus local collapse and 89.06% versus the initial 558.580644
baseline. The initial baseline remains the end-to-end comparison, not the
invalid broadcast or deferred owner-push experiments.

The matched optimized 64K run (`20261012T231100Z-optimized64k-128k-446128`)
measured 1105.987814 ± 0.095749 prefill tok/s, 35.078633 ± 0.029764 ordinary
decode tok/s, 59.255626 s prefill and 66.650918 ± 0.007248 s request wall.
This is 2.03× the initial 64K prefill rate; ordinary decode is 0.017% lower,
an observed adverse difference rather than a demonstrated regression or noise
claim. Aggregate routing differs from the initial scalar-BF16 binary, consistent
with the previously disclosed accumulation-order change; the same input corpus,
artifact, requested output count, KV capacity and warmup/repetition protocol were
used. All runs completed without CUDA or allocation errors.

The separate 4096-chunk 64K experiment
(`20261012T231733Z-chunk4096-64k-128k-446100`) measured 1138.383972 ± 0.147391
prefill tok/s, 34.439849 ± 0.017053 ordinary decode tok/s and 65.106033 ±
0.003578 s request wall. Relative to optimized chunk 2048, prefill improved
2.93%, decode fell 1.82%, and request wall improved 2.32%. Larger startup
workspaces displaced approximately 1 GiB of expert-cache capacity per card:
startup frames fell from 4615/4768/4768/4768 to 4243/4381/4381/4380. The primary's
64K request hit rate fell from about 90.7% to 83.9%; peers were around 94.1–94.6%.
Device-memory peaks were 15387/15411/15411/15411 MiB; minimum available host RAM
was 15.89 GiB. The lower sampled peak does not imply smaller workspace: the
allocator trades cache residency for workspace. Keep chunk 2048 as the balanced
setting, with 4096 an explicit prefill-heavy option. Do not extend the chunk sweep
without new evidence: this small gain already has a measurable decode/cache cost.

The final 8K MTP3 regression (`20261012T232452Z-prefill-final-mtp8k-128k-454805`)
used chunk 2048, the optimized draft head, two warmups and three repetitions.
Prefill improved from 554.734307 to 1044.100620 ± 0.241908 tok/s, while decode
fell from 111.082641 to 108.664155 ± 0.010136 tok/s (-2.18%). Draft acceptance
also fell from 570/570 (100%) to 567/579 (97.93%); this is consistent with the
changed BF16 accumulation order and downstream routing, rather than proof of a
pure decode-kernel slowdown. Total request time improved from 17.083295 to
10.214839 ± 0.001044 s (-40.21%). Retain the adverse decode and acceptance
measurements; neither the old 100% case nor this 97.93% case represents general
speculative throughput or model-quality qualification.

The shorter MTP3 regression (`20261012T232637Z-prefill-final-mtp512-128k-456264`),
512+256 with chunk 512, has a material adverse result: prefill improved from
465.694758 to 546.234309 ± 0.456412 tok/s (+17.29%), but decode fell from
79.886227 to 66.638282 ± 0.068849 tok/s (-16.58%), and total request time grew
from 4.320488 to 4.798678 s (+11.07%). Acceptance fell from 420/552 (76.09%)
to 321/477 (67.30%), with fallback steps increasing from 60 to 135 across three
repetitions. This explains additional decode work; it does not isolate every
timing difference. Retain this as a short-request tradeoff of the new arithmetic
and execution paths, not as an unchanged-decode claim. No corpus-dependent
dispatch or acceptance-based selection of the old arithmetic is introduced.
Long-prompt request time is substantially improved, but this evidence does not
establish a universal latency improvement or whole-model quality parity.

The final whole-inference trace
(`20261012T232754Z-prefill-final-profile64k-128k-454777`) used 65536+256,
chunk 2048, two warmups and one measured eager request with Nsight capture.
It is attribution evidence, not a replacement for graph benchmark throughput.
The NVTX prefill span was 59.276455 s. Within that span:

- GPU 0 expert-result gathering occupied 10.000409 s across 4608 kernels.
- Root-to-peer DMA on the receiving devices accumulated 14.809/17.094/17.127 s,
  transferring 78.985/78.047/78.047 GiB to GPUs 1/2/3. Additional root-stream
  copies transferred 7.5 GiB to each peer in about 0.57–0.61 s each.
- Peer exchange transferred about 66–67 GiB per directed pair, accumulating
  about 5.1–5.2 s per pair; peer-to-root DMA added about 32.6–32.7 GiB and 2.5 s
  per peer. These are recorded transfer directions, not sums of critical-path
  wall time.
- Host-to-device prefill traffic was 7.18 GiB on the primary and 5.05–5.10 GiB
  on each peer, taking only 0.38–0.55 s of recorded DMA time per device. Expert
  streaming is no longer the dominant recorded transfer cost in this warm case.
- The largest Q8 tensor-core variant took about 1.77–1.90 s per device. Wide
  QSA selection now ran only on GPU 0 (0.636 s), avoiding repeated remote scoring.
- Unioned kernel/copy activity was 23.01/35.57/37.50/38.52 s on GPUs 0/1/2/3.
  The remainder includes dependency waits and host scheduling; it is not proof
  of a CPU compute bottleneck. Parallel kernel/copy durations cannot be added
  as independent fractions of the 59.28 s span.

This supports PCIe activation/result exchange as the next performance target,
with expert-result gathering still the largest primary kernel cost. Keep the
owner-push experiment for future comparison; this trace does not itself prove
that overlapping its writes wins. Reducing collectives through TP2+PP2 or a
layer pipeline is a plausible architecture experiment, not an implemented or
measured result. Strata's reported IQ3_S speed remains a different quantization
and an approximate user workload, not a controlled performance baseline.

This prefill pass is implemented, built for CLI/server/benchmark, and qualified
by the affected exact transfer and independent FP64/FP32 numerical checks plus
the real-model runs above. Active root gather remains; the deferred patch passes
`git apply --check`. Dense and state ownership remain sharded; shared selection
uses transient scratch, and peer stream loans return at the end of prefill.
MTP device-memory peaks were 15469/15457/15417/15417 MiB at 8K and
15481/15425/15451/15451 MiB at 512, with minimum available host RAM 15.89/15.86 GiB.
The source, tests, scripts, documentation and deferred patch are synchronized
to the isolated server checkout. No commit or PR is created. All test containers
are gone; vLLM is active and `/health` returned HTTP 200 after the last run.

Remaining limits: the 1500–2000 prefill target is not reached; short MTP requests
can regress as quantified above; whole-model quality parity, multi-lane serving,
TP2+PP2 and a new physically occupied 128K run of the final prefill/MTP path are
unqualified. All model runs in this pass nevertheless had explicit 131072-token
INT8 capacity. The earlier near-128K occupancy result belongs to the earlier
plain-decoding implementation and does not close those final-path gaps.

### Concurrent expert writes: final-code 64K comparison

The user explicitly asked whether concurrent writes were retested at 64K after
all prefill changes. They were not: the earlier comparison was at 8K before
shared QSA selection. Run a new controlled comparison of saved current root
gather and that same source plus the preserved owner-push patch. Use alternating
fresh launches A/B/A/B, with 65536 prompt + 256 decode tokens, chunk 2048,
131072 INT8 KV capacity, CUDA graphs, two discarded warmups and three measured
repetitions per launch. Do not build during timed runs. Compare prefill, ordinary
decode, total request time, route counts and cache/memory costs, retaining the
individual launch means and spread rather than treating repeated requests as
independent launches. Require exact transport qualification and a repeatable
end-to-end gain before considering promotion; otherwise preserve it disabled.
This comparison does not retest MTP or the distinct chunk-4096 configuration.

The final-code comparison completed successfully on October 12 local time
(last launch October 13 UTC). The preserved patch applied cleanly and both
variants built. The exact transport suite, including concurrent disjoint output
writes, passed in 3.32 s. The default source and benchmark were restored/rebuilt
before timing; isolated copies of both binaries are retained on the server under
`tp4-results/concurrent64k-final-binaries/{root-gather,owner-push}`. No product
flag or default behavior was changed by this comparison.

| Launch, in order | Prefill tok/s mean ± within-launch SD | Decode tok/s | Request seconds |
|---|---:|---:|---:|
| Gather 1 | 1105.856207 ± 0.303449 | 35.060999 | 66.661844 |
| Push 1 | 1116.642784 ± 0.457396 | 35.035436 | 66.095015 |
| Gather 2 | 1105.816836 ± 0.237675 | 35.050382 | 66.666593 |
| Push 2 | 1116.619919 ± 0.125019 | 35.054651 | 66.102222 |

Equal-weight launch means (each launch has three timed requests) are 1105.836522
versus 1116.631352 prefill tok/s: **owner-push +0.9762%**. Total request time
fell from 66.664218 to 66.098619 s, **-0.8484% / 0.565599 s saved**. Ordinary
decode changed from 35.055691 to 35.045044 tok/s, **-0.0304%**; retain that small
adverse result without claiming a proven slowdown or dismissing it as noise.
The two gather launch means differ by 0.039371 tok/s, and the two push means
by 0.022865 tok/s. Individual gather repetitions span 1105.508–1106.064 tok/s;
push repetitions span 1116.115–1116.911. This is a repeatable small gain in the
observed four-launch sample, not evidence of no difference. Two launches per
variant still do not characterize all system/workload variability.

All per-GPU routing/cache diagnostic lines, including warmups, match exactly
across all four launches. Startup cache capacities are 4615/4768/4768/4768 frames;
sampled memory peaks are identical at 15433/15413/15413/15413 MiB. Minimum
available host RAM spans 15.647–15.664 GiB. The calibrated CPU rates vary, but
all launches select the same one-in-five decode-miss link policy, and prefill
uses no CPU experts. Exact transport semantics pass; aggregate routing equality
is not a substitute for a full generated-token comparison or model-quality test.

Evidence directories, under `tp4-results/` on both hosts:
`20261012T234339Z-concurrent64k-gather1-128k-465345`,
`20261012T235007Z-concurrent64k-push1-128k-469589`,
`20261012T235631Z-concurrent64k-gather2-128k-473194`, and
`20261013T000258Z-concurrent64k-push2-128k-476689`.

Disposition: retain the candidate with this positive final-code 64K result.
The earlier negative 8K result used earlier code; neither result establishes
a universal preference. Do not globally enable the T>=256 patch based solely
on this 64K comparison: final-code short/8K prompts, MTP and chunk 4096 have not
been compared, and no new trace isolates the overlap mechanism. The user's
specific 64K retest is complete; those broader experiments are separate work.
After this comparison, all benchmark containers were removed, the default source
still passed the candidate patch's apply check, and vLLM was active with HTTP 200
from `/health` after its model reload. No commit or PR was created.

## Systematic optimization campaign (authorized 2026-10-12)

All seven proposed experiments were implemented and evaluated sequentially. This
is the completed experiment record; the table gives the current disposition.
Earlier "pending" statements in the chronology below describe intermediate
states and are superseded by the integrated qualification section.

| Order | Experiment | Disposition |
|---|---|---|
| 1 | Owner-local attention | Retained: +5.17% isolated 64K prefill |
| 2 | GDN input-column weights and FP32 partial reduction | Disabled: 64K prefill -3.57%, request +3.02%; patch preserved |
| 3 | Reclaim parallel prefill scratch | Selective loans retained for memory; full-loan variant slower; near-128K MTP passes |
| 4 | Calibrated MTP draft costs | Disabled: mixed single-lane results, two-lane decode -7.98%; patch preserved |
| 5 | TP2+PP2 with overlapping prompt chunks | Optional: 1799 tok/s at 64K, with decode/short/cached-request costs; serving passes |
| 6 | Owner-local weighted expert partials | Wide-only route retained: +4.51% isolated 64K prefill; all-width variant slower at decode |
| 7 | Fixed KV owners and expert-placement rebalance | Fixed/even option retained: +7.07% ordinary 64K decode; rebalance deferred; serving passes |

The target is four 5060 Ti 16 GB cards, PCIe 5 x4/P2P, Ryzen 7600X, 96 GB RAM,
Fedora. Every model run reserves at least 131072 INT8/FP8 KV tokens. Retained
layouts keep unique dense shards, KV and recurrent state and preserve expert
offload/cache behavior. Only the authorized vLLM service is stopped for GPU tests
and restored afterward; builds do not overlap timed measurements. No commit or
PR is requested.

The starting baseline was root gather at approximately 1105.84 prefill tok/s
and 35.0557 ordinary decode tok/s on 65536+256/chunk2048. The separately qualified
raw concurrent-write candidate (+0.976% prefill at that workload) remains saved,
not silently folded into the weighted-combine path. Standard timing uses the
same artifact/corpus, one lane, graphs, two discarded warmups and three measured
repetitions. Short/8K/MTP cases expose regressions. Fresh launches check small
gains; fixed-KV's 64K benefit repeats with matched CPU-autotune splits. Cold
capacity and functional serving probes are explicitly separate from these timings.

Acceptance required direct independent FP32/FP64 or exact oracles for changed
mathematics, storage and state semantics before model benchmarking. Comparative
records include throughput, total latency, acceptance/output, cache residency,
per-GPU memory and host headroom, including adverse results and CPU-autotune
confounds. Whole-inference traces resolved the specific overlap/transfer decisions;
no unmeasured final bottleneck is assigned. Deferred patches retain their measured
bases. Product/architecture/test documentation and reproducible flags cover all
retained options, with final physical-context and serving checks below.

### Campaign stage 1: owner-local attention (screening)

The candidate keeps semantic Q/gate/K/V projection rows on their load-time KV
owners, through per-head normalization, RoPE, append, QSA and sigmoid gating.
The primary retains the index weights and pooled-key selection. Output-row
owners gather gated head outputs directly from the two owners. Separate producer
and consumer events prevent scratch reuse before all output projections finish.
MTP catch-up preserves key-record and tail-update semantics. Weight/state
ownership and permanent buffer capacities are unchanged.

Expanded existing FP64 RoPE and RMSNorm qualification to 12 query heads/one KV
head at decode, verification and 2048-column prefill extents. Existing exact
transport and four-device QSA checks passed. Added two-owner, rotating-pair
all-gather coverage and both text-position layouts; their final rerun is pending.
The benchmark now records generated token IDs for exact workload comparisons.

Fresh 8K+256/chunk2048 screening (two warmups, three measured repetitions):
root gather 1054.060893 prefill / 44.405915 decode tok/s / 13.549880 s total;
owner-local 1101.305027 / 44.799939 / 13.165584 s. Prefill +4.482%, total -2.836%.
All 257 generated tokens match in every measured repetition. CPU calibration
selected decode offload divisors 5 and 4 respectively, so the +0.887% decode
result is confounded; wide prefill used no CPU experts. Routing/cache aggregate
counts have small differences despite matching emitted tokens; exact generated
agreement is not a claim of bit-identical intermediates or universal quality.

Short 512+256 MTP3/chunk512 screening: baseline 546.378854 prefill /
66.691773 decode / 4.795136 s; candidate 549.994331 / 67.911937 / 4.720312 s.
All measured generated tokens match; both have 321/477 accepted drafts and 135
fallback steps. The small timing gain needs restart replication before a
performance claim. No regression has appeared in this screening; 64K and final
capacity qualification remain pending.
Evidence: `20261013T004200Z-stage1-owner-local-8k-493263`,
`20261013T004420Z-stage1-owner-local-mtp512-495588`,
`20261013T004527Z-stage1-baseline-tokens-8k-496365`,
`20261013T004749Z-stage1-baseline-tokens-mtp512-498449`.

Stage 1 disposition: retained after the expanded exact two-owner transport and
FP64 RoPE checks passed and the matched 64K+256/chunk2048 comparison completed.
Baseline `20261013T005103Z-stage1-baseline-tokens-64k-502160`: prefill
1105.068106 ±0.179653, decode 35.090517 ±0.015363 tok/s, total 66.707232 s.
Candidate `20261013T005731Z-stage1-owner-local-64k-505592`: prefill
1162.213635 ±0.127814, decode 35.292916 ±0.020641 tok/s, total 63.739486 s.
Prefill +5.1712%, total -4.4489% (2.967746 s saved); the small +0.5768%
decode result remains a single-launch observation, not a separately established
decode gain. Both use CPU miss divisor 5, two warmups and three measurements.
All 257 generated tokens match per measured request. GPU memory peaks match at
15433/15413/15413/15413 MiB; minimum host headroom 15.6463 versus 15.6259 GiB.
Final campaign capacity/multilane checks remain pending.

### Campaign stage 2: GDN input-column shards

The initial implementation loads each GDN output matrix by the matching local
value-head input columns, retaining every encoded code/scale exactly once. Local
projections produce FP32 partials; the root sums those in rank order and rounds
once to BF16. This removes the GDN feature all-gather and changes summation order.
An independent FP64 oracle decodes stored BF16/Q8/Q4 weights, checks local FP32
partials and final BF16 output for TP2/3/4 at widths 1/4/257/2048; all 36 cases
passed, including exact eager/graph agreement.

Initial screening failed the overall performance criterion. At 8192+256,
chunk2048, candidate prefill was 1005.746381 versus stage1 1101.305027 tok/s
(-8.677%); decode 46.015380 versus 44.799939 (+2.713%); request time 13.721087
versus 13.165584 seconds (+4.219%). Both CPU divisors were 4; emitted tokens
matched all three repetitions. At 512+256 MTP3, prefill 554.256196 versus
549.994331 (+0.775%), decode 67.248493 versus 67.911937 (-0.977%), total
4.750603 versus 4.720312 seconds (+0.642%). Acceptance fell from 321/477 to
312/510 and outputs differed; CPU divisors also differed (5 versus 4). Numerical
oracle success does not establish full-model quality equivalence.

Evidence: `20261013T010521Z-stage2-input-columns-8k-510475` and
`20261013T010718Z-stage2-input-columns-mtp512-511610`. The first candidate is
not retained. Before disposition of the layout, test a group-scaled Q8 Tensor
Core FP32 projection: exact integer codes enter BF16 MMA without code rounding,
FP32 accumulators receive stored scales, and output stays FP32. Only the GDN
caller permits shape-dependent summation order; default projection logit callers
retain their existing batch-invariance contract. Qualification must include the
new width boundaries and the same represented-weight FP64 criteria.

The Tensor Core revision passed all 108 TP2/3/4, BF16/Q8/Q4 cases across
1/4/8/9/31/32/33/63/64/65/257/2048 columns in 6.65 seconds. At 8K+256 it
recovered part of the prefill loss: 1059.457963 tok/s (-3.800% versus stage1),
decode 46.022080 (+2.728%), total 13.306886 seconds (+1.073%); all output
tokens matched and both CPU divisors were 4. At MTP512 the different prefill
rounding changed generated tokens and acceptance to 444/609 (72.91%), with
48 fallback steps: prefill 578.896743 (+5.255%), decode 85.935573 (+26.540%),
total 3.879014 seconds (-17.823%). This is a workload/result-dependent gain,
not a claim that projection compute made MTP 26.5% faster. CPU divisor was 5
versus stage1's 4. Peaks remained 15449/15421/15421/15421 MiB at 8K and
15479/15425/15451/15451 MiB at MTP512. Evidence:
`20261013T011656Z-stage2-input-columns-mma-8k-516164`,
`20261013T011852Z-stage2-input-columns-mma-mtp512-517468`.
A final transfer variant stages wide FP32 peer partials through copy engines
before local reduction; narrow decode keeps direct peer reads. Profile the
unstaged Tensor Core variant to attribute the remaining prefill loss.

### Campaign stage 3: parallel scratch lending (implementation prepared)

Replace permanently allocated maximum-prefill rank scratch with small graph-stable
buffers sized for the configured narrow decode/verification batch. Wide execution
borrows a contiguous range from each rank's own expert cache and restores the
narrow views after each call; loans remain owned across a layer walk and return
only after every peer reader is joined. Cache capacity can then absorb the returned
prefill memory during decode. Primary and peer caches use the same loan contract;
a diagnostic identifies the existing temporary-allocation fallback when no loan
is possible. Prepare all views before switching any rank, so allocation failure
cannot leave only some ranks using wide views. Forced-token appends return loans
when no other lane is prefilling. No weight/state duplication or new arithmetic.

Qualification: repeated prefill/decode and MTP requests with graph replay, exact
emitted-token comparison, cache capacity/loans and sampled VRAM, 64K prefill,
then larger chunk if capacity savings justify it. Final occupied-128K and multiple
lanes must check the retained lifetime change. The prepared code is not yet built
or retained; performance benefit is a hypothesis.

Copy-engine staging produced 1062.011220 prefill / 46.027106 decode tok/s and
13.287767 seconds at 8K+256: -3.568% prefill, +2.739% decode, +0.928% total
versus stage1. Emitted tokens matched. CPU divisor 5 versus 4 confounds the
small decode comparison. Evidence: `20261013T012325Z-stage2-input-columns-dma-8k-522015`.

The preceding Tensor Core/direct-peer profile at 8K (eager, attribution only)
measured a 7.737501-second prefill span. Its 144 GDN partial reductions occupied
0.674816 seconds on GPU0, versus 0.0731–0.0777 seconds per GPU for the new Q8
FP32 projection itself. Root expert gathers occupied another 1.257481 seconds.
This identifies centralized partial traffic as the remaining GDN cost, rather
than its matrix contraction. These overlapping kernel/copy durations are not
additive percentages of wall time; the unprofiled staged variant above determines
the actual improvement. The 64K staged variant is being measured before final
disposition. Profile: `20261013T012117Z-stage2-mma-profile8k-520705`.

Stage3 review also identified that borrowed peer arenas need explicit VMM access
for every participating GPU (ordinary cudaMalloc peer access does not grant VMM
access). The candidate grants this before mapping cache frames, so current and
future mappings support distributed broadcast and head gathers. This was caught
before GPU execution. Stage4's prepared policy uses nonblocking CUDA-event samples
covering draft/verify/commit and periodic K=0 probes; it is not yet built or tested
in the model and remains separate from stage3.

Stage2 disposition: **disabled; preserve as a candidate**. The final staged 64K
run (`20261013T012631Z-stage2-input-columns-dma-64k-524532`) reached
1120.685174 ±0.162680 prefill tok/s versus stage1's 1162.213635 (-3.573%);
decode 36.092600 versus 35.292916 (+2.266%, CPU divisors 4 versus 5);
total 65.667489 versus 63.739486 seconds (+3.025%). All generated tokens match
all three measured requests. Sampled peaks match at 15433/15413/15413/15413 MiB;
minimum host headroom was 15.7834 GiB. Numerical tests passed, but the 64K
latency regression fails the default acceptance criterion. The first scalar,
Tensor Core/direct-peer, and Tensor Core/staged binaries and all adverse/favorable
results remain. `tools/tp4/experiments/gdn-input-columns.patch` preserves the final
candidate. Stage3 starts from accepted stage1, without any stage2 arithmetic.

The first stage3 model trial failed before timing: a zero-sized optional scratch
region was passed to DeviceArena::alloc_bytes, which rejects zero allocations.
The acquired cache loans were returned by cleanup and vLLM was restarted. Fix:
leave absent dispatch/expert/resident scratch as an empty view instead of making
an allocation. Exact transport and expert-stream preflight had passed; this
failed run is `20261013T013551Z-stage3-lent-scratch-8k-530422` and establishes no
performance result. Preliminary startup showed 4986/5160/5159/5159 cache frames
versus accepted 4615/4768/4768/4768, but behavior and speed still need qualification.

The fixed stage3 candidate passes exact VMM transfers from every owner to every
reader, including grow/remap and captured replay, plus expert-stream exclusion
checks (4.55 seconds total). Its first successful 8K model screening preserves
all generated tokens but **regresses prefill 17.503%**: 908.541346 ±0.641616
versus 1101.305027 tok/s. Decode 44.775472 versus 44.799939 (-0.055%); request
14.747039 versus 13.165584 seconds (+12.012%). Both CPU divisors were 4.
Cache startup rises to 4986/5160/5159/5159 frames, with 987/1037/1037/1037 MiB
of wide loans returned per prompt. Sampled GPU peaks fall slightly to
15429/15413/15413/15413 MiB; host headroom 15.9749 GiB. Expert streaming counts
in the measured repetitions closely match stage1, so extra weight streaming
does not explain the large prefill loss. Profile before deciding whether this
memory path is viable; do not promote the memory saving alone. Evidence:
`20261013T013818Z-stage3-lent-scratch-8k-532652`.

Stage3 short MTP also regresses while preserving tokens and speculative behavior:
405.229278 prefill / 67.822132 decode tok/s / 5.057615 seconds versus stage1
549.994331 / 67.911937 / 4.720312. Prefill -26.321%, decode -0.132%, total
+7.146%; CPU divisor 5 versus 4. Both accept 321/477 drafts with 135 fallback
steps. With chunk512, loans are smaller (roughly 273–285 MiB/rank), and memory
peaks are 15469/15399/15423/15423 MiB. Evidence:
`20261013T014019Z-stage3-lent-scratch-mtp512-533857`.

Stage3 attribution: eager 8K prefill spans 9.030513 seconds. Root expert gather
remains 1.256023 seconds and cache relocation only 0.004632 seconds; arithmetic
and expert H2D volume do not explain the loss. Pitched peer copies are slower.
A focused, exact-output probe across all four cards distinguishes concurrency:
normal source/destination gather 1.5046 ms, VMM source/native destination
2.5283 ms, both VMM 2.4786 ms. Explicit cudaMemcpy3DPeerAsync improves the latter
to 1.8438 ms but remains slower than normal inference's native path. Owner-push
variants take 3.39–3.54 ms and lose. A single-copy probe was 0.446–0.451 ms for
both native and VMM: that isolated bandwidth result would have missed the
concurrent regression. The temporary probe first failed linking against the
container's driver shim; linking CUDA's full driver stub fixed it. Probe timing
is not an end-to-end gain. Sources/logs are retained under campaign-stage3 on
the server; the full model profile is `20261013T014248Z-stage3-lent-profile8k-536001`.

Revise the candidate to keep input/output and head-state transfer scratch in
native allocations; lend the expert buffers and local contraction workspace.
Separately, size output scratch as max(dense_rows * prefill_columns,
vocabulary_rows * narrow_columns), because Program's vocabulary logits are
bounded by lanes * max_width and prefill selects its final logit column. This
removes the previous vocabulary_rows * prefill_columns over-allocation, without
changing weights, state or arithmetic. The full-loan candidate and adverse
measurements remain; the revised candidate still needs all affected model checks.

The revised selective-loan screen preserves all generated tokens and speculative
counts. At 8K+256/chunk2048: 1101.267483 ±0.650231 prefill tok/s,
44.764366 ±0.070462 decode, 13.170123 ±0.010963 seconds, versus stage1
1101.305027 / 44.799939 / 13.165584 (-0.003%, -0.079%, +0.034%).
Both CPU divisors were 4. Short MTP512: 550.345190 prefill, 67.916486 decode,
4.719251 seconds (+0.064%, +0.007%, -0.022%); divisors 5 versus 4.
These are effectively matched observations, not established speed gains.
The short-MTP decode cache reports 4650/4955/4970/4969 frames, with loans
42/55/55/55 MiB returned; larger chunks recover more memory. Evidence:
`20261013T015420Z-stage3-selective-scratch-8k-545049` and
`20261013T015616Z-stage3-selective-scratch-mtp512-546322`. The 64K/chunk2048
and chunk4096 comparisons are running before acceptance.

Stage4 candidate is prepared separately: per-batch measured draft/verify/commit
costs replace the fixed width-cost prior through nonblocking CUDA-event samples.
Initial and periodic probes include plain K=0; confidence survival and conditional
acceptance estimate expected useful output. The single-GPU policy remains as
before. The deterministic host policy test passes fast/slow drafter transitions,
all-width probing and rejection-driven plain decode. GPU integration still needs
model qualification. The planned comparison covers three 512-token corpus slices,
an 8K high-acceptance prompt and two simultaneous requests, with 128K INT8 capacity
per lane. Any changed tokens or acceptance will be disclosed with speed results.

Stage3 64K/chunk2048: 1160.769971 ±0.537562 prefill, 35.474709 decode,
63.774501 seconds versus stage1 1162.213635 / 35.292916 / 63.739486:
-0.124%, +0.515%, +0.055%. CPU divisor 4 versus 5; all tokens match.
Startup cache gains approximately 199/220/219/219 frames (about 525–580 MiB
per card), and sampled peaks are 15419/15411/15411/15411 MiB. This is a memory
improvement with essentially matched observed speed, not a prefill speed claim.
At chunk4096, selective scratch gives 1196.222062 ±0.196901 prefill,
35.140094 decode, 62.173194 seconds: +2.926% / -0.433% / -2.457% versus
stage1/chunk2048, with matching tokens and CPU divisors 5. The changed chunk is
a confound for attributing that gain to scratch; a stage1/chunk4096 comparator
and fresh selective/chunk4096 launch are queued. Evidence:
`20261013T020358Z-stage3-selective-scratch-64k-550345`,
`20261013T021004Z-stage3-selective-scratch-64k-chunk4096-553396`.

Stage5 preparation is isolated from measured code. It assigns decoder layers
0–23 to GPU0/1 and 24–47 to GPU2/3, with matching complete head groups, compact
recurrent state and per-layer expert owners. A grouped layer-major walk overlaps
opposite stages on separate streams, delaying the TP4 final/MTP work until both
stages join. This prototype retains GPU0 for small residual/index coordination;
it is not a fully GPU2-local second-stage coordinator. Additional coordinator
scratch has a real cache cost. No performance or correctness result yet; the
candidate must not be represented as working until its tests and model run pass.

User steering: retain the selective scratch reclamation for its memory benefit
regardless of a speed-neutral result (it can make room for larger KV). Stage3 is
therefore retained as a capacity improvement; speed comparisons remain disclosed
and unchanged. Recovered expert frames can be surrendered to physical KV growth,
but a new maximum occupied context has not yet been measured. Final occupied-128K
qualification is still required. The full-loan/VMM transfer regression stays
rejected. Chunk4096 is a tuning candidate, independently of retaining the memory
change; its baseline comparison remains useful to choose the final chunk.

Prepared later candidates remain isolated from timed builds. Stage5's host
partition check now verifies each recurrent layer's unique complete head groups,
and all 24,576 experts have exactly one owner on the matching stage (6,144 per
GPU). GPU dispatch checks will cover explicit owner maps, empty owners, the
1024-entry route boundary and graph replay. Stage6 prepares FP32 owner-local
weighted sums plus an independent FP64 oracle with NaN-poisoned unowned slots;
its changed summation order needs model/MTP qualification, and its extra partial
buffers are a measurable memory cost. Stage7 prepares fixed KV owners 0/1 and
124/124/132/132 offloaded experts per layer, moving only the last four formerly
owned experts off each KV owner. The independent host ownership checks pass;
none of these later candidates has yet been built or timed on GPUs.

Stage3 same-chunk comparison resolves the attribution: stage1/chunk4096 gives
1197.967717 prefill, 34.679325 decode, 62.190752 seconds. Selective/chunk4096
launches give 1196.222062 and 1196.321472 prefill (-0.146%/-0.137%),
35.140094 and 35.151771 decode (+1.329%/+1.362%, CPU divisor 5 versus 4),
62.173194 and 62.166789 seconds (-0.028%/-0.039%). All tokens match. Thus
larger chunks explain the ~2.9% prefill gain; scratch reclamation itself shows
a small repeatable prefill cost at this chunk, with essentially neutral total
latency and confounded decode gains. It remains retained for memory, per the
user's explicit preference. Startup cache is 4642/4818/4817/4817 frames versus
4243/4381/4381/4380: roughly 1.03–1.13 GiB recovered per GPU at chunk4096.
The repeat's GPU0 sampled clock includes a 300MHz transition outside a stable
compute plateau; this is not evidence of sustained throttling. Evidence:
`20261013T021856Z-stage3-baseline-chunk4096-558464`,
`20261013T022454Z-stage3-selective-chunk4096-repeat-562406`.

`tools/tp4/experiments/full-scratch-loans.patch` now preserves the rejected
full-loan implementation against selective scratch and passes apply-check.
Stage4's GPU-free build of benchmark and host cost-policy test succeeds; its
model baseline/candidate campaign is running with no overlapping compilation.

Stage4 disposition: **disabled; preserve calibrated-mtp-cost.patch**. Five
matched MTP workloads (three 512-token corpus slices, 8K, and two concurrent
512-token requests) pass with exact generated tokens. Single-request decode
changes are +0.543%, +0.725%, +0.085%, and -0.814%; total-time changes are
-0.336%, -0.484%, -0.024%, and +0.215%. Two-request decode falls 7.978%
(48.2873 to 44.4348 tok/s), and request time rises 6.598% (7.17073 to 7.64387 s).
All CPU divisors are 5. The paired candidate also varies more within its run
(2.148 versus 0.602 tok/s SD). Its acceptance falls from 343/579 to 322/763,
with 826 rather than 682 fallback steps. Measured GPU costs alone and periodic
width probes do not establish a better end-to-end policy on these workloads.
No kernel-speed claim is made; the policy changes actual width/acceptance mix.
These mixed single-request results and adverse concurrent result do not justify
default promotion. The existing policy is restored for stages5–7. Evidence:
`campaign-stage4/comparison.json`, runs `20261013T023059Z` through
`20261013T024159Z`. Host policy tests and GPU integration passed; broad quality
and general prompt-distribution benefits are not established.

Stage5's initial preflight stopped before model timing. Partition, state-image/KV,
GDN, and explicit expert-owner dispatch checks passed; the weight oracle aborted
on a BF16 launch at the wide 96-row semantic-head test. Line-buffered isolation
identified the existing BF16 shared-memory helper's process-global one-time
attribute initialization: the >48-KiB allowance was not set on subsequent CUDA
devices. Use the existing per-device monotonic attribute helper instead. Both
serial and overlapped binaries are rebuilt with this correction before timing;
this correctness fix is independent of the pipeline candidate. vLLM was restored
after each failed preflight. Evidence: stage5 `screen.log`, `weights-debug.log`.

With per-device BF16 attributes fixed, all five stage5 qualification checks pass
(12.74 s), including the independent FP64 weight oracle and captured replay.
The first model load then fails the Program's fixed-allocation assertion: the
new secondary workspace was included in allocation but omitted from the declared
workspace bytes. Add max(narrow,wide) secondary workspace to the device plan for
parallel execution; keep the assertion. Both comparison binaries are rebuilt
with that accounting. This failed load has no performance result; evidence:
`20261013T025512Z-stage5-serial-8k-583662`, `screen-plan-failure.log`.

Stage5 corrected 8K/chunk2048 screen: serial TP2+PP2 reaches
1103.248257 ±1.656408 prefill / 45.407405 ±0.229113 decode tok/s,
13.074998 ±0.041401 seconds. Accepted TP4/selective gives 1101.267483 /
44.764366 / 13.170123: +0.180% prefill, +1.436% decode, -0.722% wall.
The overlapped variant reaches 1054.908847 ±1.210626 / 45.372828 ±0.208024,
13.419021 ±0.034835 seconds: -4.382% prefill, -0.076% decode, +2.631% wall
versus its serial layout. Both CPU divisors are 4. Every generated token matches
between serial/pipeline and TP4. Both pipeline-layout peaks are
15401/15383/15435/15435 MiB, minimum host headroom ~16.17–16.20 GiB.
GPU0 decode cache has 4267 frames versus selective TP4's ~4800; the extra
coordinator workspace costs capacity even though aggregate peak VRAM is similar.
Grouping reloads experts (two passes at 8K) and competes for PCIe bandwidth.
A 64K comparison and eager inference trace are pending before disposition; these
8K numbers do not establish a general TP2+PP2 win. Evidence:
`20261013T025736Z-stage5-serial-8k-586035`,
`20261013T025947Z-stage5-pipeline-8k-587649`.

Stage5 initial 64K/chunk2048 result is mixed: 1181.437710 ±0.300684 prefill
(+1.781% versus selective TP4), 34.819636 ±0.026544 decode (-1.847%),
62.934749 ±0.034725 seconds (-1.317%). CPU divisors both4; all tokens match.
Peaks 15389/15379/15427/15427 MiB; host minimum16.0775GiB. This is one launch,
not a qualified general win. The eager 8K trace shows only 0.002177 s of kernel
overlap between GPU1 (first stage) and GPU2/3 (second stage), in a 7.787056 s
prefill span. GPU0 expert gather totals1.551642s. CUDA launch API time4.500037s
and pitched-copy API time1.127670s indicate submission blocking; queuing an
entire half-model before the other stage does not achieve the intended overlap.
Therefore defer disposition and interleave one chunk/layer per stage before
retesting. This is a scheduling correction, not a change to the mathematical
layout. Stage6's GPU-free binaries can finish building, but its timing is deferred.
Evidence: `20261013T030241Z-stage5-pipeline-64k-589981`,
`20261013T030858Z-stage5-pipeline-profile8k-593129`.

Stage6's preflight passes exact transport, the extended FP64 weight oracle
(including the BF16 shared-memory regression shape), and the new owner-combine
FP64/NaN-poison/graph oracle (13.58 s). Fresh accepted baselines preserve tokens
and differ by ~0.1% from stage3 at 8K/short MTP and +0.070% prefill at64K.
The initial all-width candidate reaches1152.789815 ±0.388143 prefill at8K
versus1102.360964 (+4.575%); ordinary decode44.393981 versus44.783966 (-0.871%,
CPU divisor5 versus4), request12.885313 versus13.160104s (-2.088%). Tokens match.
GPU0 sampled peak adds20MiB; others match. Short MTP512 reaches595.672354 /
74.498498 /4.313684s versus549.797670 /67.873520 /4.722291s (+8.344% prefill,
+9.761% decode,-8.653% wall; both divisors5), but generated tokens change and
acceptance changes321/477 to378/534. The independent numeric oracle passes;
this MTP result is not pure kernel speedup or evidence of whole-model quality
parity. Initial64K is running. Prepare a shape-based wide-only version (T>8):
small calls can retain the existing ordered gather, avoiding extra reduction
launches. This is not corpus-dependent policy, and the new variant still needs
model qualification. Evidence: `20261013T032412Z-stage6-owner-combine-8k-603435`,
`20261013T032558Z-stage6-owner-combine-mtp512-604531`.

The defer marker did not change the already-loaded stage6 shell script, so its
controlled GPU sequence was allowed to finish; no builds overlap these timings.
The stage5 revision remains local while stage6 runs. It interleaves single
chunk/layer submissions, skips primary expert kernels when that pipeline stage
owns no primary experts, and joins the auxiliary stream on exception/abandonment
before returning borrowed storage. These changes are not yet GPU-tested.

Stage6 all-width64K confirms a prefill gain:1215.941481 ±0.044909 versus
1161.583153 ±0.381507 tok/s (+4.680%), plain decode35.318669 versus35.479326
(-0.453%, both CPU divisors5), request61.244353 versus63.733636s (-3.906%).
Every generated token matches. GPU0 peak15439 versus15419MiB; peers15411MiB
unchanged, host minimum15.9926GiB. Evidence:
`20261013T032704Z-stage6-owner-combine-64k-605176`.

The wide-only revision also reuses the now-consumed expert-output arena for
FP32 partials. Compute the primary sum in already-existing dense-output scratch,
then copy it and the peers' sums into the old expert outputs after all primary
reads finish. Use this route only for T>8 and K>=2*rank_count, so the buffer fits;
other shapes keep the established ordered gather/combine. Remove the earlier
extra Forward workspace allocation. Extend the FP64 oracle with poisoned
outputs, in-place arena reuse, 9/512/2048 columns and exact graph replay.
The build succeeds. A quoting error in the temporary launch script stopped it
before GPU timing and restored vLLM; the corrected separate runner now performs
the aliasing check and model comparisons. No performance result comes from that
failed launcher (`campaign-stage6/wide.log`).

Stage6 wide-only/arena-reuse qualification passes the independent FP64 aliasing
and CUDA Graph test in5.06s. At8K it reaches1150.042285 ±0.427744 prefill
(+4.325%),44.765495 ±0.081596 decode (-0.041%),12.854121 ±0.013141s
(-2.325%); CPU divisors both4 and all tokens match. At64K it reaches
1213.937002 ±0.153368 prefill (+4.507%),35.481616 ±0.039205 decode
(+0.006%, divisor4 versus5),61.300541 ±0.012008s (-3.818%); all tokens match.
Sampled GPU peaks exactly match each fresh baseline. MTP512 reaches586.946965 /
76.045619 /4.256781s (+6.757% prefill,+12.040% decode,-9.858% wall), both
divisors5, but all three output sequences change and acceptance becomes396/576
versus321/477. Wide prefill changes rounding even though narrow decoding keeps
its original ordered combine. This does not establish quality parity or a
12% kernel speedup. Retain the wide-only version for its measured prefill gain
and recovered memory, subject to final integration/capacity/serving checks.
The all-width version remains an isolated candidate with its adverse ordinary
decode and extra20MiB root peak preserved above. Evidence: campaign-stage6
wide-8k.json,wide-mtp512.json,wide-64k.json and corresponding run directories.

The stage5 interleaved schedule passes all five preflight checks (12.73s).
At8K/chunk2048 it reaches1241.250021 ±3.329487 prefill,45.393931 ±0.248857
decode,12.250554 ±0.048592s. Against serial TP2+PP2: +12.509% prefill,
-0.030% decode,-6.305% wall. Against newly retained stage6 TP4: +7.931%,
+1.404%,-4.696%. CPU divisors all4 and all emitted tokens match. Peaks
15401/15383/15435/15435MiB, host minimum17.2857GiB. The new eager trace has
6.636105s prefill with0.112943s stage-kernel overlap versus0.002177s previously;
GPU0 gather totals1.713741s. These overlapping durations are not wall percentages.
The pipeline still uses GPU0 as residual coordinator, costs ~1.4GiB root cache,
and currently submits a whole walk in one Engine step (cancellation/fairness
must be fixed before product promotion). Continue64K/MTP and combine it with
stage6's smaller weighted transfers before disposition. Evidence:
`20261013T034917Z-stage5-interleaved-8k-619073`,
`20261013T035121Z-stage5-interleaved-profile8k-621395`, campaign-stage5 comparison
and overlap JSON. The preserved pipeline patch now contains the revised schedule
against the archived post-stage3 baseline, including the per-device BF16 fix.

Stage5 interleaved64K reaches1566.334577 ±1.185898 prefill /34.841870 ±0.006427
decode /49.293733 ±0.032156s. Against stage6 TP4: +29.029% prefill,-1.803%
decode,-19.587% wall, both CPU divisors4 and all emitted tokens match. Peaks
15389/15379/15427/15427MiB; host minimum16.6421GiB. The MTP512 follow-up fails
before timing because the recurrent replay validator does not recognize the
18-layer,8-key/24-value-head pipeline shard. Extend both host validation and the
CUDA fold launch to that exact geometry and the existing independent fold oracle
to width5/batch3 commits5/3/0 and width17/batch1 commit13, with separate destinations.
The first preflight after the validator-only edit also rejects the missing CUDA
launch registration; there is no numerical or performance result from either
failure. After both changes, the combine and fold checks pass and model tests
resume. vLLM is restored on failed runs. Evidence:
`20261013T035423Z-stage5-interleaved-64k-623953`,
`20261013T035932Z-stage5-interleaved-mtp512-626973`,
`20261013T040435Z-stage5-combined-8k-630292`.

Combining stage5 with stage6 owner partials gives1408.443729 ±2.789405 prefill,
45.403496 ±0.258016 decode,11.465828 ±0.044162s at8K. Against stage6 TP4:
+22.469%,+1.425%,-10.800%, both CPU divisors4 and all tokens match. Peaks remain
15401/15383/15435/15435MiB; host minimum16.7890GiB. The FP64 combine oracle also
covers both pipeline owner pairs with poisoned inactive owners and alias replay.
64K/MTP are pending. Prepare an explicit --pipeline-parallel option across Engine,
CLI, server and benchmark, keeping normal TP available; replace the whole-walk
submission with bounded resumable steps before product qualification. This expands
integration to preserve cancellation/fairness and user choice, not benchmark scope.
Evidence: `20261013T040754Z-stage5-combined-8k-633216`.

Stage5+6 combined64K reaches1801.563264 ±0.554134 prefill,34.938239 ±0.046322
decode,43.815460 ±0.010637s. Against stage6 TP4: +48.407% prefill,-1.531%
decode,-28.524% wall; CPU divisor5 versus4 confounds the small decode difference.
All tokens match. Peaks15401/15381/15433/15433MiB; host minimum16.6276GiB.
This enters the user's reported Strata range, but Dense8/NVFP4 versus IQ3_S and
unmatched benchmark methods prevent an engine-to-engine speed or quality claim.
MTP512 succeeds after the replay fix:388.629684 ±0.717723 prefill,
86.881206 ±0.085024 decode,4.278171 ±0.005379s. Against stage6 TP4: -33.788%
prefill,+14.249% decode,+0.502% total time, both divisors5. All three outputs
change; acceptance459/609 versus396/576. Short prompts do not use prompt-chunk
pipeline overlap, and faster speculative decode is confounded by changed output
and acceptance. Preserve this adverse prefill/total-time result. Keep TP4 as the
default and qualify pipeline as an explicit long-prompt option. Numerical Op
checks pass; whole-model quality parity is not established. Evidence:
`20261013T040949Z-stage5-combined-64k-634682`,
`20261013T041427Z-stage5-combined-mtp512-637206`.

Stage7 passes partition, represented-weight, physical-state, mapped expert-owner,
transport and combine qualification. Fixed KV/even experts at8K reaches1155.598287 /
45.109876 /12.780852s versus stage6's1150.042285 /44.765495 /12.854121s:
+0.483% prefill,+0.769% decode,-0.570% wall, both CPU divisors4, exact tokens.
Rebalanced experts reach1153.965981 /45.098482 /12.790841s (+0.341%,+0.744%,
-0.492%, divisor5 versus4), exact tokens. Both peaks15427/15423/15413/15413MiB.
Fixed/even MTP512 gives586.411577 /75.993063 /4.261979s (-0.091%,-0.069%,
+0.122%), same tokens and acceptance396/576, both divisors5. Rebalancing gives
568.635350 /86.667972 /3.870469s (-3.120%,+13.968%,-9.075%), but tokens change
and acceptance becomes450/582. This is not an isolated communication-speed gain.
MTP peaks15453/15461/15411/15411MiB for both. Host minimum16.97–17.20GiB.
Evidence: stage7 runs20261013T041807Z through042311Z and comparison JSON.

Fixed/even64K reaches1214.307366 ±0.134782 prefill /37.880407 ±0.050853 decode /
60.829385 ±0.002279s. Against rotating stage6 TP4: +0.031% prefill,+6.761% decode,
-0.769% wall, all tokens match. CPU divisor5 versus4 confounds attribution;
repeat against the selectable-layout TP4 baseline before interpreting the decode
gain. Peaks15415/15417/15413/15413MiB, host minimum17.0601GiB. Root cache at64K
has4550 frames versus rotating TP4's4717: memory shifts between owners rather
than increasing aggregate capacity. Evidence:
`20261013T042448Z-stage7-fixed-64k-645961`.

Rebalanced64K reaches1214.578849 ±0.338104 prefill /38.098794 ±0.033302 decode /
60.778817 ±0.019455s. Against rotating stage6 TP4: +0.053%,+7.376%,-0.851%;
against fixed/even: +0.022%,+0.577%,-0.083%. All tokens match. Both fixed variants
use CPU divisor5, versus4 in the rotating baseline. Peaks15411/15415/15413/15413MiB,
host minimum16.3780GiB. The small incremental expert-placement benefit does not
establish a general throughput gain; short-MTP output and acceptance changed.
Evidence: `20261013T043036Z-stage7-fixed-rebalanced-64k-649144`.

### Integrated layout qualification

The active code keeps TP4 as default and exposes `--pipeline-parallel` consistently
through CLI, serving, benchmark and Engine. Pipeline scheduling now yields at bounded
Engine steps, joining both streams before another lane or cancellation; the prototype
submitted the whole walk. Primary streaming-ring sizing now counts actual owned
experts/layers, including 256 experts per pipeline stage, instead of assuming E/4.
These changes require fresh measurements rather than inheriting the prototype's1802.
All model tests continue to reserve131072 INT8 tokens; builds precede GPU timings.

The first integrated build found a scope typo in cache ownership setup (`p.pipeline`
outside the implementation scope); corrected to `impl_->pipeline`. No model test ran
with that failed build. The resumed CLI/server/benchmark build succeeds, and all12
selected ownership, transport, represented-weight, state, recurrence/replay, expert
stream/combine and public-option tests pass in26.88s. Raw logs are in
`tp4-results/campaign-final/`, including the preserved failed build.

Integrated TP4 at8192+256/chunk2048 measures1151.614517 ±0.242139 prefill,
44.801507 ±0.083725 ordinary decode,12.842441 ±0.011593s. Against accepted stage6
wide-only TP4: +0.137%,+0.080%,-0.091%, both CPU divisor4, exact tokens in all3
repetitions, identical sampled peaks15435/15419/15419/15419MiB. This is a regression
check of selectable-layout integration, not an independent optimization claim.
Host minimum17.1682GiB. Evidence:`20261013T045111Z-final-tp4-8k-660867`.

Integrated rotating TP4 at64K measures1215.617755 ±0.124682 prefill,
35.444089 ±0.025914 ordinary decode,61.239813 ±0.000640s; versus stage6 TP4:
+0.138%,-0.106%,-0.099%, with exact tokens and identical peaks15419/15411/15411/15411MiB.
CPU divisor5 versus4 remains a confound for this small integration comparison.
The immediately following fixed/even repeat gives1214.150399 ±0.122379 /
37.943990 ±0.023253 /60.825121 ±0.007263s: -0.121% prefill,+7.053% decode,
-0.677% total time, now with divisor5 in both launches and exact tokens. Thus the
ordinary-decode benefit repeats; it is not dismissed as CPU-autotune variation.
Peaks15411/15415/15413/15413MiB and host minimum16.9967GiB. Retain fixed KV/even
experts as explicit `--fixed-kv-owners` for four-GPU TP; leave rotating ownership
as default and reject combining fixed owners with pipeline stages. Keep expert
rebalancing deferred because its incremental64K gain is small and MTP output changed.
The new option needs its own integrated checks before completion. Evidence:
`20261013T045328Z-final-tp4-64k-662803`,
`20261013T045918Z-stage7-fixed-64k-repeat-666211`.

The bounded pipeline at8K measures1401.136344 ±3.874205 prefill,
45.372246 ±0.307976 ordinary decode,11.500346 ±0.054773s. Against integrated TP4:
+21.667%,+1.274%,-10.450%, exact tokens. CPU divisor5 versus4 confounds the small
decode comparison. Against the combined whole-walk prototype, prefill is0.52%
lower after the bounded-scheduling and ring-sizing changes; this comparison does
not isolate their individual costs. Peaks15401/15383/15435/15435MiB and host
minimum17.1696GiB. Evidence:`20261013T050505Z-final-pipeline-8k-669100`.

The bounded pipeline at64K measures1798.946648 ±0.169032 prefill,
34.908680 ±0.047226 ordinary decode,43.864798 ±0.006994s. Against integrated TP4:
+47.986% prefill,-1.511% decode,-28.372% request time; both CPU divisor5 and exact
tokens. Against the combined prototype, prefill is0.145% lower and decode0.085%
lower after scheduling/ring integration. Peaks15389/15379/15427/15427MiB and host
minimum17.0504GiB. The measured long-prompt prefill benefit survives bounded Engine
steps; the decode regression remains visible. Evidence:
`20261013T050648Z-final-pipeline-64k-670238`.

Integrated pipeline MTP512 reproduces the prototype:388.537460 ±0.814131 prefill,
86.851457 ±0.098765 speculative decode,4.279477 ±0.005840s. Against stage6 TP4:
-33.804% prefill,+14.210% decode,+0.533% request time, both CPU divisor5. Outputs
differ in every repetition; acceptance459/609 versus396/576, as in the prototype.
Peaks15441/15439/15405/15405MiB, host minimum17.0664GiB. The faster speculative
rate does not override the slower full request or establish quality parity.
Evidence:`20261013T051114Z-final-pipeline-mtp512-672645`.

Pipeline chunk4096 loses against2048 at64K:1748.370237 ±0.344711 prefill,
33.162047 ±0.044374 decode,45.310830 ±0.016912s, or-2.811%,-5.003%,+3.297%.
Tokens still match. CPU divisor4 versus5 confounds how much of the decode change
comes from chunking; the whole tested configuration is slower. Primary warm cache
falls from4159 to3504 frames, with more workspace reserved; peers hold4714 instead
of about4891 frames. Peaks15429/15383/15431/15431MiB, host minimum16.9523GiB.
Keep chunk2048 for pipeline and use that setting for final serving qualification.
Evidence:`20261013T051220Z-final-pipeline-64k-chunk4096-673363`.

Integrated rotating TP4 MTP8K reaches1140.009895 ±0.311564 prefill,
110.484953 ±0.087083 speculative decode,9.515393 ±0.002682s. Versus the
pre-campaign8K MTP measurement: +9.186%,+1.676%,-6.847%, with unchanged acceptance
567/579 and201 rounds. CPU divisor5 versus4 confounds the small decode gain.
The older report did not record generated token IDs, so exact output comparison
is unavailable (not a demonstrated mismatch). Peaks15467/15403/15429/15429MiB,
host minimum16.9560GiB. Evidence:`20261013T051651Z-final-tp4-mtp8k-675650`.

Integrated pipeline MTP8K measures1346.527600 ±1.502441 prefill,
104.514191 ±0.142095 speculative decode,8.551532 ±0.005914s. Against the integrated
rotating TP4 MTP8K result: +18.115% prefill,-5.404% decode,-10.129% full-request
time. Both use CPU divisor5, all generated tokens match, and acceptance is identical
at567/579 with201 rounds. This is a real speculative-decode regression at matched
output/acceptance, unlike the acceptance-confounded short-MTP gain. Peaks are
15419/15393/15423/15423MiB; host minimum17.0604GiB. Evidence:
`20261013T051822Z-final-pipeline-mtp8k-676729`.

The explicit fixed-KV option builds in CLI, server and benchmark. Six affected
partition, represented-weight, physical-state/snapshot and public-option checks
pass in 9.16 s. Fixed ownership covers both early/late attention layers and wide
BF16/Q8 projection shapes against the existing independent FP64 oracle. Options
reject non-four-GPU plans and combining fixed ownership with pipeline staging.
GDN/expert placement remains unchanged. Source and test logs are in
`tp4-results/campaign-final/build-fixed-option.log` and the corresponding campaign log.

Integrated fixed-KV TP4 at8K measures1156.245551 ±0.749155 prefill,
45.152865 ±0.070974 ordinary decode,12.769621 ±0.013838s. Versus integrated
rotating TP4: +0.402%,+0.784%,-0.567%, both CPU divisor4 and exact tokens.
Peaks15427/15423/15413/15413MiB; host minimum17.0747GiB. This reproduces the earlier
fixed/even candidate's small8K benefit. Evidence:
`20261013T052757Z-final-fixed-8k-684254`.

Integrated fixed-KV64K reproduces the candidate and its repeat:
1214.443312 ±0.104413 prefill,37.950892 ±0.027019 ordinary decode,
60.819249 ±0.011242s. Against integrated rotating TP4: -0.097% prefill,
+7.073% decode,-0.687% full-request time; both CPU divisor5 and exact tokens.
Peaks15411/15415/15413/15413MiB; host minimum17.0165GiB. Evidence:
`20261013T052955Z-final-fixed-64k-685672`.

Integrated fixed-KV MTP512 gives587.136126 ±1.051872 prefill,
76.083276 ±0.022380 decode,4.256107 ±0.002389s. Versus stage6 rotating TP4:
+0.032%,+0.050%,-0.016%, exact tokens and acceptance396/576, CPU divisor5 in both.
Peaks15453/15461/15411/15411MiB, host minimum16.5907GiB.
Fixed-KV MTP8K gives1143.872930 ±0.351036 prefill,110.905950 ±0.005662 decode,
9.483441 ±0.003729s. Against integrated rotating TP4: +0.339%,+0.381%,-0.336%,
exact tokens and acceptance567/579, CPU divisor5 in both. Peaks15461/15407/15425/15425MiB,
host minimum16.9539GiB. Fixed ownership's measured MTP benefit is small; do not
extend its7.07% ordinary64K decoding gain to speculation. Evidence:
`20261013T053543Z-final-fixed-mtp512-688691`,
`20261013T053647Z-final-fixed-mtp8k-689413`.

The final near128K probes use MTP3 plus the optimized draft head for all three
layouts, with131000 prompt tokens and32 decode steps, chunk2048, no warmup and
one measured repetition. This extends the earlier plain-decoding capacity check
to the retained speculative buffers and physical KV growth. These cold repeated-corpus
runs establish capacity/execution, not warmed throughput or long-context quality.

Rotating TP4 completes the near128K MTP probe with131000 prompt tokens and33
outputs, backing the full2048-page logical pool (1874MiB across its physical
owners), then shrinking to512 pages and returning frames. It measures1163.541970
prefill /23.374312 decode tok/s /114.203705s, with22/22 proposals accepted.
Peaks15425/15385/15411/15411MiB and host minimum16.9536GiB. The first decode has
only roughly12–14% expert-cache hits, unlike the warmed performance runs.
Pipeline also completes, with identical33 output IDs and22/22 accepted proposals:
1603.034493 prefill /17.814826 decode /83.753604s. Against that cold TP4 probe:
+37.772% prefill,-23.785% decode,-26.663% total time, both CPU divisor5. The cold
pipeline decode penalty is material and must remain visible alongside its prefill
benefit. Peaks15395/15383/15407/15407MiB, host minimum16.9507GiB. Both are single
cold capacity observations, not estimates of warmed long-context throughput.
Evidence:`20261013T053851Z-final-tp4-capacity128k-mtp-691262`,
`20261013T054125Z-final-pipeline-capacity128k-mtp-692686`.

Fixed-KV also completes131000+33 with MTP, identical output IDs and22/22 accepted
proposals:1161.492792 prefill /24.301032 decode /114.338543s. Relative to the cold
rotating probe: -0.176% prefill,+3.965% decode,+0.118% total time, CPU divisor5.
Peaks15433/15395/15419/15419MiB; host minimum16.9098GiB. Its KV pool also grows to
2048 pages then shrinks to512, with primary expert frames4402→4111→4377.
All three layouts therefore pass near-full physical context with speculation;
none of these single cold observations establishes warmed128K throughput or model
quality at that context. Evidence:`20261013T054329Z-final-fixed-capacity128k-mtp-693906`.

The first server launch reached readiness with two lanes and a requested262144-token
INT8 pool (rounded to262272), but the HTTP probe failed because Node was absent
from the target host. This is a test-client setup failure, not a model result.
The server was removed and vLLM restored. Qualification now runs the same Node22
client on the workstation through a loopback-only SSH tunnel, with the target
server's cleanup trap waiting for its exit status. Preserve the initial failure
in `campaign-final/serving-client-path-failure.txt`.

Rotating TP4 passes the HTTP probe with MTP3, two lanes, chunk2048, max-context131072,
requested shared KV capacity262144 (rounded262272 INT8 tokens), and2GiB host prefix
cache. The17016-token prompt returns the correct arithmetic answer. Repeated and
concurrent requests reuse16384 cached tokens; the unchanged question returns exactly
the same message across cold, cached, concurrent and post-cancellation requests.
Observed request times:15.565,1.980,3.655/3.723 concurrently,1.797s. These are
functional probe timings, not repeated serving performance estimates.
A separate64901-token stream is aborted after1.5s; server logs confirm cancellation
with zero output after2.5s (expected HTTP499). The following short request finishes
in1.457s, and later long-prefix reuse remains correct (16384 cached tokens,1.859s).
Evidence:`campaign-final/serving-{check,responses}.json`, cancellation JSON, metrics
and server log. The server is removed and vLLM restored before the next layout.

Pipeline passes the same server probe with correct, identical arithmetic messages
and16384-token prefix reuse, including both simultaneous lanes and reuse after
cancellation. Timings are14.962s cold,3.089s cached,5.669/5.769s concurrent,2.737s
cached afterward. These cached/concurrent observations are slower than TP4's
1.980/1.797s cached and3.655/3.723s concurrent; preserve that adverse result despite
passing correctness. They are one functional sequence, not a repeated serving
benchmark. The cancelled long request has zero output; the following short request
completes in1.263s. Pipeline is an explicit long-uncached-prefill option, not the
default for short/cache-heavy chat. Evidence:`campaign-final-pipeline/serving-*`.

Fixed-KV passes the same HTTP checks, with correct matched messages and16384 cached
tokens across repeated/concurrent/post-cancellation requests. Timings15.484s cold,
1.949s cached,3.712/3.739s concurrent,1.816s cached afterward. Cancellation occurs
with zero output after2.5s; the subsequent short request takes1.455s and later
prefix reuse takes1.849s. These observations are close to rotating TP4; no repeated
serving-performance claim is made. All three layouts now pass this functional
serving scope. Evidence:`campaign-final-fixed/serving-*`.

### Campaign completion

All seven experiments are implemented and evaluated. Retained paths are owner-local
attention, selective scratch reclamation, wide weighted expert partials, optional
bounded TP2+PP2, and optional fixed-KV/even-expert TP4. All three layouts pass the
near128K MTP and functional HTTP scopes above. Failed/slower candidates, original
patch bases, relevant traces, raw reports and final comparison JSON remain saved.
The target's CLI, server and benchmark binaries are built under `build-tp4/`;
`campaign-final/all-layouts` preserves the final benchmark binary. The editable
local tree was uncommitted at campaign completion; no PR was created then.

The current-code whole-model evidence does not establish quality parity, arbitrary
prompt acceptance, warmed128K throughput, more than two serving lanes, or broad
protocol/load behavior. Pipeline still coordinates residuals/indexing on GPU0 and
has a second primary workspace; it is not a fully GPU2-local second-stage coordinator.
Keep rotating TP4 as default, fixed KV as an optional ordinary-decode tradeoff,
and pipeline/chunk2048 for long uncached prefill. Do not infer speculative or
cache-heavy latency gains from pipeline's64K prefill improvement.

The final server probe containers and workstation SSH tunnel are removed.
vLLM is active and `/health` on8030 returns HTTP200 after reload; unrelated
containers remain running. `git diff --check` passes. Product, architecture,
benchmark, test and candidate documentation now describe the retained options
and their limitations. Probe scripts and responses are archived in the task's
`tp4-results/campaign-final*` directories; the HTTP client uses workstation Node22
through a loopback SSH tunnel because the server has no Node runtime.
