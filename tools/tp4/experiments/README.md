# Deferred performance candidates

The active implementation and complete evidence remain in the
[TP4 validation record](../../../docs/maintainer/multi-gpu-5060-ti-design.md).
These patches preserve concrete experiments for another comparison; they are
not enabled by the normal build.

## Concurrent expert-result writes

`owner-pushed-expert-results.patch` makes each peer return its disjoint top-k
output slots on its own stream for calls of at least 256 columns. Those writes
can overlap the primary's expert computation. The primary joins all producers
before the existing ordered combination. Narrow decode keeps the pull path.
The normal transport test retains exact concurrent-write coverage.

On 2026-10-12, with 8192 prompt + 256 decode tokens, chunk 2048, two warmups,
three measured repetitions and 131072 INT8 KV capacity, owner-push measured
995.740 ± 0.669 prefill tok/s versus 1010.416 ± 0.318 for root gather. Ordinary
decode was 44.455 versus 44.435 tok/s; request wall was 13.99795 versus 13.88147 s.
These runs preceded shared QSA selection. They do not characterize variation
between fresh launches. The user requested retaining this candidate: removing
other transfer/compute bottlenecks may change whether the overlap pays.

The later optimized 64K eager trace still attributes 10.00 s of a 59.28 s
prefill span to primary expert-result gather kernels. Root-to-peer activation
DMA also remains substantial. These overlapping costs establish a reason to
retain the candidate, but do not establish a win from concurrent result writes.

The subsequent **final-code 64K retest does show a small gain**. Alternating
fresh launches gather/push/gather/push, each with two warmups and three measured
65536+256 requests, chunk 2048 and 131072 INT8 KV, measured:

| Variant | Launch 1 prefill tok/s | Launch 2 prefill tok/s | Mean request time |
|---|---:|---:|---:|
| Root gather | 1105.856 | 1105.817 | 66.6642 s |
| Concurrent writes | 1116.643 | 1116.620 | 66.0986 s |

That is +0.976% prefill and -0.848% request time (0.566 s saved), with ordinary
decode 0.030% lower. All routing/cache diagnostic lines and sampled GPU-memory
peaks match across all four launches, and exact transport tests pass. The gain
repeated in both pairs and exceeds the observed launch spread, although two
launches per variant do not establish general variability. Preserve this positive
result alongside the earlier negative 8K result, which used earlier code.
The patch remains disabled globally: short/8K prompts, MTP and chunk 4096 need
final-code comparisons before making its T>=256 behavior a general default.
Saved server binaries are `tp4-results/concurrent64k-final-binaries/root-gather`
and `owner-push`; full evidence is in the validation record linked above.

The original patch requires its pre-campaign baseline. On a checkout of that
baseline (or after rebasing the patch onto current code), check before applying:

```bash
git apply --check tools/tp4/experiments/owner-pushed-expert-results.patch
git apply tools/tp4/experiments/owner-pushed-expert-results.patch
# Rebuild infernix_bench and infernix_parallel_transport_test using ../README.md.
```

Recompare both variants on the current code, with identical artifacts, cache
capacity, chunks and workloads. Use separate launches in alternating order to
measure between-launch variability, and test both 8K and 64K with at least 128K
INT8/FP8 capacity. Record prefill, decode, request wall, cache residency and route
counts. A timeline should establish whether writes overlap useful work or compete
with it. Keep it disabled unless the new measurements justify enabling it.

After the experiment, reverse only this patch (check first):

```bash
git apply --reverse --check tools/tp4/experiments/owner-pushed-expert-results.patch
git apply --reverse tools/tp4/experiments/owner-pushed-expert-results.patch
```

## GDN input-column FP32 partials

`gdn-input-columns.patch` preserves campaign stage2: uniquely load GDN output
weights by local input heads, compute FP32 partials (group-scaled Q8 Tensor Core
projection for wide calls), stage wide peer partials with copy engines, sum in
FP32 and round once to BF16. It changes summation order and is disabled.
Independent FP64 qualification passed 108 TP2/3/4 cases. Against owner-local
attention, 8K prefill fell 3.57% and whole-request time rose 0.93%; at 64K,
prefill fell 3.57% and request time rose 3.02%, despite small ordinary-decode
gains. Short MTP changed tokens and acceptance and ran faster with the Tensor
Core version; the earlier scalar version ran slower. This does not establish
quality parity or a general MTP gain. Full comparisons and the communication
profile are in the maintainer validation record. Rebase/retest this experiment
against later accepted work before enabling it.

## Lending all parallel scratch

`full-scratch-loans.patch` preserves stage3's rejected full-loan layout relative
to the retained selective-loan layout. It additionally lends input/output and
head-state transfer buffers, and restores the former vocabulary-sized wide
output allocation so the original measurements can be reproduced. It returns
roughly 1 GiB per GPU for chunk2048, but regressed 8K prefill 17.5% and short-MTP
prefill 26.3%, with matching tokens. Concurrent pitched transfers from VMM-backed
sources were slower on this PCIe topology (the single-transfer microbenchmark
missed that penalty). The accepted selective variant keeps those transfer buffers
native and corrects the oversized output allocation, recovering about 0.5–0.6 GiB
per GPU at chunk2048 with essentially matched speed. See the validation record
for failures fixed before testing, all measurements, and the explicit user choice
to retain the capacity improvement independently of a speed gain.

## Measured MTP draft costs

`calibrated-mtp-cost.patch` preserves stage4's nonblocking per-batch CUDA-event
cost model, confidence/acceptance yield estimate, and periodic width probes.
Generated tokens match across five workloads, but single-request decode ranges
from -0.81% to +0.73%, and two concurrent requests lose 7.98% decode / 6.60%
request latency. Keep disabled; sparse probes and GPU-only cost observations do
not establish a generally superior end-to-end policy. See the validation record
for acceptance changes and within-run spread.

## Pipeline prototype and alternate expert combine

`tp2-pp2-overlap.patch` preserves the isolated stage5 prototype, including its
interleaved schedule and 18-layer recurrent replay support, against the
post-stage3 source snapshot. Its pre-combine version reached1566 prefill tok/s
at64K with matching tokens, but used a whole-walk Engine submission and extra
primary workspace. It is a historical experiment; the active implementation
instead exposes `--pipeline-parallel` and uses bounded resumable steps. The
combined prototype reached1802 tok/s at64K; the active version's qualification
and its short-prompt/decode costs are recorded in the validation document.

`full-width-owner-combine.patch` preserves stage6's initial all-width reduction
against the wide-only version. It regressed ordinary64K decode0.45% and added
20MiB to the primary sampled peak. The retained route limits partial reduction
to wide calls and reuses the expert-output arena. Both variants change wide
summation order and can change speculative output/acceptance.

`fixed-kv-expert-rebalance.patch` preserves stage7 against the wide-only TP4
snapshot. It fixes both KV heads to GPUs0/1 and moves four experts per layer
from each KV GPU to its corresponding KV-free GPU (124/124/132/132 ownership).
Set its `kRebalanceExperts` constant false for the fixed-KV/even-expert comparison.
It is a separate TP4 layout experiment, not an additional flag for TP2+PP2.
The active `--fixed-kv-owners` option retains the fixed-KV/even-expert layout;
the patch preserves the additional, deferred expert-rebalancing experiment.

## Patch bases

These experiments were measured at different accepted baselines. Later owner-local
attention, selective scratch, weighted combine and pipeline integration change
their call sites; do not apply an old patch blindly to the current tree. Preserve
the earlier negative and positive results when rebasing. Patch inputs are archived
in the task workspace's `tp4-results/candidate-bases/` and the server's matching
results directory; some archives contain only affected input files. The stage5
post-stage3 and stage6 wide-owner-combine archives contain full source/test trees.
Dry-run application passes against the recorded inputs for stages2,3,4,5,6,7.
Saved benchmark binaries and raw model reports remain beside each campaign's
results. Current-code qualification is required before enabling any deferred
candidate, including concurrent raw expert-result writes, whose original patch
predates owner-local weighted combination.
