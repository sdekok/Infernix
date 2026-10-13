# Performance measurements

The serving results below use one NVIDIA GeForce RTX 5090 through Infernix's public HTTP route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.
The [experimental four-GPU Flash-Next results](#experimental-four-gpu-flash-next)
use a separate Engine-benchmark method and are labeled accordingly.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to the relevant result section. “Not published” describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8-Flash-Next has no row: its published figures are the README's
[comparison with Strata](../README.md#qwen38-flash-next-nvfp4-infernix-vs-strata), measured with that
section's client-side method rather than this page's, and its own guide's
[performance notes](qwen3_8-flash-next.md#performance).

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. Qwen3.8 measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 completion outcomes](performance/qwen3.8-27b.md#completion-outcomes).

## Experimental four-GPU Flash-Next

The separate [four-5060-Ti validation record](maintainer/multi-gpu-5060-ti-design.md#systematic-optimization-campaign-authorized-2026-10-12)
covers the Dense8/NVFP4 artifact through the public Engine benchmark, rather than
the HTTP method above. With 65,536 prompt tokens, 256 ordinary decode tokens,
chunk 2048, 131,072-token INT8 KV capacity, two warmups and three measured repetitions:

| Layout | Prefill tok/s (mean ± SD) | Ordinary decode tok/s (mean ± SD) | Request time |
|---|---:|---:|---:|
| TP4, rotating KV | 1215.62 ± 0.12 | 35.44 ± 0.03 | 61.24 s |
| TP4, fixed KV owners | 1214.44 ± 0.10 | 37.95 ± 0.03 | 60.82 s |
| TP2+PP2 | 1798.95 ± 0.17 | 34.91 ± 0.05 | 43.86 s |

All three use the same CPU offload split and produce identical tokens in this comparison.
Fixed KV owners improve ordinary decode 7.07% but lower prefill 0.10%, while
concentrating KV pressure on the first two GPUs.
Pipeline improves prefill 47.99% while reducing ordinary decode 1.51%; it also costs
primary workspace and can regress short prompts. MTP acceptance and output can
change with reduction/weight partitioning, so speculative rates are workload-specific.
Cold capacity probes also completed 131,000 prompt tokens plus 33 outputs with MTP3
and matching tokens in all three layouts. Their first 32 decode steps ran at
23.37 tok/s (rotating TP4), 24.30 (fixed KV), and 17.81 (pipeline); pipeline's cold
decode was 23.8% slower than rotating TP4 despite a 26.7% shorter whole request.
These single cold probes are separate from the warmed measurements above.

These measurements do not establish quality parity or a matched comparison with
Strata IQ3_S. Basic chat serving passes prefix reuse, two concurrent lanes and cancellation
recovery in all layouts. Pipeline cached requests took 2.7–3.1 s versus rotating
TP4’s 1.8–2.0 s in the functional probe; these are observations, not repeated
serving benchmarks. The validation record retains all campaign variants, adverse
results, capacity/serving qualification and reproduction details.

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with the README's [benchmark summary](../README.md#benchmarks).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
