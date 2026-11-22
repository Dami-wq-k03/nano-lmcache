# QMSum-based Prefix KV Cache Reuse Benchmark

[中文](zh-CN/qmsum-benchmark.md) · [Reproduction guide](../benchmarks/qmsum/README.md)

This experiment asks whether restoring **external L2 KV** reduces repeated
prefill for different questions about complete, unchanged meeting transcripts.
Key membership alone is not a successful restore.

> These are historical Pilot measurements, not a full rerun of current main.
> The PR passes storage explicitly to the timing function to remain compatible
> with the integration cleanup that removed `external.storage`. Recorded
> measurements are unchanged; the exact measurement-era source is archived.

## Fixed design

- Official [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum), audited on
  2026-11-22. Upstream main was `83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e`;
  SHA-256 hashes of all three JSONL files are also recorded.
- Measurement source: `a8652d9` plus the benchmark working tree, with a source hash.
- nano-vLLM: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`; PyTorch 2.6.0+cu124,
  Transformers 4.57.1, FlashAttention 2.7.4.post1.
- TP=1, BF16, eager, 32,768-token context, 64 output tokens reserved; no transcript
  truncation. Use the model chat template with `enable_thinking=False`.
- Compute the LCP of final token IDs across all valid questions, align to 256,
  and never reuse the prompt's last block, even when it is complete.
- Select the 20 longest eligible test meetings, four questions each. This is not
  a length-balanced sample.
- Off / Sync / Async-Flush (C1) / Async-Continuous (C2) use identical inputs,
  ordering and sampling parameters. Compare modes on the **same GPU and model**;
  rotate mode order between rounds.
- Generate 1 / 32 tokens, `ignore_eos=True`, temperature=1e-6, fixed seed;
  arrival concurrency 1 / 4 / 8; three measured rounds and two discarded warmup
  cohorts per mode and concurrency.
- Recreate L1 before every wave. Keep L2 only within a cohort, not across cohorts
  or rounds. Concurrent arrivals use different meetings; the next question wave
  enters after the previous wave returns.

## Dataset audit

| Split | Meetings | Raw query entries | Distinct valid queries | Eligible at 32K |
| --- | ---: | ---: | ---: | ---: |
| train | 162 | 1,257 | 1,256 | 161 |
| val | 35 | 272 | 271 | 35 |
| test | 35 | 281 | 280 | 34 |
| Total | **232** | **1,810** | **1,807** | **230** |

The downloaded files contain 1,810 entries, not the proposal's quoted 1,808.
Each split has one within-meeting duplicate question; there are no empty
questions. Eligible meetings provide **1,794** requests, including **273** from
the eligible test set.

Exclude `train-113` (maximum prompt 35,741 tokens) and `test-016` (33,831) because
full prompt plus reserved output exceeds 32K. Neither transcript was shortened.

| Shared-prefix bucket | All eligible meetings | Eligible test meetings | Pilot |
| --- | ---: | ---: | ---: |
| 1K–4K | 13 | 1 | **0** |
| 4K–8K | 46 | 6 | **0** |
| 8K–16K | 106 | 16 | **9** |
| >=16K | 65 | 11 | **11** |
| Total | 230 | 34 | **20** |

The Pilot freezes **80 requests**:

- Mean shared prefix: **16,988.75 tokens**; maximum: **26,958**.
- Mean cacheable complete prefix: **16,857.6 tokens**; range: **11,520–26,880**.
- Workload SHA-256:
  `a7c325109d8d6836c83696744d9bef61768ffef51cfc89c0e4dc9e2aac92718d`.
- There are **no 1K/4K Pilot samples**. This run cannot establish short-prefix
  benefits or a crossover threshold; do not extrapolate from the long meetings.

## Models and resources

The server reports an RTX 4090 / 48 GiB on GPU 0 and RTX 4090 / 24 GiB on GPU 1.
Independent TP=1 jobs initially ran simultaneously:

| GPU | Model | Role | GPU memory utilization | Rendezvous port |
| --- | --- | --- | ---: | ---: |
| 0 | Qwen3-14B, cached FP8 weights reconstructed as BF16 | Primary | 0.74 | 23410 |
| 1 | Already-cached Qwen3-0.6B, BF16 | Small-model control | 0.35 | 23411 |

The pinned loader does not implement AWQ/FP8 or Qwen3.6's architecture. We did not
pretend that the cached 32B-AWQ, 14B-FP8 or Qwen3.6 models were directly runnable.

The 14B source is `Qwen/Qwen3-14B-FP8` snapshot
`9a283b4a5efbc09ce247e0ae5b02b744739e525a`: multiply FP8 weights by 128×128 block
scales in FP32, cast to BF16, and save separate shards and source-shard hashes.
**The original model cache is unchanged.** This is BF16 inference on reconstructed
quantized weights, not native FP8 inference or the original unquantized BF16
checkpoint. The 0.6B revision is `c1899de289a04d12100db370d81485cdf75e47ca`.

Reserve VRAM for GPU KV, fetched payloads and stack/scatter temporaries. Arrival
concurrency is not a guarantee that all full contexts execute simultaneously.
Jobs share CPU, RAM and PCIe: label these as **co-run conditions**, not an
uncontended machine. Never compare GPU 0's Off with GPU 1's On. Models were not
swapped between GPUs, so cross-model speedup differences cannot be attributed
solely to parameter count.

## Measurement boundaries

1. TTFT ends when the first token is observable after local step/postprocessing,
   including that step's store. It is not HTTP streaming TTFT. Record 32-token
   E2E separately.
2. Count L2 hits only after payload fetch, GPU import and stream completion.
   Assert `prefill_tokens = prompt_tokens - l2_hit_blocks * 256` and zero L1 hits.
3. C1 flushes outside request timing; C2 has no explicit inter-question flush.
   Record pending-key waits and the actual inter-wave bookkeeping/reset gap.
4. Generic async `prepare` completes D2H in the foreground, then a worker writes
   to the pinned CPU backend. GPU export/D2H are not fully hidden. No artificial
   backend sleep was added.
5. Separate store hook, fetch and batch import CPU/CUDA spans. Never duplicate
   batch import time per request. Throughput includes C1 flush and final drain.
6. Report cold/warm samples and P50/P95 separately; compare output IDs for matching
   requests/configurations. Mismatches, missing pairs or incomplete coverage make
   the analyzer return nonzero.
7. A 10% engineering target is not a presumed outcome. Preserve BF16 token
   differences; semantic similarity does not satisfy strict ID equality.

## Measurement-era validation and completeness

- Final local CPU regression: **769 passed, 5 skipped**; Ruff and `mypy src` pass.
  The new QMSum suite has **16 passing tests**, including signed-zero byte checks.
- Measurement-server CPU suite: **801 passed, 1 failed, 2 skipped**. The unchanged
  `test_nested_dictionary_order_does_not_change_namespace` accesses
  `rope_parameters`, absent in Transformers 4.57.1. The test and core source were
  not modified for this experiment; do not report this suite as entirely green.
- Long-prefix single-request smoke: **6/6** Off/On output pairs match per model,
  with actual inference in all four modes.
- Simultaneous eight-arrival stress: **32/32** observations per model; 0.6B matches
  **16/16** output pairs, 14B **15/16**. For `test-030 / specific-0`, Off returns
  first token `The`, C2 `Based`; 77 blocks restore and prefill falls from 19,927
  to 215 tokens. This does **not** establish strict generation equivalence.
- 0.6B: **5,760/5,760** measured observations; **801/4,320** paired sequences
  differ, including 54 first-token differences. The three L2 modes agree on all
  1,440 corresponding outputs, which is not the same as agreeing with Off.
- 14B sequential: **1,920/1,920** observations; **162/1,440** paired sequences
  differ, including 18 first-token differences. Concurrency 4 then hit CUDA
  fragmentation on a 5.35-GiB restore `torch.stack`. Preserve the complete
  sequential group and exclude the 144 partial concurrency-4 observations.
- With the same 0.74 VRAM budget and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the failing cohort passes.
  Fresh concurrency-4/8 groups complete **3,840/3,840** observations. The assembled
  14B matrix has **5,760/5,760**, **531/4,320** differing paired sequences
  (54 first-token differences), and no missing pairs.
- Both models total **11,520 valid measurements**, excluding warmups/diagnostics.
  Both GPUs are released. Failed-run logs, source and recovery conditions remain
  archived. GPU 1 transitions from native-L1 diagnostics to idle during recovery;
  both-device utilization is recorded. Cross-concurrency curves are descriptive,
  **not controlled constant-background-load scaling experiments**.

## Observational performance: strict token-ID acceptance has not passed

These are completed configurations, **not an overall acceptance-success claim**.
TTFT includes only later, warm questions. Each mode has 9 meetings / 81 records
or 11 meetings / 99 records across three rounds, not 81/99 independent meetings.
Times are milliseconds; C2 is Async-Continuous; arrival concurrency and output
length are both 1.

| Model | Shared prefix | Meetings / records per mode | Off P50 / P95 | C2 P50 / P95 | P50 ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B | 8K–16K | 9 / 81 | 232.6 / 264.6 | 113.3 / 178.7 | 2.05× |
| 0.6B | >=16K | 11 / 99 | 466.2 / 777.2 | 161.2 / 227.2 | 2.89× |
| Reconstructed 14B BF16 | 8K–16K | 9 / 81 | 2,840.5 / 3,140.4 | 245.8 / 295.5 | 11.56× |
| Reconstructed 14B BF16 | >=16K | 11 / 99 | 4,740.5 / 6,953.5 | 372.0 / 447.0 | 12.74× |

Request throughput with 32 output tokens, including cold requests and final
drain, median of three rounds (req/s):

| Model | Arrival concurrency | Off | Sync | C1 | C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 1 | 0.914 | 1.053 | 1.000 | 1.005 |
| 0.6B | 4 | 1.412 | 1.832 | 1.659 | 1.679 |
| 0.6B | 8 | 1.427 | 1.835 | 1.684 | 1.679 |
| Reconstructed 14B BF16 | 1 | 0.198 | 0.423 | 0.413 | 0.413 |
| Reconstructed 14B BF16 | 4 | 0.210 | 0.484 | 0.465 | 0.466 |
| Reconstructed 14B BF16 | 8 | 0.210 | 0.480 | 0.467 | 0.463 |

Do not call a 12× warm TTFT ratio a 12× throughput improvement. Generic async
store did not outperform synchronous store: foreground D2H and another host copy
remain. All 14B configurations have only **38,144 GPU KV token slots**, insufficient
for eight independent full meetings. Plots flag failed correctness acceptance;
the combined 14B plots additionally flag mixed run conditions.

For C2 / concurrency 1 / output 1 across three rounds, each model has 240 requests
(180 warm), restores **11,853 blocks** and saves **3,034,368 prefill tokens**.
Restored/reusable prompt-block hit rate is **74.93%** including cold requests,
**99.90%** for warm requests, not an estimate from membership probes.

Measured peak PyTorch allocated VRAM is **18.77 GiB** for 0.6B and **44.06 GiB** for
14B. One-second GPU utilization, memory, power and temperature CSVs are also
archived. Process VRAM and allocator allocated/reserved bytes are different metrics.

## Correctness: native prefix numerics are not necessarily L2 corruption

1. **0.6B, `test-013 / specific-2`:** full prefill returns `They`, native L1 `The`.
   Sync/C2 logits equal native L1 exactly (maximum absolute delta 0). Each L2
   mode's **45 blocks** also pass bytewise exported-GPU / stored-CPU / imported-GPU
   comparisons.
2. **All 20 meetings / 80 sequential 32-token 0.6B requests:** native L1 agrees
   **80/80** with each of Sync/C1/C2, but differs **26/80** from full prefill.
   This is extra correctness evidence, not performance data.
3. **14B, `test-030 / specific-0`:** Sync/C2 logits likewise equal native L1.
   Each mode's **77 blocks** pass both transfer-hop byte checks (dtype, shape and
   uint8 views). In the standalone full prefill, `Based` and `The` BF16 logits
   both equal **41.75**; native L1 gives **42.0 / 41.5**.
4. The pinned sampler still draws exponential noise at temperature=1e-6; it is
   not deterministic `argmax`. Full/chunked/cached paths also consume RNG draws
   differently. There are prefix-path numeric differences and sampling-tie
   effects; do not attribute every mismatch to one cause.

This supports lossless L2 transport/native-L1 equivalence in the checked cases,
**not a retrospective pass of the original strict Off/On ID criterion**. Audit
true argmax, KV transport, numeric tolerance and answer quality separately. The
sampler was not changed halfway through this run.

A CPU argmax audit of saved logits chooses `The` (785) on all four 0.6B paths.
For 14B it chooses `The` (785) after full prefill, `Based` (28715) for native L1
and both L2 modes. True argmax removes sampling noise but **does not guarantee
BF16 full/prefix generation identity**. This is an audit of saved logits, not a
new GPU performance run.

`benchmarks/qmsum/diagnose_qmsum.py` saves logits, Top-5 and byte comparisons.
It explicitly drains async writes to inspect CPU snapshots; its timing is not a
C2 wait-time measurement and never enters performance results.

## Acceptance and reproducibility

| Item | Outcome |
| --- | --- |
| Exact Off/On generated token IDs | **Failed:** 0.6B 801/4,320; 14B 531/4,320 paired sequences differ |
| Actual L2 restore / saved prefill | Per-request accounting asserted; additional byte and native-L1 diagnostics above |
| At least one bucket improves warm TTFT by >10% | Observationally satisfied; does not override failed correctness acceptance |
| Reproducibility | Three rounds complete with input/source/model/environment/recovery metadata; no cross-machine stability claim |

Measurement source SHA-256:
`80f5ea49149512041d74f6d8cc42a765a7e903bbcd9f21a0a36b744009138b90`.
The exact snapshot is archived as `source-v1.tar.gz`. Later code adds metadata,
diagnostics, explicit-storage compatibility and qualified figures without
rewriting measured inputs, timings or outputs. Reports separately record
`analysis_code_sha256`.

Main artifact directories (kept outside Git):

- `pilot-0.6b/`: complete small-model raw results, summary, CSV, light/dark figures.
- `pilot-14b-combined/`: complete sequential and rerun concurrent groups;
  `environment.json` records source-file hashes and environments, not a fictional
  uninterrupted constant-condition invocation.
- `pilot-14b/`, `pilot-14b-c1/`, `pilot-14b-c4c8-retry/`: failed original, valid
  sequential subset, and complete retry raw observations, all retained.
- `recovery.json`, run logs, GPU CSVs and `diagnostic-*/`: recovery and correctness
  evidence. Do not commit raw meetings, weights or large/private outputs.

This completes the **20-meeting Pilot**, not the formal 34-meeting / 273-query
experiment, five rounds, 64 output tokens, disk backend or same-meeting stampede.
Resolve strict generation versus numeric-path acceptance before expanding.
