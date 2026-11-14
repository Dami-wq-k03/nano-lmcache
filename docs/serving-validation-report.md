# Serving validation follow-up (2026-11-14)

This follow-up validates real external-hit generation, asynchronous GPU export,
cold NVMe reads and a two-GPU model KV handoff. Correctness does not imply a
serving speedup. Raw samples are in [measurements/serving](measurements/serving).

## Environment and method

Linux 6.8 / Ubuntu 22.04, vLLM 0.33.0, torch 2.14.0+cu130 / CUDA 13.0,
Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, BF16.
Serving uses the idle 24 GiB RTX 4090, TP=1, eager, sequential single requests,
16-token blocks and 0.4 GPU memory utilization. Another GPU has an existing
service; clocks are not fixed. Engine starts and frequency drift can affect
comparisons, so every variant retains its own recompute baseline.

The serving benchmark uses 257 prompt tokens, 16 greedy output tokens, two
warmups, five repeat hits and five distinct-prefix store requests. APC is
disabled in both recompute and external engines. Every hit must report 256
cached tokens, perform 16 successful remote **payload GETs**, and generate
exactly the same output IDs as recomputation. Scheduler presence checks alone
are insufficient evidence. The cold/store phase uses new first-token prefixes.

TTFT is engine `first_token_latency`; TPOT is the first-to-last output-token
interval divided by 15. Wall time covers `generate()`. A separate publication
wait after the request ensures pending writes are visible before the next
request; that wait is excluded from wall time and must be considered when
reasoning about immediately following consumers. Raw samples retain all of it.

## Implementation

`AsyncTensorWriteStorage` owns GPU snapshots, copies D2H through an independent
stream and bounded pinned pool, waits on events in a worker and publishes owned
CPU values. Bounded queues apply backpressure; shutdown drains writes. GPU
gather/snapshot work remains. The connector skips export when a request has no
newly complete chunk; otherwise a partial tail would re-save its entire loaded
prefix. Remote storage now accepts a persistent packed byte backend.

## Real vLLM memory-backend results

Measured revision `9bfb38a`; medians over five requests per row (ms).
All output tokens match. Every repeat loads 256 tokens / 16 payloads.
Warmup publication is explicitly drained before measured requests.

| Mode | Phase | Recompute TTFT / TPOT / wall | External TTFT / TPOT / wall |
| --- | --- | ---: | ---: |
| sync | repeat | 32.28 / 14.61 / 251.42 | 1000.85 / 26.04 / 1368.23 |
| sync | store | 32.22 / 14.56 / 251.37 | 967.38 / 24.42 / 1333.99 |
| async | repeat | 31.76 / 13.82 / 239.79 | 962.07 / 25.68 / 1345.82 |
| async | store | 31.70 / 13.87 / 239.95 | 94.57 / 56.15 / 933.31 |

Async store improves first-token and request wall latency relative to synchronous
store in this run, but substantially worsens decode TPOT. Median publication
waits after request completion are under 0.01 ms: most publication finishes
during decode, where resource contention remains visible. This does not prove
isolated D2H overlap is the sole cause; clocks and separate engine starts are
confounders. Both external-hit variants are slower than their own recompute
controls. The **TPOT regression <5% gate fails**. No general serving speedup is
established. Larger models, longer prefixes, batching and a faster
serialization/transport path need separate tests.

## Reproduction

Use the pinned vLLM image/environment and a locally cached model (`MODEL`).
Run from the checkout with `PYTHONPATH=src`, offline model loading enabled and
`NANO_LMCACHE_GIT_SHA` set to the measured source revision:

```bash
python3 benchmarks/bench_vllm_external.py --model "$MODEL" --json sync.json
python3 benchmarks/bench_vllm_external.py --model "$MODEL" --async-store --json async.json
python3 benchmarks/bench_vllm_external.py --model "$MODEL" --async-store \
  --disk-directory /work --cold-disk --json nvme-serving.json
python3 benchmarks/bench_disk_pages.py --directory . --pages 64 \
  --page-bytes 1048576 --repeats 30 --cache-state cold
python3 benchmarks/bench_disk_pages.py --directory . --pages 64 \
  --page-bytes 1048576 --repeats 30 --cache-state warm
python3 benchmarks/bench_pd_handoff.py --model "$MODEL" --warmup 1 --repeats 5 --json pd.json
python3 benchmarks/bench_vllm_external.py --model "$MODEL" --prompt-tokens 1025 \
  --async-store --with-apc-control --json apc-control.json
```

## Longer-prefix local APC control

The optional separate APC engine uses the same model and output checks with
1025 prompt tokens. Five repeat requests per mode; medians (ms):

| Mode | TTFT | TPOT | Wall | Cached tokens | Remote payload GETs |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 32.93 | 14.42 | 249.90 | 0 | 0 |
| local_apc | 34.10 | 14.25 | 248.11 | 1024 | 0 |
| external | 3612.14 | 25.36 | 3992.00 | 1024 | 64 |

All three engines generate identical output IDs. APC is enabled only in its
separate control engine. External hits load 1024 tokens / 64 actual payloads;
local APC performs no remote payload reads. Increasing this small model's
prefix to 1025 tokens does not rescue the external path. JSON/base64 framing,
serialization, per-chunk RPC and GPU restoration are candidate costs; this
benchmark does not isolate their individual contributions. This is a
measured baseline, not a conclusion about larger models or optimized transports.

## Cold NVMe → external KV → generation

The same vLLM benchmark uses a packed disk backend with per-file DONTNEED
before requests. All five repeats load 256 tokens / 16 payloads, produce exact
baseline output IDs and record **29,388,800 physical read bytes each** in the
server process. The serving adapter fetches individual chunks; it does not yet
use the component benchmark's `get_many()` coalescing path.

| Phase | Recompute TTFT / TPOT / wall ms | External TTFT / TPOT / wall ms | External publication wait ms |
| --- | ---: | ---: | ---: |
| Repeat | 31.95 / 13.92 / 243.31 | 943.21 / 24.14 / 1305.27 | 0.004 |
| Novel store | 31.90 / 13.94 / 241.03 | 110.12 / 39.47 / 719.39 | 553.93 |

The substantial post-request store publication wait is essential: 719 ms request
wall time is not a durable-write completion latency. Medians of individual
columns should not be added to infer a median total. This establishes cold-disk
correctness, not a latency win over recomputation. The warmup race found during
validation is fixed in `9bfb38a`: drain initial warmup publication before counters
for measured requests begin. Failed preliminary samples are excluded.

## Two-GPU real-model handoff

The single-process HF prototype loads two Qwen3-0.6B instances, prefills 128
tokens on GPU 0, packs/transfers/restores 14,680,064 bytes (14 MiB) onto GPU 1,
then generates 16 greedy tokens. All five measured runs match the colocated
target-model output IDs exactly. One warmup is excluded.

| Stage | Median ms |
| --- | ---: |
| Source prefill | 28.91 |
| Pack | 0.447 |
| Transfer | 1.879 |
| Restore | 1.073 |
| First target decode step | 26.103 |

The actual transport is **pinned_host**: the RTX 4090 pair has no usable peer
access. These numbers include host staging and do not establish P2P or RDMA
bandwidth. GPU 0 shares capacity with an existing service; the prototype checks
free VRAM and caps its allocator with a margin. No existing service was stopped.
There is no engine RPC, two-engine serving scheduler, transfer/prefill overlap
or throughput/TTFT advantage claim. `queue_ms=0` is a prototype constant, not a
measured queue distribution. The first output token comes from source prefill;
`first_decode_ms` is the subsequent target decode step.

## NVMe component measurements

Samsung SSD 990 PRO 1TB; 64 × 1 MiB payloads, 30 samples per mode.
Per-file `POSIX_FADV_DONTNEED` is advisory; `/proc/self/io` independently records
67,108,864 physical bytes per packed cold sample and 67,371,008 for entry files
(including headers). Every warm sample records zero physical bytes.

| Mode | Cold median / p95 ms | Warm median / p95 ms | Payload reads |
| --- | ---: | ---: | ---: |
| Entry files | 113.85 / 165.43 | 76.15 / 79.41 | 64 |
| Packed per-page | 64.57 / 85.80 | 55.55 / 71.59 | 64 |
| Packed batch | 64.44 / 75.75 | 45.73 / 52.46 | 8 |
| Threaded prefetch | 110.57 / 123.59 | 46.54 / 63.37 | 8 |

Packed batching is 1.77× faster than entry files in this cold component test;
threading does not improve that result. These are buffered host reads, including
Python overhead, not device bandwidth or GPU-serving throughput. No direct I/O,
registered io_uring, GDS or global cache drop is used.

## Validation and remaining gates

Local full suite: **756 passed, 79 skipped**; Ruff and strict mypy pass.
Pinned real-vLLM/CUDA async-writer and connector tests: **84 passed**.
The raw benchmark output is additional real-model validation.

Two-engine nano-vLLM PD scheduling, transfer/prefill overlap, NIXL/RDMA, GDS,
registered io_uring, concurrent serving and broad performance acceptance remain
open. The following measurements apply only to this model and workload.
