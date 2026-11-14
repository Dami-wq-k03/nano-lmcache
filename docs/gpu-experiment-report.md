# GPU and data-plane experiment report

This report records the first implementation and measurements from the research
roadmap. Results are measurements of this checkout and this machine, not
portable performance guarantees.

## Current checkout validation (2026-11-14)

Local checks: full suite **748 passed, 75 skipped**; CPU tensor group
**208 passed, 3 skipped**; Ruff and strict mypy pass. `uv lock --check` and
wheel/sdist builds pass. Optional engine and CUDA tests skip locally.

This section supersedes the implementation-status claims in the historical
first-pass measurements below. The earlier model/restart numbers have not been
rerun as part of this validation.

| Component | Evidence and remaining gate |
| --- | --- |
| Page ABI, replicas, cost model, policies, prefetch | CPU tests; policy latency is modelled, not serving latency |
| CUDA/Triton gather/scatter and INT8 | GPU correctness and matched benchmark; explicit opt-in |
| Mapped-host pack/quantize | GPU correctness and A/B against pack/quantize plus DMA; no universal speedup |
| Pinned pool and async transfers | Producer-stream ordering and 10k pool reuse test; bounded pool and double-release checks |
| Two-GPU remap/restore | Both directions pass correctness tests; no dedicated peer-bandwidth acceptance claim |
| Packed disk, host prefetch, io_uring | CPU corruption/recovery/concurrency tests; io_uring is optional and requires Linux permissions; no cold-NVMe result |
| vLLM V4 | Shared-store load adapter tested on CPU and CUDA; real vLLM 0.33 external-hit generation remains open |
| Async store | Background tier writes only; GPU gather/export remains on the request path; no TPOT <5% claim |

The closeout also fixes write-behind shutdown after backend errors, serializes
submission against shutdown, preserves an explicitly supplied empty remote
backend (and its capacity), and ensures engine cleanup runs after close errors.

New remote validation on PyTorch 2.6.0+cu124 / Triton 3.2.0:

```text
Single visible GPU:
  tests/tensor/test_kernels_codecs.py tests/tensor/test_mapped_host.py
  tests/tensor/test_transfer.py tests/tensor/test_peer_pool.py
  tests/unit/test_vllm_kv_load.py tests/unit/test_coalesced_disk.py
  99 passed, 2 skipped (two-GPU cases)
Two visible GPUs:
  tests/tensor/test_peer_pool.py
  4 passed
```

The separate native-reader check completed **10 passed, 1 skipped** on Linux.
The C library compiled, but the host rejected `io_uring_setup` with EPERM;
registered reads and short-read handling therefore remain unverified on hardware.
The new test runs those checks automatically where kernel permissions allow.

The CUDA extension requires a CUDA toolkit, C++ compiler and Ninja on PATH.
Triton kernels additionally require a compatible Triton installation. Install
these in the optional CUDA environment; the core installation stays dependency
free. For the tested torch 2.6 environment, Triton was 3.2.0. For example,
activate that environment, install `ninja` if absent, and expose `$CUDA_HOME/bin`.
The first extension call compiles the packaged `page_kernel.cu`; compilation is
excluded from the timed samples. Wheel and sdist builds were checked to retain
the native `.cu` and `.c` sources.

### Matched kernel and fusion measurements

Raw results, commands, environment, source digest, p50/p95/p99, and host-call
latency are retained in [kernel JSON](measurements/rtx4090-kernels.json) and
[fusion JSON](measurements/rtx4090-fusion.json). The source marker is `ac20685`;
the SHA256 identifies the transferred snapshot. Later closeout changes affect
remote backend selection, tests and docs, not the GPU implementations.

Reproduce from the repository root in the CUDA environment:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_gpu_kernels.py \
  --pages 16 256 --tokens 16 64 --warmup 5 --repeats 30 \
  --json /tmp/kernels.json
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_gpu_fusion.py \
  --warmup 5 --repeats 30 --json /tmp/fusion.json
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/tensor/test_peer_pool.py
```

All comparisons below are CUDA-event medians in milliseconds with five warmups
and 30 samples. Page data has shape `[pages, 2, 2, 4, tokens, 64]`, FP16, and
random physical ordering. Destination buffers are preallocated. Event intervals
include host submission gaps; these are adapter-path timings, not isolated
kernel execution times. The Python loop also copies page IDs to the host.
The other GPU was not isolated; clocks were not fixed. Repeated data can reside
in L2, so effective read+write GB/s must not be interpreted as HBM bandwidth.

| Operation | Python loop | Batched torch | Triton | CUDA |
| --- | ---: | ---: | ---: | ---: |
| Gather, 256 pages x 64 tokens | 2.4988 | 0.0820 | 0.0655 | 0.0297 |
| Scatter, 256 pages x 64 tokens | 2.5154 | 0.0427 | 0.0658 | 0.0297 |
| Gather, 16 pages x 64 tokens | 0.1808 | 0.0145 | 0.0612 | 0.0204 |

CUDA improves the larger gather/scatter cases, but small gathers favor torch.
Triton has substantial host overhead in this environment. FP16 INT8 encode
improves from 0.1157 ms (torch) to 0.0681 ms (Triton), while decode regresses
from 0.0343 to 0.0594 ms. These results do not justify replacing every torch
operation with a custom kernel.

| Export operation | Torch + DMA | CUDA/Triton + DMA | Mapped-host CUDA |
| --- | ---: | ---: | ---: |
| Pack, 256 pages / 8 MiB | 0.3500 | 0.3402 | 0.3592 |
| INT8, 16 tokens | 0.1889 | 0.1330 | 0.0814 |
| INT8, 64 tokens | 0.3012 | 0.2487 | 0.8913 |
| INT8, 256 tokens | 1.2941 | 0.7350 | 3.6726 |

Mapped-host INT8 wins for the 16-token case but loses badly at longer token
lengths. It writes over PCIe from GPU threads rather than using a DMA engine;
it remains an experimental option. FP8/INT4 quality is covered by synthetic
error bounds only, not a model-task quality gate.

The 32 MiB copy + 4096-square GEMM microbenchmark reports 2.4559 ms serial
versus 1.6061 ms overlapping, with standalone copy/GEMM at 1.3718/0.8799 ms.
This demonstrates overlap in this microbenchmark, not a serving TPOT result.

## Historical first-pass scope

The first pass implemented and tested the following pieces:

- A01--A04 foundation: `KVLayout`, `KVPageRef`, generation-aware logical page
  directory, and a dependency-free reference page store.
- A05/G01--G05 baseline: environment-complete GPU copy benchmark, pinned
  buffer pool, asynchronous CUDA stream/event transfers, and page lifecycle
  publication after transfer completion.
- G06--G08 reference path: validated torch gather/scatter page operations for
  non-contiguous and repeated page IDs. These are a PyTorch reference baseline,
  not custom CUDA or Triton kernels.
- S01/S02: transparent load-versus-recompute cost model and duplicate-safe
  prefetch state machine.
- Q01: per-head/channel-axis symmetric INT8 reference codec, used by the
  existing safe tensor serializer.
- M12/N02--N03 baseline: length-prefixed JSON shared cache server, batch lookup,
  and bounded write-behind storage with backpressure.
- N5 prototype: `demo_nanovllm.py --async-store` moves tier writes to a bounded
  worker and flushes on shutdown.
- O01: existing trace schema plus transfer lifecycle events and nano-vLLM trace
  rendering.

At that first-pass snapshot, the vLLM V4 external-load path, custom GPU kernels,
NVMe direct I/O, P2P/PD transport, GDS, RDMA/NIXL, and CacheBlend were open. They need a
separate vLLM 0.33 environment, custom hardware, or a larger implementation
than this first incremental milestone.

## Hardware and software

The GPU server was described in the private, ignored `GPUserver.md` file. The
experiment used the following environment:

| Item | Value |
| --- | --- |
| Host GPU | NVIDIA GeForce RTX 4090, two physical GPUs |
| Experiment GPU | GPU 1, exposed as `cuda:0` with `CUDA_VISIBLE_DEVICES=1` |
| GPU memory | 23 GiB visible to the experiment |
| Driver | 580.178.04 |
| CUDA runtime | 12.4 for the nano-lmcache environment |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11.16 |
| Model | local Qwen3-0.6B snapshot |
| Source marker | `f761bec` plus the uncommitted changes described in this report |

The nano-vLLM demo used its existing separate environment, which has
PyTorch 2.5.1+cu121 and the installed nano-vLLM/FlashAttention dependencies.
The nano-lmcache test environment does not have `uv` on the server, so the
commands below use its absolute virtual-environment paths.

For reproduction, set `MODEL_DIR` to the local Qwen3-0.6B directory,
`NANO_VENV` to the nano-lmcache CUDA environment, and `NANOVLLM_VENV` to the
separate nano-vLLM environment. The examples use an isolated
`/tmp/nano-lmcache-gpu-work` source snapshot.

## CPU and CUDA correctness gates

Local CPU validation after the implementation:

```text
full suite: 600+ passed, optional GPU/engine tests skipped when unavailable
ruff check: passed
mypy src/nano_lmcache: passed
```

The remote non-engine GPU suite passed:

```text
33 passed, 2 skipped, 597 deselected
```

The new CUDA transfer and page-operation tests passed on the remote GPU:

```text
3 passed
```

The page-aware transfer test verifies that a `KVPageRef` remains non-ready
until its CUDA event has completed. `complete_page()` does not use a
device-wide synchronization.

## D2H/H2D copy sweep

Command:

```bash
NANO_LMCACHE_GIT_SHA=f761bec \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/nano-lmcache-gpu-work/src \
$NANO_VENV/bin/python \
/tmp/nano-lmcache-gpu-work/benchmarks/bench_gpu_dataplane.py \
  --device cuda:0 --sizes 4K 1M 32M 256M \
  --page-count 1024 --warmup 3 --repeats 10 \
  --json /tmp/nano-lmcache-gpu-dataplane-final.json
```

The table reports median CUDA-event time over ten measured samples. `call ms`
is the host-side `copy_` call duration, not the device copy duration.

| Payload | Mode | D2H GB/s | H2D GB/s | call ms (D2H) |
| ---: | --- | ---: | ---: | ---: |
| 1 MiB | pageable blocking | 3.53 | 3.28 | 0.275 |
| 1 MiB | pinned blocking | 10.00 | 10.35 | 0.093 |
| 1 MiB | pinned non-blocking | 22.57 | 21.22 | 0.007 |
| 32 MiB | pageable blocking | 7.64 | 9.39 | 4.368 |
| 32 MiB | pinned blocking | 25.60 | 24.20 | 1.299 |
| 32 MiB | pinned non-blocking | 25.14 | 24.08 | 0.008 |
| 256 MiB | pageable blocking | 7.81 | 9.85 | 34.338 |
| 256 MiB | pinned blocking | 22.38 | 23.54 | 11.983 |
| 256 MiB | pinned non-blocking | 21.00 | 23.09 | 0.006 |

For 4 KiB, fixed launch and event overhead dominates: pageable, pinned, and
pinned non-blocking D2H throughput was 0.158, 0.182, and 0.214 GB/s. The
important result is not the small-transfer bandwidth; it is that pinned
non-blocking copies reduce the Python call from roughly 12--34 ms to about
0.006--0.008 ms for the large payloads while the copy remains observable on the
CUDA stream.

The result JSON also records the driver, CUDA runtime, PyTorch version, GPU,
source marker, warm-up count, and repeat count. The JSON is intentionally not
committed as a repository artifact because it is machine-specific.

## Page gather/scatter reference baseline

The same run used 1,024 pages with shape
`[2, 2, 4, 16, 16]` per page and reversed page IDs, so the operation exercises
non-contiguous physical ordering rather than a contiguous copy.

| Operation | Bytes in reference operation | Median ms | Effective GB/s |
| --- | ---: | ---: | ---: |
| gather | 16 MiB | 0.0686 | 244.54 |
| scatter | 8 MiB | 0.0748 | 112.22 |

These numbers are the baseline for a future fused kernel. They should not be
compared with a custom CUDA kernel until the same page table, dtype, and byte
accounting are used.

## Real Hugging Face KV reuse

The existing real-model demo was run twice on the same disk directory, using
GPU1 and the local Qwen3-0.6B weights:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/nano-lmcache-gpu-work/src \
$NANO_VENV/bin/python \
/tmp/nano-lmcache-gpu-work/examples/demo_kv_reuse.py \
  --model $MODEL_DIR \
  --device cuda:0 --offload --disk /tmp/nano-lmcache-kv-exp-1
```

First process, empty disk tier:

```text
Prompt A: 268 tokens; reused 0; computed 268; cache time 1563.4 ms
Prompt B: 267 tokens; reused 256; computed 11; cache time 88.4 ms
Prompt B max |logit diff|: 6.82e-05
KV held in CPU memory: 56.0 MiB
Chunks served: 16 from memory, 0 from disk
```

Second process, fresh GPU/CPU cache and populated disk tier:

```text
Prompt A: 268 tokens; reused 256; computed 12; cache time 140.3 ms
Prompt B: 267 tokens; reused 256; computed 11; cache time 29.3 ms
Prompt B max |logit diff|: 6.82e-05
Cache metrics: hit rate 100.0%
Chunks served: 16 from memory, 16 from disk
```

The result demonstrates correctness and restart recovery. It does not claim
that disk loading always wins: the prompt is short and the model is small, so
recomputation can be faster than restoring 256 tokens.

## INT8 KV quality and storage

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/nano-lmcache-gpu-work/src \
$NANO_VENV/bin/python \
/tmp/nano-lmcache-gpu-work/benchmarks/bench_quantized_kv.py \
  --model $MODEL_DIR \
  --device cuda:0 --warmup 2 --repeats 5 --threads 1
```

The synthetic chunk was `[28, 2, 8, 16, 128]` and occupied 1,835,008 raw
bytes. INT8 serialization occupied 1,148,376 bytes, a 1.598x raw-to-encoded
ratio including FP32 scales.

| dtype | Encode ms | Decode ms | Relative L2 | Model next-token match |
| --- | ---: | ---: | ---: | --- |
| float16 | 1.331 | 3.066 | 0.00470 | yes |
| bfloat16 | 1.753 | 2.224 | 0.00496 | yes |

For the real Qwen prompt, the INT8 path matched the greedy token at all 11
computed positions for both dtypes. Relative logit error was 0.0104 for
float16 and 0.0195 for bfloat16. The lossless control also preserved exact KV
bytes; its model output differences were caused by the model execution path,
not serializer corruption. INT8 remains opt-in and lossy.

## nano-vLLM external L2 and restart recovery

The existing nano-vLLM environment was used because the main environment does
not install nano-vLLM. The demo was run twice with the same disk directory:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/nano-lmcache-gpu-work/src \
$NANOVLLM_VENV/bin/python \
/tmp/nano-lmcache-gpu-work/examples/demo_nanovllm.py \
  --model $MODEL_DIR \
  --dir /tmp/nano-lmcache-nanovllm-exp
```

First process:

```text
question-1: 0 L1 blocks, 0 L2 blocks, 1306 tokens computed, TTFT 1938.0 ms
question-2: 5 L1 blocks, 0 L2 blocks, 25 tokens computed, TTFT 41.6 ms
L2 now holds 5 blocks on disk
```

Second process:

```text
question-1: 0 L1 blocks, 5 L2 blocks, 26 tokens computed, TTFT 215.9 ms
question-2: 5 L1 blocks, 0 L2 blocks, 25 tokens computed, TTFT 25.5 ms
L2 now holds 5 blocks on disk
```

Both processes generated the same visible answer text. Metadata-only traces
were written for both runs and the second trace rendered successfully by the
offline viewer: 4 requests and 72 events.

The first process includes one-time engine compilation and should not be used
as a direct cache-speed comparison with the second process.

### Async write-behind prototype

The same two-process experiment was repeated with:

```text
--async-store --write-queue 32
```

The first process flushed five blocks at shutdown; the second process restored
five L2 blocks. Measured TTFTs were 26.9/26.3 ms for the first process and
221.2/25.0 ms for the second process. These runs had different compilation and
system-cache state, so they validate lifecycle and restart correctness, not a
causal speedup. The useful invariant is that pending writes are visible as
plans but `get()` waits for their commit before returning a payload.

## Shared cache and batching

The standard-library server uses bounded, length-prefixed JSON frames and
base64 serializer bytes. It was measured locally on the development Mac with
64 entries and 20 repetitions:

| Lookup strategy | Median time |
| --- | ---: |
| 64 individual requests | 6.334 ms |
| one batch request | 0.178 ms |

The batch path was 35.5x faster in this local loopback test. This is a control
plane/RTT result, not a GPU or network bandwidth result. The server's CPU test
also verifies two independent clients, prefix-safe parent admission, malformed
failure behavior, and `None` miss semantics.

## Historical first-pass reproduction and limitations

The source snapshot was transferred to an isolated `/tmp/nano-lmcache-gpu-work`
directory on the GPU host; the private project checkout was not overwritten.
Model weights and experiment directories remained outside the repository.

Not run in the historical first-pass experiments (see current validation above):

- vLLM 0.33.0 external load/reuse (the GPU host has no vLLM 0.33 environment or
  pytest in the separate nano-vLLM environment).
- Custom CUDA/Triton fused gather/scatter kernels; current measurements are
  the torch reference baseline.
- NVMe io_uring/GDS, P2P bandwidth, PD transfer, NIXL/RDMA, or multi-node
  experiments. GPU0 was not a clean target after an initial OOM attempt left a
  driver-visible CUDA context, so only GPU1 was isolated for these runs; no
  two-engine performance claim is made.
- A long-context 8K/32K serving matrix. The supplied Qwen3-0.6B model and
  short educational prompts were used for correctness and restart tests.

The current closeout adds the kernel/fusion comparisons above. A pinned real
vLLM 0.33 environment is still required before reporting V4 generation results.
