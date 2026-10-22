# When does nano-vLLM's L2 cache pay off?

Roadmap task N4. This page measures time to first token (TTFT) with
nano-lmcache as nano-vLLM's external (L2) KV cache, explains what limits
each setup, and says where L2 starts to beat recomputing. The design is in
[nanovllm-integration.md](nanovllm-integration.md).

**Short answer, for Qwen3-0.6B on one RTX 4090:**

- **L2 in CPU memory** is at least 10% faster than recomputing from
  **16 blocks (4,096 reused tokens)** on, and from there matches nano-vLLM's
  own GPU prefix cache (L1): at 8,192 reused tokens, 52 ms instead of 125 ms.
  Below that, every setup takes about the same time.
- **L2 on disk** is never faster than recomputing for this model, up to
  8,192 tokens. Its value here is reuse across processes and restarts, not
  latency.
- **Storing is not free**: version 1 stores synchronously, so the first
  request with a disk-backed L2 waits for its blocks to be written, about
  270 to 330 ms per block.

## Setup

| | |
| --- | --- |
| Model | Qwen3-0.6B, bfloat16; one 256-token block is 28 MiB |
| GPU | NVIDIA GeForce RTX 4090 (24 GB), PCIe |
| CPU, memory | Intel Xeon Platinum 8331C, 235 GB |
| Disk tier | `DiskKVCache` with `TensorSerializer` on a container overlay filesystem (~0.22 GiB/s cold reads) |
| Software | torch 2.6.0+cu124, flash-attn 2.7.4.post1, nano-vLLM `bb823b3`, `enforce_eager=True` |
| Prompts | random token IDs: `blocks` × 256 reusable tokens plus a 32-token tail |
| Statistic | median of 5 rounds; each round runs every setup once (interleaved) |

Reproduce with:

```bash
uv sync --extra hf --group nanovllm
CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/bench_nanovllm.py \
    --model /path/to/Qwen3-0.6B --json results.json
```

The raw results are in
[`benchmarks/nanovllm-qwen3-0.6b-rtx4090.json`](benchmarks/nanovllm-qwen3-0.6b-rtx4090.json).

The setups, all for the same prompt:

- **no cache**: plain nano-vLLM with an empty L1 (`detach`); the whole
  prompt is computed.
- **L1**: nano-vLLM's GPU prefix cache already holds the blocks.
- **L2 memory**: L1 is empty; blocks come from `CPUOffloadKVCache`
  (pinned CPU memory).
- **L2 disk**: L1 is empty; `DiskKVCache` is reopened (as after a restart)
  and its files are dropped from the OS page cache, so they are really
  read from disk. "page cache" keeps them cached, as right after writing.
- **store**: the first request with each L2 backend attached: it computes
  the whole prompt and stores its blocks before returning.

## Results

Time to first token, in milliseconds:

| blocks | prompt tokens | no cache | L1 | L2 memory | L2 disk | L2 disk (page cache) | store, memory | store, disk |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 288 | 54.5 | 52.5 | 44.3 | 228.0 | 116.7 | 73.7 | 301.5 |
| 2 | 544 | 54.9 | 53.4 | 44.7 | 397.6 | 100.3 | 67.3 | 459.7 |
| 4 | 1,056 | 50.5 | 49.6 | 45.8 | 642.7 | 125.0 | 88.6 | 1,130.2 |
| 8 | 2,080 | 53.1 | 45.7 | 50.5 | 1,331.6 | 261.9 | 116.5 | 2,170.9 |
| 16 | 4,128 | 66.2 | 55.4 | 47.6 | 2,668.4 | 534.3 | 203.0 | 4,345.0 |
| 24 | 6,176 | 85.1 | 47.2 | 56.0 | 4,042.9 | 935.6 | 221.4 | 6,261.6 |
| 32 | 8,224 | 125.0 | 49.8 | 52.4 | 7,231.3 | 563.8 | 289.6 | 10,720.6 |

Per reused block, in milliseconds. Loading is timed on its own (fetch
from L2, then copy into the GPU KV cache); recomputing is derived as
`(no cache - L1) / blocks`:

| blocks | recompute | load from memory | load from disk | load from disk (page cache) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.97 | 2.12 | 155.13 | 46.49 |
| 2 | 0.74 | 1.63 | 164.57 | 43.57 |
| 4 | 0.21 | 1.32 | 145.72 | 25.79 |
| 8 | 0.92 | 1.31 | 150.50 | 17.01 |
| 16 | 0.67 | 1.30 | 152.63 | 20.54 |
| 24 | 1.58 | 1.29 | 144.16 | 17.08 |
| 32 | 2.35 | 1.29 | 221.08 | 16.37 |

## What the numbers say

### A prefill step has a ~45 to 55 ms floor

"No cache" barely moves from 288 to 2,080 tokens, and L1 stays at the same
level whatever the size. Timing `ModelRunner.run` on its own shows why: a
prefill step takes about 44 ms whether it computes 32 tokens or 4,128.
nano-vLLM always runs prefill eagerly (its CUDA graphs only cover decode),
so every step pays Python and kernel-launch overhead for 28 layers, and
the GPU finishes each kernel faster than the CPU can launch the next one.
Computing more tokens only costs more once the GPU work outgrows that
floor, here somewhere between 2,000 and 4,000 tokens.

Under the floor, skipping computation saves nothing, so no cache, L1, and
L2 memory all land within noise of each other (the recompute column is
close to zero there). That is also why the crossover rule asks for a win
of at least 10% at a size *and every larger one*: L2 memory "wins" by
10 ms at 1 and 2 blocks, but so would any run that happened to be quick,
and it does not hold at 8 blocks.

### L2 in CPU memory: cheap, and hidden behind the floor

Loading a block from pinned CPU memory costs about 1.3 ms (28 MiB at about
22 GB/s, close to the PCIe copy rate `benchmarks/bench_offload.py`
measures). Inside a request, those copies run on the GPU while the CPU is
still launching the first kernels, so they add almost nothing: L2 memory
tracks L1 at every size. Once the prompt is long enough for compute to
matter, L2 memory saves what L1 saves. Above the floor, recomputing a
block costs about 1.5 to 2.5 ms (more for later blocks, whose attention
covers a longer prefix) and loading one about 1.3 ms, so the margin grows
with prompt length.

**Crossover: 16 blocks (4,096 reused tokens).**

### L2 on disk: slower than recomputing, for this model

A cold read of one block takes about 150 ms. Splitting one read shows it
is the storage: reading the 28 MiB file with the page cache dropped takes
about 120 ms (0.22 GiB/s on this machine's container filesystem), while
`torch.load` takes about 7 ms. With the page cache warm, a block costs
about 17 ms (10 ms to copy the file, 7 ms to deserialize). That is 8 to
75 times the ~2 ms it takes to recompute the block, so disk never pays off
for Qwen3-0.6B here.

**Crossover: none up to 32 blocks (8,192 tokens).**

### Storing sits on the request's critical path

Version 1 stores inside `hash_blocks`, before the request returns. With
pinned CPU memory that adds about 5 to 9 ms per block, mostly allocating
the pinned buffer. With disk it adds about 270 to 330 ms per block: roughly
90 ms to serialize (including the copy off the GPU) and the rest to write
and `fsync` the file. A cold 8,224-token request takes 10.7 s instead of
0.125 s.

## What would move the crossover

L2 pays off when loading a block costs less than recomputing it:

- **Bigger models** recompute each block more slowly. Compute per token
  grows with the parameter count, while block size grows only with layers,
  KV heads, and head size, which grouped-query attention keeps small. This
  is not measured here; trying a larger model is the most useful next
  experiment.
- **Faster storage**: a local NVMe drive reads at several GiB/s rather than
  0.22 GiB/s, which would bring a cold disk read close to the page-cache
  case.
- **Fewer copies on the disk path**: today a warm read copies the file into
  Python bytes and then deserializes it; memory-mapping the file or reading
  straight into pinned memory would remove one or both steps.
- **Storing off the critical path**: a background writer with batched
  `fsync` would take the store cost out of TTFT entirely.
- **A lower per-step floor**: capturing prefill in CUDA graphs would make
  compute, and therefore reuse, visible at shorter prompts.

## Limits of this measurement

- One model, one GPU, one filesystem. The floor in particular depends on
  CPU speed and the Python version.
- Single requests, one at a time. Batching several requests per step
  shares the per-step floor among them and changes the picture.
- Prompts are random token IDs; cost does not depend on token values.
- Medians of 5 rounds. Under the floor, single setups vary by about 5 to
  10 ms from run to run.
