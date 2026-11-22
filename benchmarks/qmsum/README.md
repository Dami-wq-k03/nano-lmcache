# QMSum-based Prefix KV Cache Reuse Benchmark

Replay different questions about **complete, unchanged meeting transcripts** through
single-GPU nano-vLLM. This supplements, rather than replaces,
`benchmarks/bench_nanovllm.py`.

## Requirements

Linux, CUDA, the repository's pinned nano-vLLM/FlashAttention dependencies, and a
local model **supported by that engine's weight loader**. TP must be 1. The pinned
engine implements dense Qwen3, not AWQ, FP8, GGUF, or Qwen3.5/3.6 architectures.
Two GPUs can run independent experiments; they do not make TP supported.

```bash
uv sync --extra hf --group nanovllm
# Only the plotting step needs an additional package:
uv pip install matplotlib
```

Keep datasets, converted checkpoints, and experiment outputs outside Git. The
examples below use a directory outside the repository. Do not start Python from
inside a downloaded dataset directory; `-I` isolates Python's import paths.

## 1. Download and freeze the workload (CPU only)

```bash
export EXPERIMENT="$HOME/experiments/qmsum"
export MODEL=/absolute/path/to/local/supported/model
mkdir -p "$EXPERIMENT/raw"
for split in train val test; do
  # Each download gets its own new, empty directory.
  mkdir "$EXPERIMENT/raw/$split"
  curl -fL "https://raw.githubusercontent.com/Yale-LILY/QMSum/83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e/data/ALL/jsonl/$split.jsonl" \
    -o "$EXPERIMENT/raw/$split/$split.jsonl"
done
python -I benchmarks/qmsum/prepare_qmsum.py \
  --raw "$EXPERIMENT/raw" --model "$MODEL" \
  --output "$EXPERIMENT/workload" --max-model-len 32768 \
  --max-output 64 --meetings 20 --max-queries 4
```

`workload.jsonl` contains **the actual input token IDs**, not just lengths.
`workload_stats.json` contains source-file hashes, the chat-template hash, every
meeting's eligibility and prefix statistics, and the workload hash.

- Format: system instruction, then `Speaker: content` turns, then the question.
  Use the tokenizer's chat template with `enable_thinking=False`.
- Deduplicate nonempty questions, preserving general/specific source order.
- Calculate the token-ID LCP across **all** valid queries before selecting four.
- Cacheable prefix is complete 256-token blocks, capped so the engine's last
  prompt block is always computed, even if it is complete.
- Reject a whole meeting if any question plus the reserved output exceeds the
  context window. Never shorten the transcript to manufacture a bucket.
- Prefer test meetings, longest prefixes first. Only fill a short pilot from
  val/train when test cannot supply enough meetings.
- `--meetings 0 --max-queries 0` selects **all eligible test meetings and queries**.
- Buckets are `[1K,4K)`, `[4K,8K)`, `[8K,16K)`, and `>=16K`, where K = 1024.
  Empty buckets stay empty.

## 2. Run paired experiments

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -I benchmarks/qmsum/run_qmsum.py \
  --model "$MODEL" --model-revision CHECKPOINT_REVISION \
  --source-revision "$(git rev-parse HEAD)" \
  --workload "$EXPERIMENT/workload/workload.jsonl" \
  --output "$EXPERIMENT/pilot" \
  --max-tokens 1 32 --concurrency 1 4 8 --repeats 3 --warmup 2
```

For independent simultaneous jobs, select different devices and pass distinct
`--master-port` values (for example, 23410 and 23411). The benchmark redirects
only the pinned engine's initialization rendezvous, not its inference code.
Record the other job with `--co-run-label`; compare off/on only within the same
GPU/model configuration. Simultaneous jobs share CPU, host memory and PCIe
resources, so label their results as co-run, not uncontended measurements.
Every invocation requires a new, empty output directory. Reserve memory for
**both fetched payloads and the stack/scatter temporary**, not just GPU KV blocks;
large concurrent restores may need a lower `--gpu-memory-utilization`.

| Mode | Behavior |
| --- | --- |
| `off` | Native block manager; reset L1 before every wave; no L2 |
| `sync` | Pinned `CPUOffloadKVCache`; synchronous store hook |
| `async-flush` | Generic write-behind; flush **outside** request timing (C1) |
| `async-continuous` | No explicit flush between questions; pending-key reads may wait (C2) |

The generic `AsyncWriteBehindStorage.prepare` makes an owned, completed pinned
host snapshot **on the caller thread**. A worker then writes into a locked
`CPUOffloadKVCache` backend. Thus GPU export/D2H remain on the critical path, and
the generic async path has an additional host copy. This deliberately measures
that implementation, not an imaginary fully overlapped store. It does not use
`AsyncTensorWriteStorage`; that is a different experiment. Queue capacity is the
sum of possible complete prompt blocks in the cohort plus 64, avoiding dropped
writes/backpressure exceptions. There is no artificial backend sleep.

For requested concurrency C, meetings form contiguous cohorts of at most C.
Question 0 for each meeting arrives together, then question 1, etc. L1 is replaced
between waves; L2 survives only within a cohort. All modes use identical input
IDs, request waves, output lengths (`ignore_eos=True`), near-greedy temperature,
and seed. Two warmup cohorts per mode and concurrency are discarded. Mode order
rotates across measured rounds. Insufficient GPU KV capacity can reduce actual
simultaneous execution: C is **arrival concurrency**, not a promise that C long
prompts fit on the GPU. Unexpected preemption/accounting changes fail explicitly.
This is a closed-loop, barriered replay, not an open-loop serving load test.
Same-meeting simultaneous arrival / stampede is not included.

## 3. Analyze

```bash
python -I benchmarks/qmsum/analyze_qmsum.py \
  --results "$EXPERIMENT/pilot" --figures
```

Use `--label "Model name (GPU)"` to give standalone figures a readable identity.
Failed/missing correctness checks are visibly flagged on the plots; partial
concurrency subsets do not manufacture missing prefix or throughput curves.

Outputs:

- `environment.json`: model/config/workload/code hashes, source revisions,
  package versions, GPU memory, KV layout, and settings.
- `raw_results.jsonl`: request TTFT/E2E, restored blocks, prefill tokens,
  store/fetch hook timings, generated IDs and text. Async `stored_blocks` counts
  accepted queued writes, not commits completed at request return; cohort drain
  must succeed before the observations are published.
- `waves.jsonl`: batch import CPU/CUDA timings, queue waits, outside-timing flush,
  actual inter-wave gap, wave latency, peak allocated/reserved VRAM.
- `cohorts.jsonl`: full replay wall time, **including C1 flush and final async
  drain**, for sustained request throughput.
- `summary.json`, `latency.csv`, `throughput.csv`: cold/warm separately; bucket
  sample counts, P50/P95, speedups, hit rates, and every token-ID mismatch.
- `figures/`: prefix-vs-TTFT, restored-block hit rate, concurrency-vs-throughput,
  with light/dark PNGs and machine-readable CSV tables. Plots highlight off vs
  C2; the tables retain **all four modes**.

### Metric boundaries

TTFT is arrival to the first token observable after nano-vLLM's step and store
postprocessing. It includes scheduling, restore, suffix prefill, and any store
on that step. It is **not** a remote HTTP/streaming TTFT. E2E ends after the final
token's step; 32-token E2E is never substituted for TTFT. The current CUDA stream
is synchronized at step boundaries, not inside each timing hook. Background CPU
write-behind work is not forcibly flushed in C2. Inter-wave bookkeeping/reset
introduces a measured gap, so “continuous” does not mean literally zero delay.

`store_ms` covers block hashing/missing checks, GPU gather and put/prepare;
`fetch_ms` covers payload reads and H2D enqueue. CUDA event spans include host
submission gaps, not pure DMA bandwidth. Scatter/import runs for the whole
batch and is recorded **once per wave**, not duplicated across requests. At C=1,
fetch + batch import describe that request's restore. At C>1, don't add or
attribute batch import separately to every request. Queue waits are the actual
pending-key `flush()` duration; explicit C1 flush is reported separately.

L2 hit rate = successfully fetched **and subsequently imported** blocks / all
reusable prompt blocks; cold requests contribute zero hits. Successful imports
and stream completion precede result recording. Reused tokens and saved prefill
are actual L2 hits × 256; accounting is asserted against scheduled prefill.
Membership probes alone never establish a hit. Warm reuse must cover at least
the meeting's common complete prefix; extra pairwise shared blocks are allowed.

Temperature 1e-6 is **near-greedy sampling, not deterministic argmax**: the pinned
sampler still draws exponential random noise. BF16 logit ties and different RNG
consumption across full/chunked/cached paths can therefore change generated IDs,
even with the same seed. Keep this limitation in the report; don't silently
change the sampler halfway through a measured run. A native-L1 diagnostic and
bitwise KV comparisons can distinguish transfer errors from prefix-path changes,
but do not retroactively satisfy the original strict off/on acceptance criterion.

Output IDs are compared against `off` for the **same meeting, query, concurrency,
output length, and repeat**. Mismatches and missing baselines cause analyzer exit
status 2, after writing results. BF16 full-prefill and suffix-prefill kernels may
produce different greedy tokens; do not suppress mismatches or claim semantic
quality/bitwise numerical equivalence from a cache hit alone. Low sample counts
and no positive performance gain are valid findings, not reasons to alter data.

## Diagnose a generation mismatch

On a **free GPU**, compare full prefill, native L1, sync L2 and async L2. Example
for the reconstructed 14B model on the 48-GiB card (use 0.35 instead of 0.74 for
this Pilot's 0.6B/24-GiB setup):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -I benchmarks/qmsum/diagnose_qmsum.py \
  --model "$MODEL" --model-revision CHECKPOINT_REVISION \
  --workload "$EXPERIMENT/workload/workload.jsonl" \
  --meeting test-030 --query specific-0 \
  --gpu-memory-utilization 0.74 \
  --output "$EXPERIMENT/diagnostic"
```

Select meeting/query IDs from an actual mismatch in your workload. The script
saves `comparison.json`, `logits.pt`, and provenance. It checks dtype, shape, and
**uint8 views of the tensor representations**, not just floating-point equality
(which considers positive and negative zero equal). Both exported GPU -> stored
CPU and fetched payload -> GPU import are checked. Async writes are explicitly
drained before inspecting CPU snapshots: this is a correctness diagnostic, **not
another C2 performance observation**. Load your own saved logits with
`torch.load(..., weights_only=True)`.

## Optional: reconstruct an existing block-FP8 checkpoint

If only Qwen3 fine-grained FP8 weights are already available, and dense BF16 fits
one GPU, the small offline converter can reconstruct those weights:

```bash
python -I benchmarks/qmsum/dequantize_fp8.py \
  --source /absolute/path/to/existing/Qwen3-14B-FP8/snapshot \
  --output "$EXPERIMENT/models/Qwen3-14B-reconstructed-BF16"
```

It checks the format, multiplies FP8 weights by their 128×128 block scales in
FP32, casts to BF16, removes quantization metadata, and writes new shards plus
source-shard hashes. The source checkpoint remains unchanged. This is **BF16
inference using reconstructed quantized weights**, not native FP8 inference and
not the original pre-quantization BF16 model. No activation quantization remains.
Do not compare its quality to the original BF16 model without a separate eval.

## CPU regression tests

```bash
uv run pytest -q tests/unit/test_qmsum.py
```

Dataset source: [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum). Respect the
upstream dataset's terms; do not commit meeting text, tokenized workloads, model
weights, or large/private experiment results.
