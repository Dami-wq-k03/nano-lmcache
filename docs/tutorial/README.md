English | [简体中文](../zh-CN/tutorial.md)

# From chunk keys to engine restoration

Work through these steps in order. Each step pairs a small experiment with
an explanation and the tests that establish the behavior. Run commands from
the repository root after `uv sync --locked`. Steps 1–5 use only the core
package; step 6 adds model weights and optional dependencies; step 7 needs
Linux and an NVIDIA GPU. Step 8 returns to the core package.

| Step | Question |
| --- | --- |
| [1. Chunk keys](#1-build-chunk-keys) | Why must the key depend on the whole prefix? |
| [2. Prefix matching](#2-stop-at-the-first-miss) | What happens after the first miss or an incomplete chunk? |
| [3. Eviction](#3-branch-prefixes-and-evict-leaves) | How can a bounded cache keep shared prefixes reachable? |
| [4. Storage tiers](#4-persist-and-promote-chunks) | What survives a restart, and where does a hit come from? |
| [5. Trace viewer](#5-follow-a-lookup-in-the-viewer) | Which chunks were hit, missed, or never checked? |
| [6. Real tensors](#6-reuse-real-model-tensors) | Does reuse preserve model output, and does it save time? |
| [7. Engine restoration](#7-restore-a-real-engines-cache) | How do external L2 blocks return to the engine's GPU L1? |
| [8. PD disaggregation](#8-split-prefill-and-decode) | What must a prefill worker hand over so a decode worker can continue? |

## 1. Build chunk keys

```bash
uv run python examples/demo_chunking.py
uv run pytest tests/unit/test_chunking.py
```

The demo compares two requests with chunk size 2. Their first two chunks
match, so the summary reports two shared chunks and four reusable tokens.
This introductory demo compares isolated chunk hashes. The actual cache
also passes the previous chunk's key as `prefix_key`.

Experiment: in a Python session started with `uv run python`, compare the
same suffix behind different prefixes:

```python
from nano_lmcache.core.chunking import make_chunk_key

a = make_chunk_key([10, 20], namespace="lesson")
b = make_chunk_key([99, 20], namespace="lesson")
assert make_chunk_key([30, 40], namespace="lesson", prefix_key=a) != make_chunk_key(
    [30, 40], namespace="lesson", prefix_key=b
)
```

Identical suffix tokens cannot share KV when earlier tokens differ. The
namespace also separates incompatible model/cache configurations; callers
of the generic cache must choose it correctly.

Read [key construction](../architecture.md#how-a-chunk-key-is-built) and
[`core/chunking.py`](../../src/nano_lmcache/core/chunking.py). The test
`test_chunk_key_depends_on_prefix` covers the experiment.

## 2. Stop at the first miss

```bash
uv run python examples/demo_prefix_cache.py
uv run pytest tests/unit/test_prefix_cache.py
```

The unbounded part starts cold, then reuses a shared prefix. A lookup walks
from the root and stops at its first missing chunk. Later complete chunks
count as misses in the reuse metrics even though they were not probed.

Experiment: run this in `uv run python`:

```python
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache

cache = ChunkedPrefixCache(chunk_size=2, namespace="lesson")
cache.store([10, 20, 30, 40, 50], ["kv-0", "kv-1"])
assert cache.lookup([10, 20, 30, 40, 50]).matched_tokens == 4
assert cache.lookup([10, 20, 99, 40, 50]).matched_tokens == 2
assert cache.lookup([99, 20, 30, 40, 50]).matched_tokens == 0
```

The trailing token 50 forms an incomplete chunk and is excluded from both
storage and reuse. Changing the first chunk prevents reuse of the later
chunk, even though its tokens still match.

Read [lookup](../architecture.md#lookuptoken_ids) and
[`core/prefix_cache.py`](../../src/nano_lmcache/core/prefix_cache.py).
Tests `test_lookup_stops_when_first_chunk_misses` and
`test_incomplete_trailing_chunk_is_not_reused` cover these boundaries.

## 3. Branch prefixes and evict leaves

The bounded sections of the prefix-cache demo show why ordinary per-entry
LRU is insufficient: deleting a parent strands every stored descendant.
Only a leaf can be evicted; among leaves, the least recently used goes first.

```bash
uv run python -m nano_lmcache.visualize --scenario prefix_branching --max-entries 3 \
  --output /tmp/nano-branching.html
uv run python -m nano_lmcache.visualize --scenario lru_eviction --max-entries 2 \
  --output /tmp/nano-eviction.html
uv run pytest tests/unit/test_bounded_memory.py tests/unit/test_leaf_eviction.py tests/unit/test_byte_capacity.py
```

Open the HTML files. The branching case keeps the shared ancestors while
replacing divergent leaves. The eviction case revisits an earlier root
that no longer fits in the two-entry cache.

Experiment: rerun the branching command with `--max-entries 2`. The shared
two-chunk prefix fits, but none of its third chunks can be admitted. A
rejected write must preserve the existing prefix. Compare the demo's byte
budget with its entry budget: payload size can force several leaf evictions.

Read [capacity and eviction](../architecture.md#capacity-and-eviction),
[`storage/eviction.py`](../../src/nano_lmcache/storage/eviction.py), and the
[performance report](../eviction-performance.md). The randomized leaf tests
compare heap-based selection against the original LRU policy.

## 4. Persist and promote chunks

Use a new directory for the first command, then reuse that directory:

```bash
uv run python examples/demo_disk_cache.py --dir /tmp/nano-learning-kv
uv run python examples/demo_disk_cache.py --dir /tmp/nano-learning-kv
uv run python -m nano_lmcache.inspect /tmp/nano-learning-kv
uv run python -m nano_lmcache.inspect /tmp/nano-learning-kv --json
uv run python examples/demo_tiered_storage.py
uv run pytest tests/unit/test_disk_cache.py tests/unit/test_tiered_storage.py tests/unit/test_thread_safe_storage.py
```

A fresh disk run reports `Reused 0/3 chunks`; the second process reports
`Reused 3/3 chunks`. The inspector reads persisted metadata without changing
recency or repairing files. The tiered demo shows memory hits, disk hits,
promotion, and a simulated restart with empty memory.

Experiment: use another fresh directory and add `--max-bytes 400` to both
disk runs. The budget counts headers plus serialized payloads, so only a
prefix may fit. Opening an existing directory with a budget below its
current size raises an error rather than evicting valid files on startup.

Read [disk storage](../architecture.md#disk-storage),
[tiered storage](../architecture.md#tiered-storage), and
[thread safety](../architecture.md#thread-safety). Shared memory, disk, and
tiered instances serialize backend operations across threads. A whole
multi-chunk lookup is not a snapshot, and separate objects/processes must
not write the same disk directory concurrently.

## 5. Follow a lookup in the viewer

```bash
uv run python -m nano_lmcache.visualize --scenario partial_prefix_hit \
  --output /tmp/nano-partial.html
uv run python -m nano_lmcache.visualize --scenario incomplete_chunk \
  --output /tmp/nano-incomplete.html
uv run pytest tests/unit/test_replay.py tests/unit/test_trace.py tests/unit/test_engine_trace.py
```

Open the files directly in a browser. In the partial-prefix case, the second
request hits two chunks and misses the third. A cold request stops probing
after its first miss; subsequent complete chunks are `NOT_CHECKED`. The
incomplete case labels the trailing token `INCOMPLETE`. Use Next/Previous
to connect the timeline to the prefix tree and event details.

Experiment: rerun with `--scenario full_prefix_hit` and compare the second
request. Then try `--storage tiered --storage-dir /tmp/nano-learning-trace`
to see named storage-tier events; use a fresh directory for a cold run.

With a FreeInference JSONL file you can also run:

```bash
uv run python -m nano_lmcache.visualize --freeinference-jsonl /path/to/sample.jsonl --limit 100 \
  --output /tmp/nano-freeinference.html
```

Those block IDs are opaque identities, not model token IDs. Dataset-observed
cached tokens and simulated hits are separate measurements. For large,
already-sorted JSONL inputs, add `--stream --max-entries 4096 --max-events
10000 --max-requests 200 --sample-every 10` and optionally `--summary-json
/tmp/replay-summary.json`. Replay keeps exact totals while limiting retained
detail. `--limit` selects requests; budgets are counts rather than a total
process memory limit. See [streaming replay](../streaming-replay.md) for
ordering requirements, sampling semantics, capacity experiments, and the
batch-only recorded engine JSON import.

Read [`replay/runner.py`](../../src/nano_lmcache/replay/runner.py) and
[`FreeInferenceAdapter`](../../src/nano_lmcache/datasets/adapters/freeinference.py).
Replay tests check first-miss stopping, branching, and opaque block handling.

## 6. Reuse real model tensors

This step downloads model weights unless `--model` points to a local model.
It runs on CPU with the optional `hf` dependencies:

```bash
uv sync --extra hf
uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B
uv run --extra hf pytest tests/tensor/test_hf_reuse.py tests/tensor/test_kv_chunks.py
```

Prompt B reuses the document prefix stored while processing prompt A. The
demo reports reused/computed tokens, time with caching, time from scratch,
and maximum logit difference. Lossless storage preserves cached KV;
floating-point execution can still produce numerical differences. Reuse
also pays lookup, reconstruction, and storage costs, so a hit need not be faster.

Experiment on an NVIDIA GPU: compare direct GPU storage with CPU offload,
then run the disk-backed variant twice using the same directory:

```bash
uv run --extra hf python examples/demo_kv_reuse.py --device cuda
uv run --extra hf python examples/demo_kv_reuse.py --device cuda --offload
uv run --extra hf python examples/demo_kv_reuse.py --device cuda --offload \
  --disk /tmp/nano-learning-tensors
```

Read [real tensors](../architecture.md#real-kv-tensors),
[CPU offload](../architecture.md#gpu-to-cpu-offload), and
[`tensor/hf.py`](../../src/nano_lmcache/tensor/hf.py). For lossy compression,
use the separate [INT8 experiment](../quantized-kv.md); use a different cache
directory when changing serializers or incompatible model configurations.

## 7. Restore a real engine's cache

Requirements: Linux, an NVIDIA GPU, Python 3.11 or 3.12, and a local
Qwen3-0.6B model. Use a new disk directory for the first invocation:

```bash
uv sync --extra hf --group nanovllm
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B \
  --dir /tmp/nano-learning-engine --trace-output /tmp/nano-cold.json
uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B \
  --dir /tmp/nano-learning-engine --trace-output /tmp/nano-restart.json
uv run python -m nano_lmcache.visualize --trace-json /tmp/nano-restart.json \
  --output /tmp/nano-engine.html
```

The second process starts with empty GPU L1 and memory L2, restores the
shared prefix from disk, and imports it into engine KV blocks. Later
requests within that process can reuse GPU L1. The demo's fast L2 is
`MemoryKVCache`, which keeps tensors on their existing device; it is not
an explicit pinned CPU-offload tier.

Experiment: compare the cold and restart traces, then distinguish an
`l2_lookup` availability probe from an actual `l2_hit`/`l2_restore`.
GPU copy events describe stream enqueue order, not CUDA completion time.
The engine always computes its last prompt block to obtain logits, even
when that block is complete; the generic cache's incomplete-chunk rule
alone does not explain this behavior.

Run the real-model checks in the same GPU environment:

```bash
NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B uv run \
  --group nanovllm pytest tests/integration/test_nanovllm_engine.py
```

Tests cover restored KV, matching generated tokens, disk reuse after
restart, and trace-to-HTML conversion. Without optional engine dependencies,
[`test_engine_trace.py`](../../tests/unit/test_engine_trace.py) still checks
trace import and rendering on CPU.

Read the [integration rules](../nanovllm-integration.md#rules-the-integration-must-keep)
and [engine trace guide](../engine-traces.md). Use the controlled
[TTFT benchmark](../nanovllm-benchmark.md) for speed comparisons: demo TTFT
includes tracing overhead and synchronous writes and is not an isolated
measurement of cache speedup.

A [GPU validation record](../gpu-validation.md) records a completed run of
these examples, including two-process restoration and trace rendering.

## 8. Split prefill and decode

```bash
uv run python examples/demo_pd_disaggregation.py
uv run pytest tests/unit/test_disaggregation.py
```

This step needs only the core package. `prefill` computes a prompt's KV and
first token, then places a checksummed handoff in a channel. `decode` claims
it and generates the rest. The demo prints identical colocated and
disaggregated tokens, shows that decode computes only new positions, and
demonstrates checksum, model-identity, and single-claim failures. Its
timeline compares one GPU with a prefill GPU plus a decode GPU. Colocated,
a long prompt stalls a running chat request; disaggregated, the stall
disappears, but every TTFT includes the KV transfer.

Experiment: compare the prompt's complete chunks with the prefix cache's
rule from step 2. The handoff includes every prompt position because a
decode worker has no way to recompute a missing tail.

Read the [PD disaggregation guide](../pd-disaggregation.md), then the
single teaching module
[`disaggregation.py`](../../src/nano_lmcache/disaggregation.py).

Return to the [documentation index](../README.md), or choose a contribution
from the [roadmap](../../ROADMAP.md).
