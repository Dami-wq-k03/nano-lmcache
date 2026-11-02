# Real nano-vLLM engine traces

VIS-10 connects a real inference engine to the same offline viewer used by
synthetic workloads. Recording requires Linux, an NVIDIA GPU, and the pinned
nano-vLLM dependency group. Reading the JSON and generating HTML require only
the core package: no torch, GPU, web server, or CDN.

## Run the experiment

Use a local Qwen3-0.6B model directory and a fresh cache directory for the
first invocation. Use the same cache directory in the second invocation:

```bash
uv sync --extra hf --group nanovllm
uv run python examples/demo_nanovllm.py \
  --model /path/to/Qwen3-0.6B --dir /tmp/engine-kv \
  --trace-output cold.json
uv run python examples/demo_nanovllm.py \
  --model /path/to/Qwen3-0.6B --dir /tmp/engine-kv \
  --trace-output restart.json
uv run python -m nano_lmcache.visualize \
  --trace-json restart.json --output engine-trace.html
```

Open `engine-trace.html` directly in a browser. Next, Previous, Play, Pause,
and Reset navigate the timeline; selecting an event highlights its request's
block and prefix key and shows the full metadata. The chunk view and summary
show final per-request results, while the timeline selects individual steps.

The first process computes the document's KV and writes complete blocks to
disk. The second process starts with empty L1 and memory L2, restores the
shared prefix from disk, and promotes it to the fast L2 tier. Later requests
inside either process reuse the shared prefix from GPU L1.

The demo makes two generation calls per question: `/ttft` generates one token
and `/answer` generates a short answer. These are separate requests, each
with a unique `/seq-N` suffix. Warmup happens before tracing is attached.
`--memory-max-entries N` bounds the fast L2 tier so its leaf evictions can be
observed when writes or promotions exceed capacity.

## Read the events

| Event | Observation |
| --- | --- |
| `request_started` | Prompt length, block size, and model label at allocation |
| `l1_hit` | One committed block reused in nano-vLLM's GPU prefix cache |
| `l2_lookup` | Membership probe; `available` does not mean KV was fetched |
| `l2_hit` / `l2_restore` | A block payload was successfully read from L2 |
| `gpu_import` | Copy of a restored payload into the engine's KV cache enqueued |
| `gpu_export` | Copy out of the engine's KV cache enqueued before storage |
| `storage_get` / `storage_put` | Actual access to the named memory or disk tier |
| `eviction` / `promotion` | Tier-local removal / slow-to-fast transfer |
| `prefill_step` | Number of uncached prompt tokens computed in this step |
| `request_completed` | Demo generation call completed; optional measured TTFT |

`chunk_index` is a logical prefix position. `gpu_block_id` is a physical slot
in the engine's KV tensor. Keys include the preceding prefix and the model
layout namespace; physical slots can be reused for unrelated keys.
Queued imports retain their original sequence identity even when several
requests are restored in one forward pass after their allocation contexts end.

Only committed, indexed `l1_hit` and `l2_hit` events count toward reuse in the
viewer. Repeated scheduling probes do not inflate the count. Blocks labeled
`COMPUTE` need prefill; nano-vLLM always computes the last prompt block even
when it is complete, to obtain logits. Synthetic replay's `INCOMPLETE` rule
therefore differs from this engine rule.

GPU copy events have `timing: stream_enqueued`. They describe host-side
ordering, not CUDA completion timestamps or measured transfer durations;
tracing adds no per-copy CUDA synchronization. The demo's one-token TTFT is
measured with synchronization around generation and includes Trace overhead,
L2 reads, and synchronous disk writes. The answer call has no TTFT estimate.
Use [the benchmark](nanovllm-benchmark.md) for controlled speed comparisons.

The demo's fast L2 is `MemoryKVCache`, which retains exported tensors on their
existing device. It is not a pinned CPU-offload tier. The disk serializer
stages tensors through CPU; a `gpu_export` event alone does not imply CPU
offload. `CPUOffloadKVCache` is available separately for explicit pinned CPU
storage (see [the integration design](nanovllm-integration.md)).

## Trace format and privacy

`TraceRecorder.write_json()` produces an event array. The demo writes a
versioned envelope, also accepted by `--trace-json`:

```json
{"format_version": 1, "source": "nanovllm", "events": []}
```

Each event follows `observability/events.py`; sequence numbers must be
strictly increasing. Unknown event types remain visible in the timeline.
The importer validates the event schema and rejects unsupported versions.

The demo records metadata, keys, generic request labels, and the model
directory's basename. It does not record prompt text, token IDs, KV tensors,
model paths, or server credentials. Custom labels and custom event metadata
are preserved in JSON and HTML: review them before sharing. Traces and HTML
are held in memory in this version; bounded large-dataset streaming remains
a separate VIS-11 task.

## Validation

`tests/integration/test_nanovllm_engine.py` covers cold storage, reopened disk
with empty L1, warm L1, and batched import attribution using the real
scheduler. Its GPU tests verify identical generated tokens after restoration
and an engine trace-to-HTML round trip. To include the real model tests:

```bash
NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B \
  uv run --group nanovllm pytest tests/integration/test_nanovllm_engine.py
```

Without the optional engine dependencies, `tests/unit/test_engine_trace.py`
still checks the JSON importer, CLI, hit accounting, and HTML data handling.
