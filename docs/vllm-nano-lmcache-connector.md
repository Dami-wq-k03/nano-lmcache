# vLLM with nano-lmcache connector

## vLLM version baseline (vllm==0.33.0)

Run from the repository root in a separate Linux environment with a compatible
NVIDIA GPU and vLLM CUDA stack. Do not combine the `vllm` group with the `hf`,
`torch`, `tensor-test`, or `nanovllm` groups; their PyTorch sources differ.

```bash
uv sync --locked --group vllm
uv run --locked --group vllm python examples/demo_vllm_with_nano_lmcache_connector.py \
  --model /path/to/Qwen3-0.6B
```

Use local model weights to avoid a download; the demo defaults to
`Qwen/Qwen3-0.6B` if `--model` is omitted. Pinning 0.33.0 gives V1–V5 a
reproducible baseline for connector development and testing.

## Current scope and validation

The connector implements shared external prefix lookup, synchronous KV
save/load, and disk-backed restart reuse. It is not a no-op connector.
The teaching baseline is one GPU, TP = PP = 1, dense full attention,
unquantized KV, one cache group, and `chunk_size == block_size`.

- Scheduler and worker must share a `remote_cache` server for external reuse;
  private in-memory caches do not share worker saves with the scheduler.
- Local APC reuses resident GPU blocks. An external hit instead fetches a
  payload and restores it into allocated GPU blocks. The demo compares
  recomputation, local APC, and external storage across engine/server restart.
- Loads and saves are synchronous. The standalone `AsyncTensorWriteStorage`
  remains experimental and is not wired into this connector; the benchmark's
  `--async-store` setting is currently ignored by the connector.
- Missing or invalid load payloads fail explicitly rather than reporting
  failed block IDs for automatic recomputation. This version does not track
  load errors or emit KV cache events.

The generation and HF-reference tests use cached `Qwen/Qwen3-0.6B` weights at
revision `c1899de289a04d12100db370d81485cdf75e47ca`. Shared external-hit and
load-failure tests additionally need `NANO_LMCACHE_TEST_MODEL` pointing to local
Qwen3-0.6B weights. For current source checks in the vLLM environment:

```bash
uv run --locked --group vllm pytest tests/unit/test_nano_lmcache_connector.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NANO_LMCACHE_TEST_MODEL=/path/to/Qwen3-0.6B uv run --locked --group vllm \
  pytest tests/integration/vllm/test_vllm_with_nano_lmcache_connector.py
```

The integration tests check generated token agreement, saved KV against a
Hugging Face reference, shared external hits with APC disabled, and explicit
load failure. The [serving report](serving-validation-report.md) preserves
historical real-model measurements and their source revision: external hits
were correct but slower than recomputation. Its async measurements are not a
claim about the simplified connector, and no general serving speedup is established.

## Request-processing sequence

This shows a synchronous load step. The scheduler and worker are separate roles, and planning, forwarding, and feedback repeat for many steps.

```mermaid
sequenceDiagram
    participant S as Scheduler-side
    participant W as Worker-side

    S->>S: get_num_new_matched_tokens(request, num_computed_tokens)
    S->>S: update_state_after_alloc(request, blocks, num_external_tokens)
    S->>S: build_connector_meta(scheduler_output)
    S-->>W: scheduler_output with connector metadata
    W->>W: bind_connector_metadata(connector_metadata)
    W->>W: start_load_kv(forward_context, **kwargs)
    loop each attention layer
        W->>W: wait_for_layer_load(layer_name)
        W->>W: attention forward
        W->>W: save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
    end
    W->>W: wait_for_save()
    W->>W: get_finished(finished_req_ids)
    W->>W: get_kv_connector_kv_cache_events()
    W-->>S: aggregated connector output
    S->>S: update_connector_output(connector_output)
```

## NanoLMCacheConnector hook flow

This is a concrete example with small prompts so slot numbers stay readable. It
uses Qwen3-0.6B (28 layers, 8 KV heads, `head_dim = 128`), BF16 KV,
`block_size = chunk_size = 16`, one full-attention cache group, TP = PP = 1,
APC off, and a shared `remote_cache`. One stored chunk has shape
`[28, 2, 8, 16, 128]`; one worker layer has shape
`[num_blocks, 8, 16, 256]`.

- **Request A (cold):** 48 prompt tokens, block table `[7, 2, 9]`. The
  external cache is empty. vLLM schedules all 48 tokens in one prefill step.
- **Request B (warm):** 48 prompt tokens. Tokens 0–31 are identical to A's;
  tokens 32–47 differ. Its block table is `[5, 8, 12]`, so the same logical
  KV occupies different physical blocks.
- After A, the external cache holds A's chunks 0–2. B can load chunks 0–1
  (32 tokens) and must compute and publish chunk 2.

The hook order for B is:

1. scheduler `get_num_new_matched_tokens` returns `(32, False)`;
2. scheduler `update_state_after_alloc` turns the match into a load plan;
3. scheduler `build_connector_meta` emits a `ReqMeta` with load=32 and
   save-skip=32;
4. worker `bind_connector_metadata` installs that metadata;
5. worker `start_load_kv` scatters chunks 0–1;
6. worker `wait_for_layer_load` checks that `start_load_kv` ran;
7. attention computes tokens 32–47;
8. worker `save_kv_layer` collects chunk 2 layer by layer;
9. worker `wait_for_save` verifies all 28 layers arrived;
10. worker `get_finished` and `get_kv_connector_kv_cache_events` return their
    per-step results (both empty);
11. scheduler `update_connector_output` consumes the worker feedback;
12. scheduler `request_finished` drops the request state;
13. worker `shutdown` releases connector state when the engine exits.

### __init__ (scheduler and worker)

1. Store the transfer role, `block_size`, `chunk_size = block_size`, and a
   namespace fingerprint over the model path/revision, HF config, dtype,
   quantization, KV dtype, TP/PP sizes, chunk size, and layout version.
2. If `remote_cache` is configured, parse `host`/`port` and create
   `RemoteCacheClient(..., serializer=TensorSerializer())`. Both roles point
   at the same server.
3. Create the role cache: the scheduler owns `_lookup_cache`; the worker owns
   `_save_cache`. Both are `ChunkedPrefixCache` instances using the shared
   storage. Without `remote_cache`, each process gets a private in-memory
   cache, so worker saves are invisible to scheduler lookups.
4. Initialize scheduler state (`_request_trackers`, `_load_specs`) or worker
   state (`_layer_names`, `_kv_caches`, `_pending_save_layers`,
   `_load_started`). The constructor performs no CUDA work.

In the example, both processes derive the same namespace, and each creates a
client for the shared server.

### Scheduler: get_num_new_matched_tokens

1. Drop any stale `_load_specs[request_id]` from an earlier scheduling
   attempt.
2. If the request has no prompt token IDs, report zero matches.
3. Without shared storage, call `_lookup_cache.lookup(token_ids)`, cap the
   match at `(len(token_ids) - 1) // 16 * 16`, and subtract
   `num_computed_tokens` to obtain the additional tokens.
4. With shared storage, reject an unaligned local prefix
   (`num_computed_tokens % 16 != 0`).
5. Otherwise build chained chunk keys for every complete chunk in
   `token_ids[:-1]`, stop at the first key absent from storage, and count the
   matching chunks. Cap the result at `(len(token_ids) - 1) // 16 * 16` and
   subtract `num_computed_tokens`.
6. If the additional count is positive, remember
   `LoadSpec(vllm_cached_tokens=num_computed_tokens,
   external_cached_tokens=matched_end)`. Return
   `(additional_matched_tokens, False)`; `False` means a synchronous load.

For A the cache is empty, so the method returns `(0, False)`. Even with a
nonempty cache it inspects only chunks 0–1, because `token_ids[:-1]` has 47
tokens and the final prompt token is reserved for logits.

For B the first 32 tokens are present. The method builds keys for chunks 0–1
of B's 47-token prefix, both `contains` checks succeed, and it returns
`(32, False)` with `LoadSpec(0, 32)`. vLLM therefore marks 32 tokens as
computed and schedules only the remaining 16 tokens.

### Scheduler: update_state_after_alloc

1. vLLM passes the request, the blocks it allocated, and
   `num_external_tokens`.
2. If no `LoadSpec` was stored, this is a cold request; return without
   enabling a load.
3. If `num_external_tokens <= 0`, set `can_load = False`.
4. Compute `expected = external_cached_tokens - vllm_cached_tokens`, take
   `allocated = min(num_external_tokens, expected)`, and round down to a
   complete chunk. If it is zero, set `can_load = False`.
5. Set `external_cached_tokens = vllm_cached_tokens + allocated` and
   `can_load = True`.

For B, vLLM allocates blocks `[5, 8, 12]` and reports 32 external tokens.
`expected` and `allocated` are both 32, so the load covers tokens 0–31.
Request A has no `LoadSpec`, so this hook returns immediately.

### Scheduler: build_connector_meta

1. Remove trackers and load specs for finished or preempted requests.
2. Create a `RequestTracker` for each new request, initialized with its full
   prompt, first-group block table, and current computed-token count. For
   cached requests, update the computed count and append new blocks; replace
   the table when the request resumed.
3. For every scheduled request, prefer the scheduler's current block state
   when supplied, then pop its `LoadSpec`.
4. If a load is possible, raise `tracker.num_saved_tokens` to at least the
   external hit rounded down to a chunk, so the worker does not republish
   loaded chunks.
5. Build `ReqMeta` from the tracker: `load_end` is the external hit;
   `compute_end` is `min(prompt_len, num_computed + num_scheduled)`;
   `save_end` is `compute_end` rounded down to a chunk; `skip` is
   `num_saved_tokens` rounded down; `can_save` requires uncomputed prompt
   tokens and `save_end > skip`. The span ends at
   `max(save_end if can_save else 0, load_end)`, slots are derived from the
   block table, and the load spec is attached when `load_end > 0`. When
   `can_save` is true the tracker's `num_saved_tokens` advances to
   `save_end`.
6. Increment the step number, record the scheduled request IDs, and return
   `NanoLMCacheConnectorMetadata`.

For A, the tracker is `(48 tokens, blocks [7, 2, 9], computed 0)`. The plan
has `skip=0`, `save_end=48`, slots for all 48 tokens
(`112–127, 32–47, 144–159`), and no load.

For B, the tracker starts at 32 computed tokens with blocks `[5, 8, 12]` and
16 scheduled tokens. The load spec sets `num_saved_tokens=32`; the plan has
`load_end=32`, `save_end=48`, `skip=32`, `can_save=True`, and spans all 48
tokens (`80–95, 128–143, 192–207`). The worker must load the first 32 slots
and save only the last 16.

### Worker: register_kv_caches

1. vLLM calls this once during worker initialization with a mapping from
   layer name to the layer's paged KV tensor.
2. Reject a later call that changes the layer set.
3. Sort layer names naturally (layer 2 before layer 10) and keep the ordered
   mapping used for chunk assembly.
4. With shared storage, validate that every tensor is four-dimensional,
   matches the first tensor's shape and dtype, has `block_size` in the token
   axis and an even last dimension. This is the
   `[num_blocks, num_kv_heads, block_size, 2 * head_dim]` layout.

In the example, 28 tensors of shape `[num_blocks, 8, 16, 256]` are registered;
the expected stored chunk shape is `[28, 2, 8, 16, 128]`.

### Worker: bind_connector_metadata

1. Reset the per-step `load_started` flag.
2. Clear pending per-layer save buffers.
3. Delegate to the base class.

For B this installs the one-request metadata produced by
`build_connector_meta`; all load state starts empty.

### Worker: start_load_kv

1. Select requests whose `load_spec` exists and has `can_load=True`. If none
   exist, return immediately (request A). On a no-load step, vLLM's V2
   runner may place this call after forward, which is harmless because there
   is nothing to scatter.
2. Require registered KV layers.
3. Compute chained keys for `token_ids[:external_cached_tokens]`. Iterate
   chunk indices from `vllm_cached_tokens // 16` to
   `external_cached_tokens // 16`.
4. For each chunk, fetch it from shared storage, then scatter it into all
   layers: concatenate the stored K and V halves, permute to
   `[tokens, heads, 2 * head_dim]`, and assign
   `layer[slot // block_size, :, slot % block_size]`.
5. After all requests, set `load_started = True`.

For B, chunk indices 0 and 1 are fetched and written to slots 80–95 (block 5)
and 128–143 (block 8) in every one of the 28 layers. Request A has no load
requests, so the method returns before touching the cache.

### Worker: wait_for_layer_load

1. Check whether the metadata contains any loadable request. If it does and
   `start_load_kv` has not run, raise an error; attention must not consume a
   layer before the scatter is issued.

For A no loads exist, so the hook is a no-op.

### Worker: save_kv_layer

1. Return immediately for a `kv_consumer` role.
2. If the metadata has no requests, return; decode-only steps and steps that
   complete no new chunk have nothing to save.
3. Validate that the layer name was registered.
4. For each request with `can_save=True`, set `start = skip_leading_tokens`
   and `end = floor(len(token_ids) / 16) * 16`. Skip if `start >= end` or
   `start` is not chunk-aligned.
5. Gather `kv_layer[slots // 16, :, slots % 16]` for that span, copy it to
   CPU, split K and V, permute each to `[heads, tokens, head_dim]`, and keep
   them in `_pending_save_layers[request_id][layer_name]`.
6. When all registered layer names are present, order the layers, call
   `split_kv()` to form chunks, and call
   `save_cache.store(token_ids[:end], chunks, start_chunk=start // 16,
   request_id=...)`.

For A, `start=0` and `end=48`; after all 28 layers arrive, three chunks are
stored. For B, `start=32` and `end=48`; only chunk 2 is gathered from block 12
and stored at index 2. The two loaded chunks are not saved again.

### Worker: wait_for_save

1. For every request in `_pending_save_layers`, compute the missing layer
   names against the registered set.
2. Raise an error if any layer is missing; otherwise the save is complete.

In the example all 28 layers arrive for both requests. If vLLM skipped a layer
hook, this check fails before the step can finish silently with partial KV.

### Worker: get_finished

1. Delegate to the base implementation. This connector has no asynchronous
   sends or receives, so the result is empty.

Both A and B finish with no pending transfers.

### Worker: get_block_ids_with_load_errors

Not overridden. The base implementation returns an empty set: the connector
assumes every external load succeeds and does not track load failures.

### Scheduler: update_connector_output

1. Delegate to the base class.

In the example there are no asynchronous sends, and the invalid-block set is
empty on success.

### Scheduler: request_finished

1. Remove the request's `RequestTracker` and any leftover `LoadSpec`.
2. Delegate to the base class. Chunks already published to the external cache
   remain available for later requests.

When A and B finish, their planning state is dropped; A's chunks remain in the
cache and are what B reused.

### Worker: get_kv_connector_kv_cache_events

Returns `None`. This connector does not emit a KV cache event stream.

### Worker: shutdown

1. Clear pending save buffers.
2. Reset the `load_started` flag.
3. Delegate to the base class. Shared cache entries are not deleted; disk or
   server contents survive for a later engine.

After B finishes, shutdown releases connector state while the external cache
still contains A's chunks and B's newly saved chunk 2.

