# Glossary and FAQ

Short explanations of the terms used in nano-lmcache, a teaching project
about KV cache reuse. Each entry links to source code or the relevant
section of [Architecture](architecture.md). For a runnable sequence, follow
the [guided tutorial](tutorial/README.md).
Task status is maintained in the [roadmap](../ROADMAP.md).

Later entries build on earlier ones, so reading top to bottom works well.

## Glossary

### Key-Value (KV) cache

While processing text, a transformer computes a key (K) and a value (V) tensor for every token at every layer. A KV cache keeps them so the model does not recompute them for tokens it has already processed. nano-lmcache keeps this data between requests: a new request that starts with tokens seen before can reuse it.

With a real model, one chunk's payload is a single tensor holding K and V for every layer: `[num_layers, 2, num_kv_heads, chunk_tokens, head_dim]`.

See: [Real KV tensors](architecture.md#real-kv-tensors), [`prefix_cache.py`](../src/nano_lmcache/core/prefix_cache.py)

### Prefill

The stage where the model reads the whole input prompt and builds its KV cache. Its cost grows with prompt length, so it is the work nano-lmcache tries to skip: matched chunks are loaded instead of recomputed.

Even a fully cached prompt still recomputes its last token, because the model needs one forward step to produce the logits for the next token.

See: [`run_with_prefix_cache`](../src/nano_lmcache/tensor/hf.py), [Real KV tensors](architecture.md#real-kv-tensors)

### Decode

The stage that generates the answer one token at a time, reading the KV cache and adding each new token's K and V to it. Prefill processes the prompt; decode generates the continuation.

nano-lmcache does not implement decoding for real models. It reuses the KV produced by prefill, and `run_with_prefix_cache` returns logits, not generated text. The PD teaching module decodes with a toy model only.

See: [`run_with_prefix_cache`](../src/nano_lmcache/tensor/hf.py)

### Prefill/decode (PD) disaggregation

Running prefill and decode on separate workers, usually separate GPUs. The prefill worker sends the prompt's KV and first token to a decode worker, which continues generation without recomputing the prompt. This prevents long prefills from stalling running decodes, but every request's TTFT then includes the KV transfer. The receiving worker must check that the KV is intact, complete, and from the same model.

See: [PD disaggregation guide](pd-disaggregation.md)

### Time to First Token (TTFT)

The time between submitting a request and receiving its first generated token. For long prompts, prefill is a large part of it, so skipping prefill work can lower TTFT, but only if loading the cached data costs less than the computation it replaces.

The nano-vLLM integration includes a controlled TTFT comparison (N4). It
measures recompute, GPU L1 reuse, pinned CPU L2, disk L2, and synchronous
storage costs. A high hit rate alone does not establish a latency gain.

See: [When does nano-vLLM's L2 cache pay off?](nanovllm-benchmark.md)

### Token

A unit of text that a model processes: a word, part of a word, punctuation, or another fragment, depending on the tokenizer. Models work on sequences of integer token IDs, which is what nano-lmcache's functions take as `token_ids`.

See: [`chunk_tokens`](../src/nano_lmcache/core/chunking.py)

### Chunk

A group of `chunk_size` consecutive tokens, stored and matched as one unit. In the current implementation, only complete chunks participate in cache reuse. A shorter trailing chunk is excluded to keep storage and matching aligned to fixed-size chunk boundaries.

See: [`chunk_tokens`](../src/nano_lmcache/core/chunking.py), [Core cache concepts](architecture.md#nano-core-cache-concepts)

### Prefix

The beginning of a token sequence. In `A B C D`, both `A B` and `A B C` are prefixes; `B C` is not.

Only prefixes can be reused. A causal model lets each token attend only to the tokens before it, so a token's K and V depend on those tokens and on its position. The same tokens after a different prefix have different K and V.

See: [Why can only prefixes be reused?](#3-why-can-only-prefixes-be-reused)

### Prefix-aware key

The key of a chunk is a SHA-256 hash of the cache's `namespace`, the previous chunk's key (`prefix_key`), and the chunk's own tokens. Each key includes its parent's key, so it depends on the whole prefix: identical tokens after a different prefix get a different key and are never reused by mistake. The first chunk has no `prefix_key`.

See: [`make_chunk_key`](../src/nano_lmcache/core/chunking.py), [How a chunk key is built](architecture.md#how-a-chunk-key-is-built)

### Continuous prefix matching

A lookup checks chunks in order from the start and stops at the first miss. The chunks before the miss are the reusable prefix; everything after it must be recomputed.

For example, if chunks `A` and `B` are stored but `C` is not, the match ends after `B`. `D` is never checked, because its key depends on `C`.

See: [`ChunkedPrefixCache.lookup`](../src/nano_lmcache/core/prefix_cache.py), [`lookup(token_ids)`](architecture.md#lookuptoken_ids)

### Cache hit and cache miss

A hit means a chunk's payload was found in storage; a miss means it was not. `CacheMetrics` counts both per chunk (`hit_chunks`, `miss_chunks`, `hit_tokens`). Because a lookup stops at its first miss, every complete chunk after it is counted as a miss too, so `hit_rate` reads as "what fraction of reusable work was actually reused". A trailing partial chunk is counted on neither side.

See: [`metrics.py`](../src/nano_lmcache/core/metrics.py), [Why metrics count the way they do](architecture.md#why-metrics-count-the-way-they-do)

### Cache eviction

Removing stored entries to make room when a bounded cache is full.
`max_entries` limits the number of chunk payloads; `max_bytes` limits their
recorded size. Memory uses `size_fn`, CPU offload uses tensor bytes, and
disk uses complete serialized entry-file bytes. When both limits are set,
both must hold; these are backend budgets, not total process-memory limits.

See: [Capacity and eviction](architecture.md#capacity-and-eviction)

### Leaf

A stored chunk that has no stored children. In the chain `A -> B -> C -> D`, only `D` is a leaf, and `A` is the root.

See: [Why eviction must respect prefix chains](architecture.md#why-eviction-must-respect-prefix-chains)

### Prefix-safe eviction

Eviction that never breaks a chain: only a leaf may be evicted, and the least recently used leaf goes first. The parent of the entry being added is never the victim. If no victim exists, or the new entry's parent is not stored, the new entry is rejected instead.

Evicting `A` first would leave `B`, `C`, and `D` stored but unreachable, because a lookup starts at `A`. Prefix-safe eviction keeps this invariant: every stored entry has its whole ancestor chain stored.

See: [`PrefixLRUIndex`](../src/nano_lmcache/storage/eviction.py), [Why eviction must respect prefix chains](architecture.md#why-eviction-must-respect-prefix-chains)

### Paged attention

A way of managing KV cache memory in fixed-size blocks instead of one contiguous region per sequence. vLLM uses it. A block is a unit of memory in the engine; a chunk is a group of tokens in nano-lmcache's cache keys. They are related but not the same thing.

nano-lmcache delegates attention execution to the inference engine. Its
nano-vLLM adapter exports and imports fixed-size engine KV blocks and keys
them by model layout and logical prefix. Physical GPU block slots can be
reused for different logical prefixes.

See: [nano-vLLM block format](nanovllm-integration.md#block-format-n0)

### Cache offloading

Moving cached data from fast but small memory to larger but slower memory,
and copying it back on a hit. `CPUOffloadKVCache` keeps chunk tensors in
pinned (page-locked) CPU memory and returns them on the GPU. It adds
capacity, but every hit pays a copy cost. The
[TTFT experiment](nanovllm-benchmark.md) measures when loading beats recompute
for one model and GPU. The former G3 task was incorporated into N4.

See: [`offload.py`](../src/nano_lmcache/tensor/offload.py), [GPU to CPU offload](architecture.md#gpu-to-cpu-offload)

### Tiered storage

A small fast backend placed in front of a large slow one, for example
`MemoryKVCache` in front of `DiskKVCache`. `TieredStorage` is write-through:
writes try the slow tier first and then the fast tier, subject to each
tier's admission limits. Slow hits are promoted when the fast tier admits
them. A rejected slow overwrite preserves both existing copies; a rejected
fast overwrite invalidates the stale fast subtree. Only writes admitted
to the disk tier survive a restart.

See: [`tiered.py`](../src/nano_lmcache/storage/tiered.py), [Tiered storage](architecture.md#tiered-storage)

## Frequently Asked Questions

### 1. Why cache KV data instead of recomputing it?

Computing KV data for a long prompt is expensive, and many requests start with the same tokens. Reusing the stored KV of a shared prefix avoids repeating that work. It pays off only when loading the data costs less than the computation it replaces (see question 4).

See: [Real KV tensors](architecture.md#real-kv-tensors)

### 2. Why not cache by full prompt?

A full-prompt cache hits only when the whole input repeats exactly. In practice requests often share a long beginning, such as a system prompt or a document, and differ at the end. Chunked prefix keys let the shared beginning hit even when the endings differ.

See: [How a chunk key is built](architecture.md#how-a-chunk-key-is-built)

### 3. Why can only prefixes be reused?

In a causal model, the K and V of a token depend only on that token and the tokens before it. Identical leading tokens therefore have identical K and V, so reuse is exact, not approximate. The same tokens after a different prefix do not, which is why each chunk key hashes in the previous key.

The tests check both sides: reused-prefix logits match a from-scratch forward pass, and the same tokens after a different prefix have different K and V.

See: [`test_hf_reuse.py`](../tests/tensor/test_hf_reuse.py), [Real KV tensors](architecture.md#real-kv-tensors)

### 4. Why is reuse sometimes slower than recomputing?

Loading is not free. Looking up chunks, copying them between memory tiers, and rebuilding the model's cache from chunk tensors all take time, and for a short prefix that can exceed the cost of recomputing it.

The [nano-vLLM benchmark](nanovllm-benchmark.md) found that pinned CPU L2
became at least 10% faster than recompute at 4,096 reused tokens for
Qwen3-0.6B on one RTX 4090. Disk L2 did not beat recompute in that experiment,
through 8,192 reused tokens. These thresholds depend on the model, storage,
hardware, and measurement method; they are not general guarantees.

See: [`demo_kv_reuse.py`](../examples/demo_kv_reuse.py), [TTFT results and limits](nanovllm-benchmark.md)

### 5. What happens when the cache is full?

The cache must make room or reject the new entry. nano-lmcache uses prefix-safe eviction: it evicts the least recently used leaf, so no stored chunk is left without its ancestors. After an eviction, a request usually still hits its prefix and recomputes only the lost tail.

See: [Capacity and eviction](architecture.md#capacity-and-eviction)

### 6. What does the hit rate tell me?

How much of the reusable work was served from the cache, counted in chunks. It does not measure speed (see question 4). Read it together with the eviction counters in `StorageStats`: a falling hit rate with few evictions suggests prompts stopped sharing prefixes, while rising evictions suggest the working set no longer fits.

See: [`metrics.py`](../src/nano_lmcache/core/metrics.py), [`stats.py`](../src/nano_lmcache/storage/stats.py)

### 7. Why does a short prompt never hit the cache?

Only complete chunks are stored and matched. A prompt shorter than one `chunk_size` has no complete chunk, so nothing is stored for it and nothing can be matched.

See: [Core cache concepts](architecture.md#nano-core-cache-concepts)

### 8. How is nano-lmcache different from LMCache?

[LMCache](https://github.com/LMCache/LMCache) is a production KV cache system. nano-lmcache is a teaching project: it is not a reimplementation or a replacement of LMCache. It rebuilds the core ideas one concept at a time, with tests and runnable examples, and does not aim for production performance, API compatibility, or every LMCache feature.

See: [ROADMAP.md](../ROADMAP.md)

### 9. Do I need a GPU?

No, not for the cache concepts. `core/` and `storage/` are pure Python with no runtime dependencies, so chunking, prefix matching, metrics, eviction, and the memory, disk, and tiered backends run on any computer.

Real tensor operations need PyTorch; the Hugging Face model demo additionally
needs transformers (`uv sync --extra hf`) and can run on CPU. CUDA offload
and real nano-vLLM generation/trace recording need an NVIDIA GPU; the latter
also needs Linux and the `nanovllm` dependency group. Reading a saved engine
trace and producing offline HTML need only the core package.

See: [README](../README.md), [CONTRIBUTING.md](../CONTRIBUTING.md)

## Further reading

- [Architecture](architecture.md)
- [Roadmap](../ROADMAP.md)
- [README](../README.md)
