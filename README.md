# nano-lmcache

An educational implementation of LMCache concepts.

nano-lmcache is a small, incremental implementation designed to help
developers understand the core ideas behind LMCache and KV cache reuse.

## Goals

- Understand KV cache reuse and prefix matching.
- Implement cache keying and storage abstractions.
- Explore in-memory caching before introducing more complex backends.
- Progress toward integration with real LLM inference engines.

## Development

Requirements: Python 3.11+ and uv.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/nano_lmcache
```

## Try it

```bash
uv run python examples/demo_chunking.py       # chunk keys in isolation
uv run python examples/demo_prefix_cache.py   # full cache loop + metrics
```

The prefix-cache demo serves four requests against one cache and prints
per-chunk hit/miss decisions plus cumulative reuse metrics.

## Roadmap

1. **Nano** (current): Token chunking, prefix-aware chunk keys, an
   in-memory backend, continuous-prefix lookup, and hit/miss metrics.
2. **Mini**: A storage interface so cache logic no longer depends on
   `MemoryKVCache` directly, plus richer eviction/cache behavior.
3. **Integration**: Real KV tensors, GPU or remote backends, and
   inference-engine integration such as vLLM.

## Scope

The initial implementation prioritizes clarity and testability over
production performance. Distributed caching, GPU-specific optimizations,
and inference-engine integration are introduced only when the core
concepts are established.
