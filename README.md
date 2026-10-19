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

## Roadmap

1. **Nano**: Cache keys, an in-memory backend, and basic cache lookup.
2. **Mini**: Chunked prefix reuse, storage abstractions, and cache metrics.
3. **Integration**: Real KV tensors and inference-engine integration.

## Scope

The initial implementation prioritizes clarity and testability over
production performance. Distributed caching, GPU-specific optimizations,
and inference-engine integration are introduced only when the core
concepts are established.
