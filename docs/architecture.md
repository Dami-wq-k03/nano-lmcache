# Architecture

## Nano: Core cache concepts

The initial implementation explores a minimal cache lifecycle:

1. Receive a sequence of token IDs.
2. Derive a deterministic cache key.
3. Look up the corresponding cached state.
4. Report a cache hit or miss.
5. Store the state in an in-memory backend.

The first version uses pure Python so that cache behavior can be
understood without a GPU or an inference engine.

## Mini: Extend the cache model

The next stage introduces chunked prefix matching, partial prefix reuse,
explicit storage interfaces, and basic metrics.

## Inference integration

A later stage explores KV tensor handling and integration with an
inference engine such as vLLM.

The core package should remain independently understandable and
testable. Integration-specific dependencies should be introduced only
when required.

## Design principle

Each stage should add one meaningful concept, with tests and runnable
examples demonstrating its behavior.
