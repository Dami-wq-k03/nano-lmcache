"""Runnable walkthrough of chunked prefix caching and its hit/miss metrics.

Run with: uv run python examples/demo_prefix_cache.py
"""

from nano_lmcache.core.chunking import chunk_tokens
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache

CHUNK_SIZE = 2
NAMESPACE = "demo-model"


def compute_kv_payload(chunk: tuple[int, ...]) -> str:
    """Stand-in for a real prefill pass over one chunk.

    A real engine would return KV tensors here; a string keeps the demo
    dependency-free while still being something we can store and reuse.
    """
    return "kv:" + "-".join(str(token) for token in chunk)


def serve_request(
    cache: ChunkedPrefixCache,
    label: str,
    token_ids: list[int],
) -> None:
    """Serve one request the way an engine would: look up, then fill gaps."""
    chunks = chunk_tokens(token_ids, CHUNK_SIZE)
    complete = [chunk for chunk in chunks if len(chunk) == CHUNK_SIZE]

    result = cache.lookup(token_ids)

    print(f"--- {label} ---")
    print(f"Tokens: {token_ids}")

    for index, chunk in enumerate(chunks):
        if len(chunk) < CHUNK_SIZE:
            print(f"  chunk {index} {chunk} SKIP  incomplete tail")
        elif index < result.matched_chunks:
            key = result.keys[index]
            print(f"  chunk {index} {chunk} HIT   reused {key[:12]}...")
        else:
            print(f"  chunk {index} {chunk} MISS  recomputed")

    # Reuse cached payloads for hit chunks; pay compute only for the rest.
    reused = list(result.values)
    recomputed = [
        compute_kv_payload(chunk)
        for chunk in complete[result.matched_chunks :]
    ]
    cache.store(token_ids, reused + recomputed)

    print(
        f"  reused {result.matched_chunks}/{len(complete)} chunks, "
        f"{result.matched_tokens}/{len(token_ids)} tokens"
    )
    print()


def print_metrics(cache: ChunkedPrefixCache) -> None:
    metrics = cache.metrics
    print("=== Cumulative metrics ===")
    print(f"Hit chunks:  {metrics.hit_chunks}")
    print(f"Miss chunks: {metrics.miss_chunks}")
    print(f"Hit tokens:  {metrics.hit_tokens}")
    print(f"Hit rate:    {metrics.hit_rate:.1%}\n")


def run_unbounded() -> None:
    cache = ChunkedPrefixCache(chunk_size=CHUNK_SIZE, namespace=NAMESPACE)

    print("=== Part 1: unbounded cache ===\n")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Namespace: {NAMESPACE}\n")

    requests = [
        # Cold start: the cache is empty, so every chunk is computed.
        ("Request A (cold cache)", [101, 205, 307, 409, 511, 613]),
        # Shares A's first 4 tokens and diverges at the 3rd chunk.
        ("Request B (shares 4-token prefix)", [101, 205, 307, 409, 777, 888]),
        # Identical to A: the whole prefix is reusable.
        ("Request A again (fully warm)", [101, 205, 307, 409, 511, 613]),
        # Odd length: the short tail is neither reused nor counted in hit rate.
        ("Request C (incomplete tail)", [101, 205, 307, 409, 511]),
    ]

    for label, token_ids in requests:
        serve_request(cache, label, token_ids)

    print_metrics(cache)


def run_bounded() -> None:
    storage = MemoryKVCache(max_entries=4)
    cache = ChunkedPrefixCache(
        chunk_size=CHUNK_SIZE, namespace=NAMESPACE, storage=storage
    )

    print("=== Part 2: bounded cache (max_entries=4) ===\n")

    requests = [
        # 3 chunks fill 3 of the 4 slots.
        ("Request A (fills 3 of 4 slots)", [101, 205, 307, 409, 511, 613]),
        # Shares A's first chunk and adds 2 more, evicting one leaf (A's tail).
        ("Request B (forces eviction)", [101, 205, 888, 777, 666, 555]),
        # A's tail was evicted, but the shared [101, 205] and A's middle remain.
        ("Request A again (suffix evicted)", [101, 205, 307, 409, 511, 613]),
    ]

    for label, token_ids in requests:
        serve_request(cache, label, token_ids)
        print(f"  entries in cache: {len(storage)}/{storage.max_entries}\n")

    print_metrics(cache)

    stats = storage.stats
    print("=== Storage stats ===")
    print(f"Evictions:                 {stats.evictions}")
    print(f"Rejected (no room):        {stats.rejected_no_room}")
    print(f"Rejected (missing parent): {stats.rejected_missing_parent}\n")


def main() -> None:
    print("=== Nano-LMCache: Chunked Prefix Cache Demo ===\n")
    run_unbounded()
    run_bounded()


if __name__ == "__main__":
    main()
