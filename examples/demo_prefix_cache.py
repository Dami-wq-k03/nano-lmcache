"""Runnable walkthrough of chunked prefix caching and its hit/miss metrics.

Run with: uv run python examples/demo_prefix_cache.py
"""

from nano_lmcache.core.chunking import chunk_tokens
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache

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

    # 命中的 chunk 直接复用缓存里的 payload，只为剩下的 chunk 付计算代价。
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


def main() -> None:
    cache = ChunkedPrefixCache(chunk_size=CHUNK_SIZE, namespace=NAMESPACE)

    print("=== Nano-LMCache: Chunked Prefix Cache Demo ===\n")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Namespace: {NAMESPACE}\n")

    requests = [
        # 冷启动：缓存是空的，每个 chunk 都要算。
        ("Request A (cold cache)", [101, 205, 307, 409, 511, 613]),
        # 与 A 共享前 4 个 token，第 3 个 chunk 开始分叉。
        ("Request B (shares 4-token prefix)", [101, 205, 307, 409, 777, 888]),
        # 与 A 完全相同：整段前缀都能复用。
        ("Request A again (fully warm)", [101, 205, 307, 409, 511, 613]),
        # 奇数长度：尾部不足一个 chunk，既不复用也不计入命中率。
        ("Request C (incomplete tail)", [101, 205, 307, 409, 511]),
    ]

    for label, token_ids in requests:
        serve_request(cache, label, token_ids)

    metrics = cache.metrics
    print("=== Cumulative metrics ===")
    print(f"Hit chunks:  {metrics.hit_chunks}")
    print(f"Miss chunks: {metrics.miss_chunks}")
    print(f"Hit tokens:  {metrics.hit_tokens}")
    print(f"Hit rate:    {metrics.hit_rate:.1%}")


if __name__ == "__main__":
    main()
