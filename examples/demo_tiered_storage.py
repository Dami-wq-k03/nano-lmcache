"""Show a small memory tier in front of a disk tier.

    uv run python examples/demo_tiered_storage.py

Memory holds only 4 chunks; disk holds everything (write-through). Watch
where each lookup is served from: memory, disk (then promoted), or neither.
The last part simulates a restart: memory starts empty, disk does not.
"""

import shutil
import tempfile
from pathlib import Path

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.tiered import TieredStorage

CHUNK_SIZE = 2
MEMORY_ENTRIES = 4


def open_cache(directory: Path) -> tuple[ChunkedPrefixCache, TieredStorage]:
    storage = TieredStorage(
        fast=MemoryKVCache(max_entries=MEMORY_ENTRIES),
        slow=DiskKVCache(directory),
    )
    cache = ChunkedPrefixCache(
        chunk_size=CHUNK_SIZE, namespace="demo-model", storage=storage
    )
    return cache, storage


def serve(
    cache: ChunkedPrefixCache,
    storage: TieredStorage,
    label: str,
    tokens: list[int],
) -> None:
    before = storage.tier_stats
    fast, slow = before.fast_hits, before.slow_hits

    result = cache.lookup(tokens)
    complete = len(tokens) // CHUNK_SIZE
    payloads = list(result.values) + [
        f"kv-{tokens[i * CHUNK_SIZE]}" for i in range(result.matched_chunks, complete)
    ]
    cache.store(tokens, payloads)

    stats = storage.tier_stats
    print(
        f"{label:<34} reused {result.matched_chunks}/{complete} chunks "
        f"(memory {stats.fast_hits - fast}, disk {stats.slow_hits - slow})"
    )


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="nano-lmcache-tiered-"))
    try:
        print(f"Memory tier: {MEMORY_ENTRIES} entries. Disk tier: unbounded.\n")
        cache, storage = open_cache(directory)

        document = [10, 11, 12, 13, 14, 15]
        serve(cache, storage, "Doc + question 1 (cold)", document + [20, 21])
        serve(cache, storage, "Doc + question 2 (warm)", document + [30, 31])
        serve(cache, storage, "Unrelated request (pushes memory)", [90, 91, 92, 93])
        serve(cache, storage, "Doc + question 1 again", document + [20, 21])

        print("\n--- restart: new objects, same disk directory ---\n")
        cache, storage = open_cache(directory)
        serve(cache, storage, "Doc + question 1 after restart", document + [20, 21])
        serve(cache, storage, "Doc + question 1 once more", document + [20, 21])

        stats = storage.tier_stats
        print(
            f"\nAfter restart: hit rate {stats.hit_rate:.0%}, "
            f"served from memory {stats.fast_hit_rate:.0%}, "
            f"{stats.promotions} chunks promoted from disk"
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
