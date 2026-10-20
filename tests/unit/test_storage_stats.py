from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.stats import StorageStats


def test_new_stats_start_at_zero():
    stats = StorageStats()

    assert stats.evictions == 0
    assert stats.rejected_no_room == 0
    assert stats.rejected_missing_parent == 0
    assert stats.rejected == 0


def test_rejected_is_the_sum_of_both_reasons():
    stats = StorageStats(rejected_no_room=2, rejected_missing_parent=3)

    assert stats.rejected == 5


def test_unbounded_cache_never_evicts_or_rejects():
    cache = MemoryKVCache()

    for i in range(50):
        cache.put(f"k{i}", i)

    assert cache.stats == StorageStats()


def test_eviction_is_counted_when_writes_exceed_capacity():
    cache = MemoryKVCache(max_entries=2)

    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.stats.evictions == 0

    cache.put("c", 3)
    cache.put("d", 4)

    assert cache.stats.evictions == 2
    assert cache.stats.rejected == 0


def test_overwrite_is_not_an_eviction():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)

    cache.put("a", 10)

    assert cache.stats.evictions == 0


def test_suffix_that_does_not_fit_is_counted_as_no_room():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2, parent_key="a")

    # The only leaf is b, which is the new entry's parent, so nothing can go.
    cache.put("c", 3, parent_key="b")

    assert cache.stats.rejected_no_room == 1
    assert cache.stats.rejected_missing_parent == 0
    assert cache.stats.evictions == 0


def test_entry_with_missing_parent_is_counted_separately():
    cache = MemoryKVCache(max_entries=2)

    cache.put("orphan", 1, parent_key="never-stored")

    assert cache.stats.rejected_missing_parent == 1
    assert cache.stats.rejected_no_room == 0


def test_missing_parent_is_counted_even_when_unbounded():
    cache = MemoryKVCache()

    cache.put("orphan", 1, parent_key="never-stored")

    assert cache.stats.rejected_missing_parent == 1


def test_long_prompt_counts_every_dropped_chunk():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=MemoryKVCache(max_entries=2),
    )

    # Chunk 2 does not fit (no room); chunks 3 and 4 then have no parent.
    cache.store(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ["kv-0", "kv-1", "kv-2", "kv-3", "kv-4"],
    )

    storage = cache.storage
    assert isinstance(storage, MemoryKVCache)
    assert storage.stats.rejected_no_room == 1
    assert storage.stats.rejected_missing_parent == 2
    assert storage.stats.evictions == 0


def test_stats_are_independent_of_lookup_metrics():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=MemoryKVCache(max_entries=1),
    )
    cache.store([1, 2], ["kv-0"])
    cache.store([3, 4], ["kv-1"])

    cache.lookup([3, 4])

    storage = cache.storage
    assert isinstance(storage, MemoryKVCache)
    assert storage.stats.evictions == 1
    assert cache.metrics.hit_chunks == 1
