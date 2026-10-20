"""The same prefix-cache behavior, checked against every storage backend.

A backend is only interchangeable if ChunkedPrefixCache behaves the same
on top of it. Add new backends to BACKENDS.
"""

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache

BACKENDS = {
    "memory": lambda tmp_path, max_entries: MemoryKVCache(max_entries=max_entries),
    "disk": lambda tmp_path, max_entries: DiskKVCache(
        tmp_path / "disk", max_entries=max_entries
    ),
}


@pytest.fixture(params=sorted(BACKENDS))
def make_cache(request, tmp_path):
    def factory(max_entries=None):
        storage = BACKENDS[request.param](tmp_path, max_entries)
        return ChunkedPrefixCache(
            chunk_size=2, namespace="model-a", storage=storage
        )

    return factory


def test_store_then_lookup_returns_all_chunks(make_cache):
    cache = make_cache()
    tokens = [101, 205, 307, 409]
    cache.store(tokens, ["kv-0", "kv-1"])

    result = cache.lookup(tokens)

    assert result.matched_chunks == 2
    assert result.values == ("kv-0", "kv-1")


def test_lookup_returns_longest_matching_prefix(make_cache):
    cache = make_cache()
    cache.store([101, 205, 307, 409, 511, 613], ["kv-0", "kv-1", "kv-2"])

    result = cache.lookup([101, 205, 307, 409, 777, 888])

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4


def test_different_first_chunk_misses(make_cache):
    cache = make_cache()
    cache.store([101, 205, 307, 409], ["kv-0", "kv-1"])

    assert cache.lookup([999, 205, 307, 409]).matched_chunks == 0


def test_incomplete_tail_is_not_reused(make_cache):
    cache = make_cache()
    cache.store([101, 205, 307, 409, 511], ["kv-0", "kv-1"])

    result = cache.lookup([101, 205, 307, 409, 511])

    assert result.matched_tokens == 4


def test_metrics_count_remaining_chunks_as_misses(make_cache):
    cache = make_cache()
    cache.store([1, 2, 3, 4, 5, 6], ["kv-0", "kv-1", "kv-2"])

    cache.lookup([1, 2, 3, 4, 7, 8])

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 1


def test_capacity_keeps_a_usable_prefix(make_cache):
    cache = make_cache(max_entries=2)
    tokens = [1, 2, 3, 4, 5, 6]

    cache.store(tokens, ["kv-0", "kv-1", "kv-2"])

    assert cache.lookup(tokens).matched_chunks == 2


def test_eviction_never_breaks_a_shared_prefix(make_cache):
    cache = make_cache(max_entries=3)
    cache.store([1, 2, 3, 4], ["a0", "a1"])
    cache.store([1, 2, 7, 8], ["a0", "b1"])

    cache.store([9, 9, 8, 8], ["c0", "c1"])

    assert cache.lookup([1, 2, 3, 4]).matched_chunks >= 1
    assert cache.lookup([1, 2, 7, 8]).matched_chunks >= 1
