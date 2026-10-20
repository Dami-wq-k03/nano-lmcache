import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage


def build_chain(cache: MemoryKVCache, names: list[str]) -> None:
    """Store names as a prefix chain: names[0] <- names[1] <- ..."""
    parent: str | None = None
    for name in names:
        cache.put(name, f"v-{name}", parent_key=parent)
        parent = name


# ---- capacity configuration -------------------------------------------


def test_unbounded_by_default():
    cache = MemoryKVCache()

    for i in range(100):
        cache.put(f"k{i}", i)

    assert len(cache) == 100
    assert cache.max_entries is None


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_capacity(bad):
    with pytest.raises(ValueError):
        MemoryKVCache(max_entries=bad)


def test_bounded_cache_still_satisfies_protocol():
    assert isinstance(MemoryKVCache(max_entries=2), CacheStorage)


# ---- eviction order ---------------------------------------------------


def test_independent_entries_evict_least_recently_used():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)

    cache.put("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_get_refreshes_recency():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")

    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_overwrite_does_not_evict_or_grow():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)

    cache.put("a", 10)

    assert len(cache) == 2
    assert cache.get("a") == 10
    assert cache.get("b") == 2


# ---- prefix safety ----------------------------------------------------


def test_never_evicts_an_entry_that_has_children():
    cache = MemoryKVCache(max_entries=3)
    build_chain(cache, ["a", "b", "c"])
    # "x" is an independent root; a and b have descendants and must stay.
    cache.put("x", 0)

    assert cache.get("a") is not None
    assert cache.get("b") is not None
    assert cache.get("c") is None
    assert cache.get("x") == 0


def test_ancestor_becomes_evictable_once_children_are_gone():
    cache = MemoryKVCache(max_entries=2)
    build_chain(cache, ["a", "b"])

    cache.put("x", 0)  # evicts leaf b
    cache.put("y", 0)  # a is now a leaf and the least recently used

    assert cache.get("a") is None
    assert cache.get("x") == 0
    assert cache.get("y") == 0


def test_chain_longer_than_capacity_keeps_longest_fitting_prefix():
    cache = MemoryKVCache(max_entries=3)

    build_chain(cache, ["a", "b", "c", "d", "e"])

    # A new entry's insert cannot evict its own parent, so the suffix that
    # does not fit is dropped.
    assert len(cache) == 3
    assert [cache.get(k) for k in "abc"] == ["v-a", "v-b", "v-c"]
    assert cache.get("d") is None
    assert cache.get("e") is None


def test_child_with_missing_parent_is_not_stored():
    cache = MemoryKVCache(max_entries=3)

    cache.put("orphan", 1, parent_key="never-stored")

    assert len(cache) == 0
    assert cache.get("orphan") is None


def test_no_orphan_suffix_after_many_inserts():
    cache = MemoryKVCache(max_entries=4)
    parents: dict[str, str | None] = {}
    for root in range(6):
        parent: str | None = None
        for depth in range(3):
            key = f"{root}-{depth}"
            cache.put(key, key, parent_key=parent)
            parents[key] = parent
            parent = key

    for key, parent in parents.items():
        if cache.get(key) is not None and parent is not None:
            assert cache.get(parent) is not None


# ---- through ChunkedPrefixCache --------------------------------------


def make_prefix_cache(max_entries: int | None) -> ChunkedPrefixCache:
    return ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=MemoryKVCache(max_entries=max_entries),
    )


def test_prefix_cache_without_capacity_behaves_as_before():
    cache = make_prefix_cache(None)
    tokens = [1, 2, 3, 4, 5, 6]
    cache.store(tokens, ["kv-0", "kv-1", "kv-2"])

    assert cache.lookup(tokens).matched_chunks == 3


def test_capacity_smaller_than_prompt_keeps_a_usable_prefix():
    cache = make_prefix_cache(2)
    tokens = [1, 2, 3, 4, 5, 6]

    cache.store(tokens, ["kv-0", "kv-1", "kv-2"])
    result = cache.lookup(tokens)

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4
    assert result.values == ("kv-0", "kv-1")


def test_eviction_never_breaks_shared_prefix_of_other_request():
    cache = make_prefix_cache(3)
    cache.store([1, 2, 3, 4], ["a0", "a1"])
    cache.store([1, 2, 7, 8], ["a0", "b1"])  # shares the first chunk with A

    # An unrelated request must push out a leaf, not the shared [1, 2].
    cache.store([9, 9, 8, 8], ["c0", "c1"])

    assert cache.lookup([1, 2, 3, 4]).matched_chunks >= 1
    assert cache.lookup([1, 2, 7, 8]).matched_chunks >= 1


def test_metrics_after_eviction_count_lost_suffix_as_miss():
    cache = make_prefix_cache(2)
    tokens = [1, 2, 3, 4, 5, 6]
    cache.store(tokens, ["kv-0", "kv-1", "kv-2"])

    cache.lookup(tokens)

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 1
    assert cache.metrics.hit_tokens == 4


def test_evicted_request_can_be_restored_by_storing_again():
    cache = make_prefix_cache(2)
    cache.store([1, 2, 3, 4], ["a0", "a1"])
    cache.store([9, 9, 8, 8], ["c0", "c1"])  # evicts a1, then a0

    assert cache.lookup([1, 2, 3, 4]).matched_chunks == 0

    cache.store([1, 2, 3, 4], ["a0", "a1"])

    assert cache.lookup([1, 2, 3, 4]).matched_chunks == 2
