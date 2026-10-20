import random
import sys

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.stats import StorageStats


def payload_size(value: object) -> int:
    assert isinstance(value, (bytes, bytearray))
    return len(value)


def test_default_size_is_shallow_python_object_size():
    value = b"payload"
    cache = MemoryKVCache(max_bytes=sys.getsizeof(value))
    cache.put("a", value)

    assert cache.get("a") == value
    assert cache.nbytes == cache.max_bytes == sys.getsizeof(value)


def test_unbounded_cache_tracks_injected_sizes():
    cache = MemoryKVCache(size_fn=payload_size)
    cache.put("a", b"abc")
    cache.put("b", b"de", parent_key="a")

    assert cache.max_bytes is None
    assert cache.nbytes == 5


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_invalid_byte_capacity(bad):
    with pytest.raises(ValueError, match="max_bytes"):
        MemoryKVCache(max_bytes=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_invalid_size_leaves_values_and_recency_unchanged(bad):
    def size_fn(value):
        return bad if value == b"bad" else len(value)

    cache = MemoryKVCache(max_bytes=2, size_fn=size_fn)
    cache.put("a", b"a")
    cache.put("b", b"b")
    with pytest.raises(ValueError, match="size_bytes"):
        cache.put("a", b"bad")

    assert cache.nbytes == 2
    assert cache.stats == StorageStats()
    cache.put("c", b"c")
    assert "a" not in cache
    assert cache.get("b") == b"b"


def test_size_function_exception_does_not_evict_or_replace():
    def size_fn(value):
        if value == b"bad":
            raise RuntimeError("cannot measure")
        return len(value)

    cache = MemoryKVCache(max_bytes=2, size_fn=size_fn)
    cache.put("a", b"aa")
    with pytest.raises(RuntimeError, match="cannot measure"):
        cache.put("a", b"bad")

    assert cache.get("a") == b"aa"
    assert cache.nbytes == 2
    assert cache.stats == StorageStats()


def test_large_write_evicts_multiple_lru_leaves():
    cache = MemoryKVCache(max_bytes=6, size_fn=payload_size)
    for key in "abc":
        cache.put(key, b"xx")
    cache.get("a")

    cache.put("d", b"dddd")

    assert cache.get("a") == b"xx"
    assert "b" not in cache and "c" not in cache
    assert cache.get("d") == b"dddd"
    assert cache.nbytes == 6
    assert cache.stats.evictions == 2


@pytest.mark.parametrize(
    ("max_entries", "max_bytes", "sizes", "survivors"),
    [(2, 10, [1, 1, 1], "bc"), (4, 5, [2, 2, 4], "c"), (2, 5, [1, 1, 5], "c")],
)
def test_both_capacity_limits_apply(max_entries, max_bytes, sizes, survivors):
    cache = MemoryKVCache(
        max_entries=max_entries, max_bytes=max_bytes, size_fn=payload_size
    )
    for key, size in zip("abc", sizes, strict=True):
        cache.put(key, b"x" * size)

    assert {key for key in "abc" if key in cache} == set(survivors)
    assert len(cache) <= max_entries
    assert cache.nbytes <= max_bytes


@pytest.mark.parametrize("key", ["a", "new"])
def test_oversized_write_preserves_existing_data_and_recency(key):
    cache = MemoryKVCache(max_bytes=4, size_fn=payload_size)
    cache.put("a", b"aa")
    cache.put("b", b"bb")
    cache.put(key, b"xxxxx")

    assert len(cache) == 2
    assert cache.nbytes == 4
    assert cache.stats == StorageStats(rejected_no_room=1)
    cache.put("c", b"cc")
    assert "a" not in cache
    assert cache.get("b") == b"bb"


def test_prefix_that_cannot_fit_rejects_without_unrelated_evictions():
    cache = MemoryKVCache(max_bytes=6, size_fn=payload_size)
    cache.put("other", b"x")
    cache.put("a", b"aaa")
    cache.put("b", b"bb", parent_key="a")
    cache.put("c", b"cc", parent_key="b")

    assert cache.get("other") == b"x"
    assert cache.get("a") == b"aaa"
    assert cache.get("b") == b"bb"
    assert "c" not in cache
    assert cache.nbytes == 6
    assert cache.stats == StorageStats(rejected_no_room=1)
    cache.put("d", b"d", parent_key="c")
    assert cache.stats.rejected_missing_parent == 1


def test_multiple_evictions_remove_descendants_before_ancestors():
    cache = MemoryKVCache(max_bytes=6, size_fn=payload_size)
    cache.put("a", b"aa")
    cache.put("b", b"bb", parent_key="a")
    cache.put("c", b"cc", parent_key="b")
    cache.put("x", b"xxxx")

    assert cache.get("a") == b"aa"
    assert cache.get("x") == b"xxxx"
    assert "b" not in cache and "c" not in cache
    assert cache.nbytes == 6
    assert cache.stats.evictions == 2


def test_growing_overwrite_can_evict_descendants_but_keeps_chain_position():
    cache = MemoryKVCache(max_bytes=6, size_fn=payload_size)
    cache.put("a", b"aa")
    cache.put("b", b"bb", parent_key="a")
    cache.put("c", b"cc", parent_key="b")
    cache.put("b", b"bbbb", parent_key="missing")

    assert cache.get("a") == b"aa"
    assert cache.get("b") == b"bbbb"
    assert cache.parent_of("b") == "a"
    assert "c" not in cache
    assert cache.nbytes == 6
    assert cache.stats.evictions == 1


def test_incoming_parent_stays_protected_after_its_old_child_is_evicted():
    cache = MemoryKVCache(max_bytes=6, size_fn=payload_size)
    cache.put("a", b"aa")
    cache.put("b", b"bb", parent_key="a")
    cache.put("other", b"xx")
    cache.put("c", b"cccc", parent_key="a")

    assert cache.get("a") == b"aa"
    assert cache.get("c") == b"cccc"
    assert cache.parent_of("c") == "a"
    assert "b" not in cache and "other" not in cache
    assert cache.nbytes == 6
    assert cache.stats.evictions == 2


def test_growing_leaf_overwrite_never_evicts_itself():
    cache = MemoryKVCache(max_bytes=5, size_fn=payload_size)
    cache.put("a", b"a")
    cache.put("b", b"b")
    cache.put("a", b"aaaa")

    assert cache.get("a") == b"aaaa"
    assert cache.get("b") == b"b"
    assert cache.nbytes == 5
    assert cache.stats.evictions == 0


def test_overwrite_that_cannot_fit_ancestors_preserves_old_payload():
    cache = MemoryKVCache(max_bytes=5, size_fn=payload_size)
    cache.put("other", b"x")
    cache.put("a", b"aaa")
    cache.put("b", b"b", parent_key="a")
    cache.put("b", b"bbb")

    assert cache.get("other") == b"x"
    assert cache.get("b") == b"b"
    assert cache.parent_of("b") == "a"
    assert cache.nbytes == 5
    assert cache.stats == StorageStats(rejected_no_room=1)


def test_shrinking_overwrite_releases_bytes_for_next_insert():
    cache = MemoryKVCache(max_bytes=5, size_fn=payload_size)
    cache.put("a", b"aaaa")
    cache.put("a", b"a")
    assert cache.nbytes == 1
    cache.put("b", b"bbbb", parent_key="a")

    assert cache.nbytes == 5
    assert cache.stats.evictions == 0


def test_zero_size_payload_still_obeys_entry_limit():
    cache = MemoryKVCache(max_entries=1, max_bytes=1, size_fn=payload_size)
    cache.put("a", b"")
    cache.put("b", b"")

    assert "a" not in cache
    assert cache.get("b") == b""
    assert cache.nbytes == 0


def test_mutable_payload_is_measured_again_only_on_put():
    cache = MemoryKVCache(max_bytes=4, size_fn=payload_size)
    value = bytearray(b"a")
    cache.put("a", value)
    value.extend(b"bc")
    assert cache.nbytes == 1
    cache.put("a", value)
    assert cache.nbytes == 3


def test_chunked_prefix_cache_keeps_longest_prefix_that_fits_in_bytes():
    storage = MemoryKVCache(max_bytes=5, size_fn=payload_size)
    cache = ChunkedPrefixCache(chunk_size=2, namespace="bytes", storage=storage)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.store(tokens, [b"aaa", b"bb", b"c", b"d"])

    result = cache.lookup(tokens)
    assert result.matched_chunks == 2
    assert result.matched_tokens == 4
    assert result.values == (b"aaa", b"bb")
    assert storage.nbytes == 5
    assert storage.stats == StorageStats(rejected_no_room=1, rejected_missing_parent=1)


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_writes_preserve_all_ancestors_and_capacity(seed):
    rng = random.Random(seed)
    cache = MemoryKVCache(max_entries=8, max_bytes=24, size_fn=payload_size)
    parents = {
        f"{root}-{depth}": None if depth == 0 else f"{root}-{depth - 1}"
        for root in range(4)
        for depth in range(5)
    }
    keys = list(parents)
    for _ in range(400):
        key = rng.choice(keys)
        cache.put(key, b"x" * rng.randrange(30), parent_key=parents[key])
        cache.get(rng.choice(keys))

        stored = [key for key in keys if key in cache]
        assert len(stored) == len(cache) <= 8
        assert cache.nbytes == sum(payload_size(cache.get(key)) for key in stored)
        assert 0 <= cache.nbytes <= 24
        for key in stored:
            parent = cache.parent_of(key)
            while parent is not None:
                assert parent in cache
                parent = cache.parent_of(parent)
