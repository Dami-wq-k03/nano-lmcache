"""Tests for pluggable leaf eviction policies."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path

import pytest

from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.eviction_policies import (
    LFUEvictionPolicy,
    LRUEvictionPolicy,
)
from nano_lmcache.storage.memory import MemoryKVCache


def test_default_policy_is_lru_and_keeps_existing_order() -> None:
    cache = MemoryKVCache(max_entries=2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)

    assert cache.eviction_policy == "lru"
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_lfu_evicts_low_frequency_leaf_before_hot_leaf() -> None:
    cache = MemoryKVCache(max_entries=3, eviction_policy="lfu")
    for key in ("a", "b", "c"):
        cache.put(key, key)
    for _ in range(5):
        assert cache.get("a") == "a"

    cache.put("d", "d")

    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache
    assert cache._index.frequency_of("a") == 6


def test_lfu_uses_recent_access_as_tie_breaker() -> None:
    cache = MemoryKVCache(max_entries=2, eviction_policy="lfu")
    cache.put("a", "a")
    cache.put("b", "b")
    # Both become frequency 2; b is the more recent of the two.
    cache.get("a")
    cache.get("b")

    cache.put("c", "c")

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


def test_lfu_uses_key_order_when_frequency_and_time_match() -> None:
    index = PrefixLRUIndex(max_entries=2, eviction_policy="lfu")
    index.restore("b", None)
    index.restore("a", None)
    index.restore_access("b", 10)
    index.restore_access("a", 10)

    assert index.admit("c", None).evicted == ("a",)


def test_lfu_retains_internal_frequency_when_node_becomes_leaf() -> None:
    cache = MemoryKVCache(max_entries=4, eviction_policy="lfu")
    cache.put("a", "a")
    cache.put("b", "b", parent_key="a")
    cache.put("c", "c", parent_key="b")
    cache.put("x", "x")
    for _ in range(5):
        assert cache.get("b") == "b"

    # c is the only leaf eligible for this write and its removal exposes b.
    cache.put("y", "y")
    assert "c" not in cache
    assert cache._index.frequency_of("b") == 6
    assert cache.parent_of("b") == "a"

    # b is now a leaf, but its accumulated frequency still makes it safer
    # than the untouched leaves.
    cache.put("z", "z")
    assert "b" in cache


def test_lfu_exposed_leaf_uses_old_frequency_in_ranking() -> None:
    cache = MemoryKVCache(max_entries=4, eviction_policy="lfu")
    cache.put("a", "a")
    cache.put("b", "b", parent_key="a")
    cache.put("c", "c", parent_key="b")
    cache.put("x", "x")
    for _ in range(5):
        assert cache.get("x") == "x"

    cache.put("y", "y")  # removes c and exposes b at frequency 1
    cache.put("z", "z")

    assert "b" not in cache
    assert "x" in cache


def test_lfu_never_evicts_internal_or_protected_nodes() -> None:
    cache = MemoryKVCache(max_entries=3, eviction_policy="lfu")
    cache.put("a", "a")
    cache.put("b", "b", parent_key="a")
    cache.put("c", "c", parent_key="b")

    # The incoming parent c is protected, and a/b are internal.
    assert not cache.try_put("d", "d", parent_key="c")
    assert set(cache.keys()) == {"a", "b", "c"}


def test_lfu_rejects_an_impossible_chain_without_eviction() -> None:
    cache = MemoryKVCache(max_entries=2, eviction_policy="lfu")
    cache.put("root", 1)
    cache.put("child", 2, parent_key="root")
    before = tuple(cache.keys())

    assert not cache.try_put("grandchild", 3, parent_key="child")
    assert tuple(cache.keys()) == before
    assert set(cache.keys()) == {"root", "child"}


def test_repeated_lfu_hits_update_one_record_per_key() -> None:
    cache = MemoryKVCache(max_entries=4, eviction_policy="lfu")
    for key in "abcd":
        cache.put(key, key)
    for _ in range(10_000):
        assert cache.get("a") == "a"

    assert len(cache._index._entries) == len(cache) == 4
    assert cache.frequency_of("a") == 10_001
    cache.put("e", "e")
    assert "b" not in cache


def test_disk_recovery_resets_lfu_frequency_but_restores_recency(
    tmp_path: Path,
) -> None:
    first = DiskKVCache(tmp_path, max_entries=2, eviction_policy="lfu")
    first.put("a", "a")
    first.put("b", "b")
    for _ in range(5):
        assert first.get("a") == "a"
    assert first._index.frequency_of("a") == 6

    reopened = DiskKVCache(tmp_path, max_entries=2, eviction_policy="lfu")

    assert reopened._index.frequency_of("a") == 1
    assert reopened._index.frequency_of("b") == 1
    # The mtime replay is recency metadata only, so b (the older file) leaves.
    reopened.put("c", "c")
    assert "b" not in reopened
    assert "a" in reopened


def test_access_state_is_not_shared_between_caches() -> None:
    policy = LFUEvictionPolicy()
    first = MemoryKVCache(max_entries=2, eviction_policy=policy)
    second = MemoryKVCache(max_entries=2, eviction_policy=policy)

    first.put("a", 1)
    first.get("a")
    second.put("a", 1)

    assert not vars(policy)  # Sharing a stateless scorer cannot share access records.
    assert first._index.frequency_of("a") == 2
    assert second._index.frequency_of("a") == 1


def _assert_prefix_invariant(cache: MemoryKVCache) -> None:
    keys = set(cache.keys())
    assert keys == set(cache._values)
    assert len(cache) <= 6
    for key in keys:
        parent = cache.parent_of(key)
        while parent is not None:
            assert parent in keys
            parent = cache.parent_of(parent)


def _reference_step(
    parents: OrderedDict[str, str | None],
    frequencies: dict[str, int],
    last_access: dict[str, int],
    counter: list[int],
    key: str,
    parent: str | None,
    operation: str,
    policy: str,
) -> None:
    """Small independent model for the randomized entry-limit workload."""
    if operation == "get":
        if key in parents:
            parents.move_to_end(key)
            counter[0] += 1
            last_access[key] = counter[0]
            if policy == "lfu":
                frequencies[key] += 1
        return

    if operation == "invalidate":
        if key not in parents:
            return
        doomed = {key}
        changed = True
        while changed:
            changed = False
            for candidate, candidate_parent in parents.items():
                if candidate_parent in doomed and candidate not in doomed:
                    doomed.add(candidate)
                    changed = True
        for candidate in doomed:
            parents.pop(candidate)
            frequencies.pop(candidate)
            last_access.pop(candidate)
        return

    present = key in parents
    if present:
        parent = parents[key]
    elif parent is not None and parent not in parents:
        return

    if not present:
        protected = parent
        while len(parents) >= 6:
            leaves = [
                candidate
                for candidate in parents
                if candidate != protected and candidate not in parents.values()
            ]
            if not leaves:
                return
            if policy == "lru":
                victim = min(leaves, key=lambda item: (last_access[item], item))
            else:
                victim = min(
                    leaves,
                    key=lambda item: (
                        frequencies[item],
                        last_access[item],
                        item,
                    ),
                )
            parents.pop(victim)
            frequencies.pop(victim)
            last_access.pop(victim)
    else:
        parents.move_to_end(key)
        counter[0] += 1
        last_access[key] = counter[0]
        return

    parents[key] = parent
    counter[0] += 1
    frequencies[key] = 1
    last_access[key] = counter[0]


@pytest.mark.parametrize("policy", ["lru", "lfu"])
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_insert_touch_evict_reinsert_invalidate_preserves_invariants(
    policy: str, seed: int
) -> None:
    rng = random.Random(seed)
    cache = MemoryKVCache(max_entries=6, eviction_policy=policy)
    parents = OrderedDict[str, str | None]()
    frequencies: dict[str, int] = {}
    last_access: dict[str, int] = {}
    counter = [0]
    known_parents = {
        "r0": None,
        "r1": None,
        "a": "r0",
        "b": "a",
        "c": "b",
        "x": "r1",
        "y": "x",
    }

    for step in range(500):
        key = rng.choice(list(known_parents))
        roll = rng.random()
        if roll < 0.25:
            operation = "get"
        elif roll < 0.38:
            operation = "invalidate"
        else:
            operation = "put"
        parent = known_parents[key]

        _reference_step(
            parents,
            frequencies,
            last_access,
            counter,
            key,
            parent,
            operation,
            policy,
        )
        if operation == "get":
            cache.get(key)
        elif operation == "invalidate":
            cache.invalidate(key)
        else:
            cache.put(key, step, parent_key=parent)

        assert list(cache.keys()) == list(parents)
        assert set(cache.keys()) == set(parents)
        for stored, expected_parent in parents.items():
            assert cache.parent_of(stored) == expected_parent
        _assert_prefix_invariant(cache)
        if policy == "lfu":
            for stored, frequency in frequencies.items():
                assert cache._index.frequency_of(stored) == frequency


def test_lru_and_lfu_policy_classes_have_distinct_names() -> None:
    assert LRUEvictionPolicy().name == "lru"
    assert LFUEvictionPolicy().name == "lfu"


@pytest.mark.parametrize("policy", ["lru", "lfu"])
def test_overwrite_refreshes_recency_without_counting_a_hit(policy: str) -> None:
    cache = MemoryKVCache(max_entries=2, eviction_policy=policy)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 3)
    assert cache.frequency_of("a") == 1
    cache.put("c", 4)
    assert set(cache.keys()) == {"a", "c"}


def test_custom_scorer_only_sees_unprotected_leaves() -> None:
    class Scorer:
        name = "checked-lru"

        def __call__(self, key, frequency, last_access):
            assert key == "other"  # root is internal, child is protected.
            assert frequency == 1
            return (last_access,)

    index = PrefixLRUIndex(max_entries=3, eviction_policy=Scorer())
    index.admit("root", None)
    index.admit("child", "root")
    index.admit("other", None)
    assert index.admit("new", "child").evicted == ("other",)


def test_invalidation_removes_access_state_before_reinsertion() -> None:
    index = PrefixLRUIndex(eviction_policy="lfu")
    index.admit("a", None, size_bytes=2)
    index.admit("b", "a", size_bytes=3)
    index.touch("b")
    assert index.invalidate("a") == ("b", "a")
    assert index.nbytes == len(index) == 0
    with pytest.raises(KeyError):
        index.frequency_of("b")
    index.admit("b", None, size_bytes=1)
    assert index.frequency_of("b") == 1
    assert index.nbytes == 1


@pytest.mark.parametrize("bad", ["unknown", "", "fifo"])
def test_unknown_policy_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown eviction_policy"):
        MemoryKVCache(eviction_policy=bad)
