"""Preserve exact LRU semantics while avoiding a scan of internal nodes."""

import random
from collections import OrderedDict

import pytest

from nano_lmcache.storage.eviction import Admission, PrefixLRUIndex


def test_newly_exposed_leaf_uses_its_original_access_time():
    index = PrefixLRUIndex(max_entries=4)
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("x", None)
    index.admit("y", None)

    assert index.admit("z", None).evicted == ("b",)
    # a became a leaf after x and y, but was last used before both.
    assert index.admit("w", None).evicted == ("a",)


def test_internal_node_hit_controls_order_when_it_becomes_a_leaf():
    index = PrefixLRUIndex(max_entries=4)
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("x", None)
    index.touch("a")
    index.admit("y", None)

    assert index.admit("z", None).evicted == ("b",)
    assert index.admit("w", None).evicted == ("x",)
    assert index.admit("v", None).evicted == ("a",)


def test_protected_oldest_leaf_is_available_for_the_next_write():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", None)

    assert index.admit("c", "a").evicted == ("b",)
    assert index.admit("d", None).evicted == ("c",)
    assert index.admit("e", None).evicted == ("a",)


def test_rejected_writes_do_not_lose_the_only_leaf_candidate():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", "a")

    for _ in range(5):
        assert index.admit("c", "b") == Admission(admitted=False, evicted=())
    assert index.admit("x", None).evicted == ("b",)


def test_stale_leaf_hits_and_reinserted_keys_do_not_evict_recent_values():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", None)
    for _ in range(100):
        index.touch("a")
    assert index.admit("c", None).evicted == ("b",)
    assert index.admit("b", None).evicted == ("a",)
    assert index.admit("a", None).evicted == ("c",)


def test_restore_and_replayed_recency_preserve_leaf_order():
    index = PrefixLRUIndex(max_entries=4)
    index.restore("a", None)
    index.restore("b", "a")
    index.restore("x", None)
    index.restore("y", None)
    for key in ["b", "x", "a", "y"]:
        index.touch(key)

    assert index.admit("z", None).evicted == ("b",)
    assert index.admit("w", None).evicted == ("x",)
    assert index.admit("v", None).evicted == ("a",)


def test_deep_chain_victim_selection_does_not_scan_all_entries():
    class CountedOrder(OrderedDict):
        inspected = 0

        def __iter__(self):
            for key in super().__iter__():
                self.inspected += 1
                yield key

    index = PrefixLRUIndex(max_entries=2048)
    parent = None
    for depth in range(2048):
        key = str(depth)
        index.admit(key, parent)
        parent = key
    order = CountedOrder(index._parents)
    index._parents = order

    assert index.admit("new", None).evicted == ("2047",)
    # Count inspected entries instead of asserting a noisy wall-clock limit.
    assert order.inspected < 32


def test_repeated_leaf_hits_keep_candidate_memory_bounded():
    index = PrefixLRUIndex(max_entries=4)
    for key in "abcd":
        index.admit(key, None)
    for _ in range(10000):
        index.touch("a")

    assert len(index._leaf_heap) <= 64
    assert index.admit("e", None).evicted == ("b",)


def test_heap_compaction_keeps_newly_exposed_parent_in_original_order():
    index = PrefixLRUIndex(max_entries=4)
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("x", None)
    index.admit("y", None)
    for _ in range(1000):
        index.touch("y")

    assert index.admit("z", None).evicted == ("b",)
    assert index.admit("w", None).evicted == ("a",)


@pytest.mark.parametrize("capacity", [1, 2, 8, None])
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_operations_match_the_original_linear_scan(capacity, seed):
    rng = random.Random(seed)
    index = PrefixLRUIndex(max_entries=capacity)
    # Independent reference: insertion-ordered dictionaries model recency.
    parents = {}
    evictions = missing = no_room = 0
    keys = [str(key) for key in range(16)]

    for _ in range(600):
        key = rng.choice(keys)
        if key in parents and rng.randrange(3) == 0:
            parents[key] = parents.pop(key)
            index.touch(key)
        else:
            parent = rng.choice([None, "missing", *keys])
            victims = ()
            admitted = True
            if key in parents:
                parents[key] = parents.pop(key)
            elif parent is not None and parent not in parents:
                admitted = False
                missing += 1
            else:
                if capacity is not None and len(parents) >= capacity:
                    victim = next(
                        (
                            candidate
                            for candidate in parents
                            if candidate != parent and candidate not in parents.values()
                        ),
                        None,
                    )
                    if victim is None:
                        admitted = False
                        no_room += 1
                    else:
                        del parents[victim]
                        evictions += 1
                        victims = (victim,)
                if admitted:
                    parents[key] = parent

            assert index.admit(key, parent) == Admission(admitted, victims)

        assert list(index.keys_by_recency()) == list(parents)
        assert index.stats.evictions == evictions
        assert index.stats.rejected_missing_parent == missing
        assert index.stats.rejected_no_room == no_room
        for stored, parent in parents.items():
            assert index.parent_of(stored) == parent
