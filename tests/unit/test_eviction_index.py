import pytest

from nano_lmcache.storage.eviction import Admission, PrefixLRUIndex


def test_unbounded_index_admits_everything():
    index = PrefixLRUIndex()

    for i in range(20):
        assert index.admit(f"k{i}", None) == Admission(admitted=True, evicted=())

    assert len(index) == 20
    assert index.max_entries is None


@pytest.mark.parametrize("bad", [0, -3])
def test_rejects_non_positive_capacity(bad):
    with pytest.raises(ValueError):
        PrefixLRUIndex(max_entries=bad)


def test_contains_and_parent_of():
    index = PrefixLRUIndex()
    index.admit("a", None)
    index.admit("b", "a")

    assert "a" in index
    assert "missing" not in index
    assert index.parent_of("a") is None
    assert index.parent_of("b") == "a"


def test_full_index_names_the_lru_leaf_as_evicted():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", None)

    decision = index.admit("c", None)

    assert decision == Admission(admitted=True, evicted=("a",))
    assert "a" not in index
    assert list(index.keys_by_recency()) == ["b", "c"]


def test_touch_makes_an_entry_most_recently_used():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", None)
    index.touch("a")

    assert index.admit("c", None).evicted == ("b",)


def test_readmitting_a_key_touches_it_without_evicting():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", None)

    decision = index.admit("a", None)

    assert decision == Admission(admitted=True, evicted=())
    assert list(index.keys_by_recency()) == ["b", "a"]


def test_entries_with_children_are_never_evicted():
    index = PrefixLRUIndex(max_entries=3)
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("c", "b")

    assert index.admit("x", None).evicted == ("c",)


def test_parent_of_new_entry_is_protected():
    index = PrefixLRUIndex(max_entries=2)
    index.admit("a", None)
    index.admit("b", "a")

    decision = index.admit("c", "b")

    assert decision == Admission(admitted=False, evicted=())
    assert index.stats.rejected_no_room == 1


def test_missing_parent_is_rejected():
    index = PrefixLRUIndex()

    decision = index.admit("orphan", "never-admitted")

    assert decision == Admission(admitted=False, evicted=())
    assert "orphan" not in index
    assert index.stats.rejected_missing_parent == 1


def test_evictions_are_counted():
    index = PrefixLRUIndex(max_entries=1)
    index.admit("a", None)
    index.admit("b", None)
    index.admit("c", None)

    assert index.stats.evictions == 2


def test_restore_rebuilds_without_eviction_or_stats():
    index = PrefixLRUIndex(max_entries=2)

    index.restore("a", None)
    index.restore("b", "a")

    assert list(index.keys_by_recency()) == ["a", "b"]
    assert index.parent_of("b") == "a"
    assert index.stats.evictions == 0
    # b is a's child, so a is protected exactly as if admitted normally.
    assert index.admit("c", None).evicted == ("b",)


def test_restore_requires_the_parent_first():
    index = PrefixLRUIndex()

    with pytest.raises(ValueError):
        index.restore("b", "a")


def test_restore_beyond_capacity_is_an_error():
    index = PrefixLRUIndex(max_entries=1)
    index.restore("a", None)

    with pytest.raises(ValueError):
        index.restore("b", None)
