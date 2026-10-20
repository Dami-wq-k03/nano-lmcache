import pytest

from nano_lmcache.storage.memory import MemoryKVCache


def test_get_missing_key_returns_none():
    cache = MemoryKVCache()
    assert cache.get("missing") is None


def test_put_then_get_returns_stored_value():
    cache = MemoryKVCache()
    cache.put("key-a", {"kv": [1, 2, 3]})
    assert cache.get("key-a") == {"kv": [1, 2, 3]}


def test_put_overwrites_existing_value():
    cache = MemoryKVCache()
    cache.put("key-a", "first")
    cache.put("key-a", "second")
    assert cache.get("key-a") == "second"


def test_keys_are_independent():
    cache = MemoryKVCache()
    cache.put("key-a", "value-a")
    cache.put("key-b", "value-b")
    assert cache.get("key-a") == "value-a"
    assert cache.get("key-b") == "value-b"


def test_contains_and_parent_of():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2, parent_key="a")

    assert "a" in cache
    assert "missing" not in cache
    assert cache.parent_of("b") == "a"
    assert cache.parent_of("a") is None
    with pytest.raises(KeyError):
        cache.parent_of("missing")


def test_contains_does_not_change_recency():
    cache = MemoryKVCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)

    assert "a" in cache
    cache.put("c", 3)

    assert "a" not in cache
