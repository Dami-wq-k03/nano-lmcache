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
