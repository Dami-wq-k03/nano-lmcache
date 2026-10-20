import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage


class DictStorage:
    """A second, independent backend that never inherits from MemoryKVCache."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        self.data[key] = value

    def get(self, key: str) -> object | None:
        return self.data.get(key)


def test_memory_cache_satisfies_storage_protocol():
    assert isinstance(MemoryKVCache(), CacheStorage)


def test_custom_backend_satisfies_storage_protocol():
    assert isinstance(DictStorage(), CacheStorage)


def test_prefix_cache_works_with_custom_backend():
    storage = DictStorage()
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=storage,
    )
    tokens = [101, 205, 307, 409]
    cache.store(tokens, ["kv-0", "kv-1"])

    result = cache.lookup(tokens)

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4
    assert result.values == ("kv-0", "kv-1")
    # 数据确实写进了注入的后端，而不是默认的 MemoryKVCache。
    assert set(storage.data) == set(result.keys)


def test_custom_backend_keeps_prefix_matching_behavior():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=DictStorage(),
    )
    cache.store(
        [101, 205, 307, 409, 511, 613],
        ["kv-0", "kv-1", "kv-2"],
    )

    result = cache.lookup([101, 205, 307, 409, 777, 888])

    assert result.matched_chunks == 2
    assert result.values == ("kv-0", "kv-1")

    miss = cache.lookup([999, 205, 307, 409])

    assert miss.matched_chunks == 0
    assert miss.values == ()


def test_custom_backend_keeps_metrics_behavior():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=DictStorage(),
    )
    cache.store(
        [101, 205, 307, 409, 511, 613],
        ["kv-0", "kv-1", "kv-2"],
    )

    cache.lookup([101, 205, 307, 409, 777, 888])

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 1
    assert cache.metrics.hit_tokens == 4


def test_two_caches_can_share_one_backend():
    storage = DictStorage()
    writer = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=storage
    )
    reader = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=storage
    )
    tokens = [101, 205]
    writer.store(tokens, ["kv-0"])

    assert reader.lookup(tokens).matched_chunks == 1


def test_store_rejects_none_values():
    # None 在存储约定里专门表示 miss，不能同时当作合法缓存值。
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=DictStorage(),
    )

    with pytest.raises(ValueError, match="None"):
        cache.store([101, 205, 307, 409], ["kv-0", None])


def test_rejected_store_writes_nothing():
    storage = DictStorage()
    cache = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=storage
    )

    with pytest.raises(ValueError):
        cache.store([101, 205, 307, 409], ["kv-0", None])

    # 校验必须发生在写入之前，否则会留下残缺的前缀。
    assert storage.data == {}
