import pytest

from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.remote import (
    RemoteCacheClient,
    RemoteCacheError,
    RemoteCacheServer,
)
from nano_lmcache.storage.serializers import JsonSerializer


@pytest.fixture
def server():
    instance = RemoteCacheServer()
    instance.start()
    yield instance
    instance.close()


def client(server):
    host, port = server.address
    return RemoteCacheClient(host, port, serializer=JsonSerializer())


def test_two_clients_share_prefix_safe_storage(server):
    writer = client(server)
    reader = client(server)
    writer.put("root", {"value": 1})
    writer.put("child", {"value": 2}, parent_key="root")

    assert "root" in reader
    assert reader.get("root") == {"value": 1}
    assert reader.get_many(["root", "child", "missing"]) == [
        {"value": 1},
        {"value": 2},
        None,
    ]


def test_missing_parent_is_rejected_without_stranding_child(server):
    cache = client(server)
    with pytest.raises(RemoteCacheError, match="rejected"):
        cache.put("child", "value", parent_key="missing")
    assert cache.get("child") is None


def test_get_failure_is_a_miss():
    cache = RemoteCacheClient("127.0.0.1", 1, timeout=0.01)
    assert cache.get("missing") is None
    assert "missing" not in cache


def test_none_is_reserved_for_misses(server):
    with pytest.raises(ValueError, match="None"):
        client(server).put("key", None)


def test_server_preserves_empty_caller_storage_and_its_capacity():
    backend = MemoryKVCache(max_entries=1)
    instance = RemoteCacheServer(storage=backend)
    instance.start()
    try:
        assert instance.storage is backend
        cache = client(instance)
        cache.put("first", "one")
        cache.put("second", "two")
        assert cache.get("first") is None
        assert cache.get("second") == "two"
        assert len(backend) == 1
    finally:
        instance.close()
