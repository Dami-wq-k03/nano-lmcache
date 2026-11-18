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


class RecordingBatchMemory(MemoryKVCache):
    def __init__(self):
        super().__init__()
        self.get_many_calls = []

    def get_many(self, keys):
        snapshot = tuple(keys)
        self.get_many_calls.append(snapshot)
        return [MemoryKVCache.get(self, key) for key in snapshot]


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


def test_server_delegates_batch_reads_to_a_capable_backend():
    backend = RecordingBatchMemory()
    instance = RemoteCacheServer(storage=backend)
    instance.start()
    try:
        cache = client(instance)
        cache.put("root", {"value": 1})
        cache.put("child", {"value": 2}, parent_key="root")

        assert cache.get_many(["root", "child", "missing"]) == [
            {"value": 1},
            {"value": 2},
            None,
        ]
        assert backend.get_many_calls == [("root", "child", "missing")]
    finally:
        instance.close()


def test_batch_response_streams_when_combined_payloads_exceed_one_frame():
    serializer = JsonSerializer()
    backend = RecordingBatchMemory()
    assert backend.try_put("root", serializer.dumps("r" * 160))
    assert backend.try_put(
        "child", serializer.dumps("c" * 160), parent_key="root"
    )
    instance = RemoteCacheServer(storage=backend, max_frame_size=350)
    instance.start()
    try:
        host, port = instance.address
        cache = RemoteCacheClient(
            host, port, serializer=serializer, max_frame_size=350
        )

        assert cache.get("root") == "r" * 160
        assert cache.get("child") == "c" * 160
        assert cache.get_many(["root", "child"]) == ["r" * 160, "c" * 160]
        assert backend.get_many_calls == [("root", "child")]
    finally:
        instance.close()


def test_server_read_errors_are_explicit_for_scalar_and_batch_requests():
    class FailingStorage(RecordingBatchMemory):
        def get(self, key):
            raise ValueError("backend read failed")

        def get_many(self, keys):
            raise ValueError("backend read failed")

    instance = RemoteCacheServer(storage=FailingStorage())
    instance.start()
    try:
        cache = client(instance)
        with pytest.raises(RemoteCacheError, match="backend read failed"):
            cache.get("key")
        with pytest.raises(RemoteCacheError, match="backend read failed"):
            cache.get_many(["key"])
    finally:
        instance.close()


def test_missing_parent_is_rejected_without_stranding_child(server):
    cache = client(server)
    with pytest.raises(RemoteCacheError, match="rejected"):
        cache.put("child", "value", parent_key="missing")
    assert cache.get("child") is None


def test_get_failure_is_a_miss():
    cache = RemoteCacheClient("127.0.0.1", 1, timeout=0.01)
    assert cache.get("missing") is None
    assert cache.get_many(["first", "second"]) == [None, None]
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


def test_server_can_use_persistent_packed_byte_storage(tmp_path):
    from nano_lmcache.storage.coalesced import CoalescedDiskKVCache

    with CoalescedDiskKVCache(tmp_path) as backend:
        instance = RemoteCacheServer(storage=backend)
        instance.start()
        try:
            cache = client(instance)
            cache.put("root", {"value": 42})
            cache.put("child", {"value": 43}, parent_key="root")
        finally:
            instance.close()
    with CoalescedDiskKVCache(tmp_path) as backend:
        instance = RemoteCacheServer(storage=backend)
        instance.start()
        try:
            assert client(instance).get_many(["root", "child"]) == [
                {"value": 42},
                {"value": 43},
            ]
            assert backend.read_ops == 1
        finally:
            instance.close()
