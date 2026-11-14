import threading

import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.storage.memory import MemoryKVCache  # noqa: E402
from nano_lmcache.tensor.async_store import AsyncTensorWriteStorage  # noqa: E402

pytestmark = pytest.mark.torch


def test_host_snapshot_is_owned_and_close_rejects_writes():
    backend = MemoryKVCache()
    storage = AsyncTensorWriteStorage(backend)
    source = torch.arange(16.0)
    storage.put("root", source)
    source.zero_()
    storage.close()
    assert torch.equal(backend.get("root"), torch.arange(16.0))
    with pytest.raises(RuntimeError, match="closed"):
        storage.put("late", source)


@pytest.mark.gpu
def test_cuda_snapshot_survives_source_reuse_and_pool_reuse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("device-wide synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    backend = MemoryKVCache()
    storage = AsyncTensorWriteStorage(backend, max_pending=2, max_pinned_bytes=4096)
    try:
        source = torch.arange(256.0, device="cuda")
        for index in range(10):
            source.fill_(index)
            storage.put(str(index), source)
            source.fill_(-1)
        storage.flush()
        for index in range(10):
            assert torch.equal(
                backend.get(str(index)), torch.full((256,), float(index))
            )
        assert storage.pool.allocated_bytes <= 4096
        assert storage.pool.free_bytes == storage.pool.allocated_bytes
    finally:
        storage.close()


@pytest.mark.gpu
def test_failed_cuda_backend_returns_pinned_buffers():
    class Failed(MemoryKVCache):
        def put(self, key, value, *, parent_key=None):
            raise OSError("injected failure")

    storage = AsyncTensorWriteStorage(Failed())
    storage.put("root", torch.ones(16, device="cuda"))
    with pytest.raises(RuntimeError, match="worker failed"):
        storage.close()
    assert storage.pool.free_bytes == storage.pool.allocated_bytes


def test_bounded_queue_stalls_instead_of_losing_parent_chain():
    gate = threading.Event()
    started = threading.Event()

    class Slow(MemoryKVCache):
        def put(self, key, value, *, parent_key=None):
            started.set()
            assert gate.wait(timeout=5)
            super().put(key, value, parent_key=parent_key)

    storage = AsyncTensorWriteStorage(Slow(), max_pending=1)
    try:
        storage.put("a", torch.ones(1))
        assert started.wait(timeout=5)
        storage.put("b", torch.ones(1), parent_key="a")
        stalled = threading.Event()
        original_flush = storage.flush

        def flush():
            stalled.set()
            original_flush()

        storage.flush = flush
        producer = threading.Thread(
            target=lambda: storage.put("c", torch.ones(1), parent_key="b")
        )
        producer.start()
        assert stalled.wait(timeout=5)
        gate.set()
        producer.join(timeout=5)
        assert not producer.is_alive()
        storage.flush()
        assert storage.stalls == 1
        assert storage.get("c") is not None
    finally:
        gate.set()
        storage.close()
