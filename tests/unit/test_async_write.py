import threading

import pytest

from nano_lmcache.storage.async_write import (
    AsyncWriteBehindStorage,
    WriteBehindBackpressure,
)
from nano_lmcache.storage.memory import MemoryKVCache


def test_reads_wait_for_accepted_writes():
    backend = MemoryKVCache()
    storage = AsyncWriteBehindStorage(backend, max_pending=4)
    storage.put("root", "value")
    # The worker may already have committed; get must see the accepted write.
    assert storage.get("root") == "value"
    storage.flush()
    assert backend.get("root") == "value"
    storage.close()


@pytest.mark.parametrize("flush", [True, False])
def test_close_stops_worker_after_backend_failure(flush):
    class FailedStorage(MemoryKVCache):
        def put(self, key, value, *, parent_key=None):
            raise OSError("injected disk failure")

    storage = AsyncWriteBehindStorage(FailedStorage())
    storage.put("root", "value")
    with pytest.raises(RuntimeError, match="worker failed"):
        storage.close(flush=flush)
    assert not storage._worker.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        storage.put("late", "value")
    storage.close()


def test_close_rejects_writes_while_draining():
    started = threading.Event()
    release = threading.Event()

    class SlowStorage(MemoryKVCache):
        def put(self, key, value, *, parent_key=None):
            started.set()
            assert release.wait(timeout=5)
            super().put(key, value, parent_key=parent_key)

    storage = AsyncWriteBehindStorage(SlowStorage())
    storage.put("root", "value")
    assert started.wait(timeout=5)
    closing = threading.Event()
    original_flush = storage.flush

    def flush():
        closing.set()
        original_flush()

    storage.flush = flush
    closer = threading.Thread(target=storage.close)
    closer.start()
    try:
        assert closing.wait(timeout=5)
        with pytest.raises(RuntimeError, match="closed"):
            storage.put("late", "value")
    finally:
        release.set()
        closer.join(timeout=5)
    assert not closer.is_alive()
    assert storage.backend.get("root") == "value"


def test_ordered_parent_writes_and_owned_prepare():
    seen: list[object] = []
    storage = AsyncWriteBehindStorage(
        MemoryKVCache(),
        prepare=lambda value: (seen.append(value), value)[1],
    )
    storage.put("root", "root")
    storage.put("child", "child", parent_key="root")
    storage.flush()
    assert seen == ["root", "child"]
    assert storage.backend.get("child") == "child"
    storage.close()


def test_backpressure_is_bounded():
    class SlowStorage(MemoryKVCache):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def put(self, key, value, *, parent_key=None):
            self.started.set()
            self.release.wait(timeout=2)
            super().put(key, value, parent_key=parent_key)

    backend = SlowStorage()
    storage = AsyncWriteBehindStorage(backend, max_pending=1)
    try:
        storage.put("a", "a")
        assert backend.started.wait(timeout=2)
        storage.put("b", "b")
        with pytest.raises(WriteBehindBackpressure):
            storage.put("c", "c")
    finally:
        backend.release.set()
        storage.close()
