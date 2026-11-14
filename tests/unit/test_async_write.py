import threading

import pytest

from nano_lmcache.storage.async_write import (
    AsyncWriteBehindStorage,
    WriteBehindBackpressure,
)
from nano_lmcache.storage.memory import MemoryKVCache


def test_writes_are_not_visible_until_flush():
    backend = MemoryKVCache()
    storage = AsyncWriteBehindStorage(backend, max_pending=4)
    storage.put("root", "value")
    assert backend.get("root") is None
    assert storage.get("root") == "value"
    storage.flush()
    assert backend.get("root") == "value"
    storage.close()


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
