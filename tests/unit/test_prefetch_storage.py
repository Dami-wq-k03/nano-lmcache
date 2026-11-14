import threading
from concurrent.futures import CancelledError

import pytest

from nano_lmcache.storage.coalesced import CoalescedDiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.prefetch import AsyncPrefetchStorage
from nano_lmcache.storage.protocol import CacheStorage


class BlockingStorage(MemoryKVCache):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.read_keys = []

    def get(self, key):
        self.read_keys.append(key)
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().get(key)


def test_prefetch_uses_real_coalesced_batch_and_synchronous_protocol(tmp_path):
    with CoalescedDiskKVCache(tmp_path) as backend:
        with AsyncPrefetchStorage(backend) as storage:
            assert isinstance(storage, CacheStorage)
            storage.put("a", b"a")
            storage.put("b", b"b", parent_key="a")
            assert storage.prefetch(["b", "missing", "a"]).result(timeout=5) == [
                b"b",
                None,
                b"a",
            ]
            assert backend.read_ops == 1
            assert storage.get_many(["a", "b"]) == [b"a", b"b"]
            assert storage.get("missing") is None
        assert storage.pending == 0
        assert backend.get("a") == b"a"  # wrapper doesn't own backend lifecycle


def test_no_stale_prefetch_result_is_published_by_synchronous_get():
    backend = MemoryKVCache()
    backend.put("a", "old")
    with AsyncPrefetchStorage(backend) as storage:
        future = storage.prefetch(["a"])
        assert future.result(timeout=5) == ["old"]
        storage.put("a", "new")
        assert future.result() == ["old"]  # explicitly owned snapshot
        assert storage.get("a") == "new"


def test_running_and_queued_work_are_bounded_even_after_cancel():
    backend = BlockingStorage()
    backend.put("a", "a")
    storage = AsyncPrefetchStorage(backend, max_workers=1, max_pending=2)
    try:
        active = storage.prefetch(["a"])
        assert backend.started.wait(timeout=5)
        assert not active.cancel()  # running host I/O cannot be stopped
        queued = storage.prefetch(["never-read"])
        assert storage.pending == 2
        assert queued.cancel()
        with pytest.raises(CancelledError):
            queued.result()
        # Cancellation must not permit unlimited cancelled executor work items.
        for _ in range(10):
            with pytest.raises(RuntimeError, match="full"):
                storage.prefetch(["overflow"])
        assert storage.pending == 2
    finally:
        backend.release.set()
        storage.close()
    assert active.result() == ["a"]
    assert backend.read_keys == ["a"]
    assert storage.pending == 0


def test_close_cancels_queued_read_and_repeated_waiting_close_joins():
    backend = BlockingStorage()
    storage = AsyncPrefetchStorage(backend, max_workers=1, max_pending=2)
    try:
        active = storage.prefetch(["running"])
        assert backend.started.wait(timeout=5)
        queued = storage.prefetch(["queued"])
        storage.close(wait=False)
        assert queued.cancelled()
        assert not active.done()
        with pytest.raises(RuntimeError, match="closed"):
            storage.prefetch(["late"])
    finally:
        backend.release.set()
        storage.close(wait=True)
    assert active.result() == [None]
    assert storage.pending == 0
    assert backend.read_keys == ["running"]


def test_close_can_drain_instead_of_cancel():
    backend = BlockingStorage()
    storage = AsyncPrefetchStorage(backend, max_workers=1, max_pending=2)
    try:
        active = storage.prefetch(["a"])
        assert backend.started.wait(timeout=5)
        queued = storage.prefetch(["b"])
        storage.close(wait=False, cancel_pending=False)
    finally:
        backend.release.set()
        storage.close(cancel_pending=False)
    assert active.result() == [None]
    assert queued.result() == [None]
    assert backend.read_keys == ["a", "b"]


def test_worker_errors_propagate_and_shutdown_still_completes():
    class BrokenStorage(MemoryKVCache):
        def get(self, key):
            if key == "bad":
                raise OSError("read pipeline failure")
            return super().get(key)

    backend = BrokenStorage()
    backend.put("good", "value")
    storage = AsyncPrefetchStorage(backend)
    bad = storage.prefetch(["bad"])
    good = storage.prefetch(["good"])
    storage.close(cancel_pending=False)
    with pytest.raises(OSError, match="pipeline failure"):
        bad.result()
    assert good.result() == ["value"]
    assert storage.pending == 0
    storage.close()


def test_key_sequence_is_snapshotted_before_queued_read():
    backend = BlockingStorage()
    storage = AsyncPrefetchStorage(backend, max_workers=1, max_pending=2)
    try:
        storage.prefetch(["blocking"])
        assert backend.started.wait(timeout=5)
        keys = ["original"]
        future = storage.prefetch(keys)
        keys[0] = "mutated"
    finally:
        backend.release.set()
        storage.close(cancel_pending=False)
    assert future.result() == [None]
    assert backend.read_keys == ["blocking", "original"]


def test_batch_limit_and_submit_failure_release_slot(monkeypatch):
    storage = AsyncPrefetchStorage(MemoryKVCache(), max_batch_size=1)
    try:
        with pytest.raises(ValueError, match="max_batch_size"):
            storage.prefetch(["a", "b"])
        assert storage.pending == 0

        def fail(*args, **kwargs):
            raise RuntimeError("executor unavailable")

        monkeypatch.setattr(storage._executor, "submit", fail)
        with pytest.raises(RuntimeError, match="unavailable"):
            storage.prefetch(["a"])
        assert storage.pending == 0
    finally:
        storage.close()


def test_empty_batch_and_closed_synchronous_operations():
    storage = AsyncPrefetchStorage(MemoryKVCache())
    assert storage.prefetch([]).result(timeout=5) == []
    storage.close()
    for operation in (
        lambda: storage.get("a"),
        lambda: storage.get_many(["a"]),
        lambda: storage.put("a", "a"),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize(
    "limits", [{"max_workers": 0}, {"max_pending": 0}, {"max_batch_size": 0}]
)
def test_invalid_limits(limits):
    with pytest.raises(ValueError):
        AsyncPrefetchStorage(MemoryKVCache(), **limits)
