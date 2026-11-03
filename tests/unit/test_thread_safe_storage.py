"""Concurrent storage access keeps each backend's state consistent."""

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.tiered import TieredStorage

MAX_ENTRIES = 8
WORKERS = 8
OPERATIONS_PER_WORKER = 60


def _make_storage(kind: str, tmp_path):
    if kind == "memory":
        return MemoryKVCache(
            max_entries=MAX_ENTRIES,
            max_bytes=48,
            size_fn=lambda value: len(str(value)),
        )
    if kind == "disk":
        return DiskKVCache(tmp_path / "disk", max_entries=MAX_ENTRIES)
    if kind == "tiered":
        return TieredStorage(
            fast=MemoryKVCache(max_entries=MAX_ENTRIES),
            slow=DiskKVCache(tmp_path / "tiered", max_entries=MAX_ENTRIES),
        )
    raise AssertionError(f"unknown backend {kind}")


def _tokens(rng: random.Random) -> list[int]:
    """Return three chunks with a small shared-prefix working set."""
    root = rng.randrange(3)
    branch = rng.randrange(6)
    tail = rng.randrange(20)
    return [root, 100 + root, 200 + branch, 300 + branch, 400 + tail, 500 + tail]


def _storage_layers(storage: object) -> tuple[MemoryKVCache | DiskKVCache, ...]:
    if isinstance(storage, TieredStorage):
        assert isinstance(storage.fast, (MemoryKVCache, DiskKVCache))
        assert isinstance(storage.slow, (MemoryKVCache, DiskKVCache))
        return storage.fast, storage.slow
    assert isinstance(storage, (MemoryKVCache, DiskKVCache))
    return (storage,)


def _assert_layer_invariants(layer: MemoryKVCache | DiskKVCache) -> None:
    assert len(layer) <= MAX_ENTRIES
    if isinstance(layer, MemoryKVCache) and layer.max_bytes is not None:
        assert layer.nbytes <= layer.max_bytes

    # Workers have joined before this inspection, so taking a stable view of
    # the private index is appropriate for checking the backend invariant.
    keys = tuple(layer._index.keys_by_recency())
    assert len(keys) == len(layer)
    for key in keys:
        parent = layer.parent_of(key)
        assert parent is None or parent in layer

    stats = layer.stats
    assert stats.rejected == (
        stats.rejected_no_room + stats.rejected_missing_parent
    )

    if isinstance(layer, DiskKVCache):
        assert len(list(layer.directory.glob("*.entry"))) == len(layer)
        assert not list(layer.directory.glob("*.tmp"))


@pytest.mark.parametrize("kind", ["memory", "disk", "tiered"])
def test_concurrent_store_and_lookup_keep_storage_consistent(
    kind, tmp_path, monkeypatch
):
    """Concurrent prefix-cache calls preserve limits, chains, and counters."""
    # The test stresses cache bookkeeping rather than durability and would
    # otherwise spend most of its time flushing hundreds of tiny files.
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    storage = _make_storage(kind, tmp_path)
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="thread-test",
        storage=storage,
    )
    start = threading.Barrier(WORKERS)

    def worker(worker_id: int) -> tuple[int, int]:
        rng = random.Random(worker_id)
        lookup_chunks = 0
        tier_gets = 0
        start.wait()

        for step in range(OPERATIONS_PER_WORKER):
            tokens = _tokens(rng)
            if rng.random() < 0.55:
                cache.store(
                    tokens,
                    [f"{worker_id}-{step}-{chunk}" for chunk in range(3)],
                )
                continue

            result = cache.lookup(tokens)
            lookup_chunks += len(tokens) // cache.chunk_size
            # A lookup probes every matched chunk and one additional chunk
            # unless all chunks matched.
            tier_gets += min(
                result.matched_chunks + 1,
                len(tokens) // cache.chunk_size,
            )

        return lookup_chunks, tier_gets

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = [executor.submit(worker, worker_id) for worker_id in range(WORKERS)]
        totals = [result.result() for result in results]

    expected_lookup_chunks = sum(chunks for chunks, _ in totals)
    assert cache.metrics.total_chunks == expected_lookup_chunks
    assert cache.metrics.hit_chunks + cache.metrics.miss_chunks == (
        expected_lookup_chunks
    )

    for layer in _storage_layers(storage):
        _assert_layer_invariants(layer)

    if isinstance(storage, TieredStorage):
        expected_tier_gets = sum(gets for _, gets in totals)
        assert storage.tier_stats.lookups == expected_tier_gets


def test_memory_get_waits_for_put_to_publish_next_state(monkeypatch):
    """A get waits for an in-progress put to publish its next cache state."""
    storage = MemoryKVCache(max_entries=1)
    storage.put("a", "payload-a")
    original_admit = storage._index.admit
    admission_started = threading.Event()
    allow_put_to_finish = threading.Event()
    reader_started = threading.Event()
    reader_checked_values = threading.Event()

    class SignalingValues(dict[str, object]):
        def __contains__(self, key: object) -> bool:
            reader_checked_values.set()
            return super().__contains__(key)

    storage._values = SignalingValues(storage._values)

    def pause_before_payload_publish(key, parent_key, *, size_bytes=0):
        decision = original_admit(key, parent_key, size_bytes=size_bytes)
        admission_started.set()
        assert allow_put_to_finish.wait(timeout=2)
        return decision

    def read_evicted_entry() -> object | None:
        reader_started.set()
        return storage.get("a")

    monkeypatch.setattr(storage._index, "admit", pause_before_payload_publish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer_future = executor.submit(storage.put, "b", "payload-b")
        assert admission_started.wait(timeout=2)

        reader_future = executor.submit(read_evicted_entry)
        assert reader_started.wait(timeout=2)
        # The writer still owns MemoryKVCache._lock, so the reader cannot
        # inspect _values before the write has published its final state.
        assert not reader_checked_values.wait(timeout=0.1)

        allow_put_to_finish.set()
        writer_future.result(timeout=2)
        assert reader_future.result(timeout=2) is None

    assert storage.get("a") is None
    assert storage.get("b") == "payload-b"


def test_disk_get_keeps_a_file_stable_until_the_read_finishes(tmp_path, monkeypatch):
    """An eviction waits while another thread reads the same disk backend."""
    storage = DiskKVCache(tmp_path, max_entries=1)
    storage.put("a", "payload-a")
    original_read = storage._read
    read_started = threading.Event()
    allow_read = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def blocked_read(path):
        read_started.set()
        assert allow_read.wait(timeout=2)
        return original_read(path)

    def put_replacement() -> None:
        writer_started.set()
        storage.put("b", "payload-b")
        writer_finished.set()

    monkeypatch.setattr(storage, "_read", blocked_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        read_future = executor.submit(storage.get, "a")
        assert read_started.wait(timeout=2)
        write_future = executor.submit(put_replacement)
        assert writer_started.wait(timeout=2)
        assert not writer_finished.wait(timeout=0.1)

        allow_read.set()
        assert read_future.result(timeout=2) == "payload-a"
        write_future.result(timeout=2)

    assert storage.get("a") is None
    assert storage.get("b") == "payload-b"


def test_tiered_get_waits_for_a_slow_hit_promotion(monkeypatch, tmp_path):
    """A fast-tier hit waits for another get to finish its promotion."""
    fast = MemoryKVCache(max_entries=2)
    slow = DiskKVCache(tmp_path / "slow")
    storage = TieredStorage(fast=fast, slow=slow)
    storage.put("a", "payload-a")
    storage.put("b", "payload-b")
    fast.invalidate("a")

    original_fast_get = fast.get
    original_fast_put = fast.put
    promotion_started = threading.Event()
    allow_promotion = threading.Event()
    second_get_started = threading.Event()
    second_get_reached_fast = threading.Event()

    def signaling_fast_get(key: str) -> object | None:
        if key == "b":
            second_get_reached_fast.set()
        return original_fast_get(key)

    def blocked_fast_put(
        key: str, value: object, *, parent_key: str | None = None
    ) -> None:
        if key == "a":
            promotion_started.set()
            assert allow_promotion.wait(timeout=2)
        original_fast_put(key, value, parent_key=parent_key)

    def get_fast_entry() -> object | None:
        second_get_started.set()
        return storage.get("b")

    monkeypatch.setattr(fast, "get", signaling_fast_get)
    monkeypatch.setattr(fast, "put", blocked_fast_put)
    with ThreadPoolExecutor(max_workers=2) as executor:
        promotion_future = executor.submit(storage.get, "a")
        assert promotion_started.wait(timeout=2)

        fast_hit_future = executor.submit(get_fast_entry)
        assert second_get_started.wait(timeout=2)
        # The first get owns TieredStorage._lock while it promotes a.  A
        # second get cannot even reach the fast tier before promotion ends.
        assert not second_get_reached_fast.wait(timeout=0.1)

        allow_promotion.set()
        assert promotion_future.result(timeout=2) == "payload-a"
        assert fast_hit_future.result(timeout=2) == "payload-b"

    assert fast.get("a") == "payload-a"
    assert storage.tier_stats.fast_hits == 1
    assert storage.tier_stats.slow_hits == 1
    assert storage.tier_stats.promotions == 1
