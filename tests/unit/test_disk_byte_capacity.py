"""Disk byte limits count complete entry files and preserve prefix chains."""

import os
import random

import pytest

from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.serializers import JsonSerializer
from nano_lmcache.storage.stats import StorageStats


class Utf8Serializer:
    def dumps(self, value):
        return value.encode("utf-8")

    def loads(self, data):
        return data.decode("utf-8")


def disk_bytes(directory):
    return sum(path.stat().st_size for path in directory.glob("*.entry"))


def snapshot(directory):
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.iterdir()
    }


@pytest.fixture
def measure_size(tmp_path_factory):
    """Measure real files without duplicating the header encoding rules."""

    def measure(key, value, *, serializer=None):
        cache = DiskKVCache(tmp_path_factory.mktemp("measure"), serializer=serializer)
        cache.put(key, value)
        return cache._path(key).stat().st_size

    return measure


@pytest.fixture
def chain(tmp_path):
    cache = DiskKVCache(tmp_path)
    for key, parent in [("a", None), ("b", "a"), ("c", "b")]:
        cache.put(key, "xx", parent_key=parent)
    return DiskKVCache(tmp_path, max_bytes=cache.nbytes)


@pytest.mark.parametrize("serializer", [JsonSerializer(), Utf8Serializer()])
@pytest.mark.parametrize("margin", [None, 0, -1])
def test_complete_file_size_controls_admission(
    tmp_path, measure_size, serializer, margin
):
    key, value = "根\n😀", "雪\n🚀"
    size = measure_size(key, value, serializer=serializer)
    budget = None if margin is None else size + margin
    cache = DiskKVCache(tmp_path, max_bytes=budget, serializer=serializer)

    cache.put(key, value)

    admitted = margin != -1
    assert cache.max_bytes == budget
    assert cache.get(key) == (value if admitted else None)
    assert cache.nbytes == disk_bytes(tmp_path) == (size if admitted else 0)
    assert cache.stats.rejected_no_room == int(not admitted)
    assert size > len(serializer.dumps(value))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_invalid_byte_capacity(tmp_path, bad):
    with pytest.raises(ValueError, match="max_bytes"):
        DiskKVCache(tmp_path, max_bytes=bad)


def test_restart_preserves_bytes_and_lru_for_multiple_evictions(tmp_path, measure_size):
    size = measure_size("a", "xx")
    cache = DiskKVCache(tmp_path, max_bytes=3 * size)
    for key in "abc":
        cache.put(key, "xx")
    cache.get("a")
    before = snapshot(tmp_path)

    reopened = DiskKVCache(tmp_path, max_bytes=cache.max_bytes)

    assert reopened.nbytes == cache.nbytes == 3 * size
    assert snapshot(tmp_path) == before
    reopened.put("d", "d" * (2 + size))
    assert {key for key in "abcd" if key in reopened} == {"a", "d"}
    assert reopened.nbytes == disk_bytes(tmp_path) == reopened.max_bytes
    assert reopened.stats == StorageStats(evictions=2)


@pytest.mark.parametrize("key", ["a", "new"])
def test_rejected_write_preserves_files_and_recency(tmp_path, measure_size, key):
    cache = DiskKVCache(tmp_path, max_bytes=2 * measure_size("a", "xx"))
    for name in "ab":
        cache.put(name, "xx")
    before = snapshot(tmp_path)

    cache.put(key, "x" * cache.max_bytes)

    assert snapshot(tmp_path) == before
    assert cache.nbytes == disk_bytes(tmp_path) == cache.max_bytes
    assert cache.stats == StorageStats(rejected_no_room=1)
    cache.put("c", "xx")
    assert "a" not in cache
    assert cache.get("b") == "xx"


def test_overwrite_uses_original_parent_and_releases_shrunk_bytes(chain):
    cache = chain
    leaf_size = cache._path("c").stat().st_size
    cache.put("b", "b" * (2 + leaf_size), parent_key="missing-" * 100)

    assert cache.get("b") == "b" * (2 + leaf_size)
    assert cache.parent_of("b") == "a"
    assert "c" not in cache
    assert cache.nbytes == disk_bytes(cache.directory) == cache.max_bytes

    cache.put("b", "")
    cache.put("c", "xx", parent_key="b")
    assert cache.get("a") == cache.get("c") == "xx"
    assert cache.get("b") == ""
    assert cache.nbytes == disk_bytes(cache.directory) < cache.max_bytes
    assert cache.stats == StorageStats(evictions=1)


def test_byte_pressure_keeps_a_complete_prefix(chain, measure_size):
    cache = chain
    root_size = cache._path("a").stat().st_size
    overhead = measure_size("x", "x") - 1
    value = "x" * (cache.max_bytes - root_size - overhead)

    cache.put("x", value)

    assert {key for key in "abcx" if key in cache} == {"a", "x"}
    assert cache.get("a") == "xx"
    assert cache.get("x") == value
    assert cache.nbytes == disk_bytes(cache.directory) == cache.max_bytes
    assert cache.stats.evictions == 2


def test_failed_write_propagates_and_reopen_restores_bytes(
    tmp_path, measure_size, monkeypatch
):
    budget = measure_size("a", "old")
    cache = DiskKVCache(tmp_path, max_bytes=budget)
    cache.put("a", "old")

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="injected write failure"):
            cache.put("a", "")

    reopened = DiskKVCache(tmp_path, max_bytes=budget)
    assert reopened.get("a") == "old"
    assert reopened.nbytes == disk_bytes(tmp_path) == budget


@pytest.mark.parametrize("max_entries,budget_units", [(2, 4), (4, 2), (2, 2)])
def test_entry_and_byte_limits_both_apply(
    tmp_path, measure_size, max_entries, budget_units
):
    budget = budget_units * measure_size("a", "xx")
    cache = DiskKVCache(tmp_path, max_entries=max_entries, max_bytes=budget)
    for key in "abc":
        cache.put(key, "xx")
    assert {key for key in "abc" if key in cache} == {"b", "c"}
    assert len(cache) <= max_entries
    assert cache.nbytes == disk_bytes(tmp_path) <= budget


def test_reopening_over_budget_preserves_files(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put("a", "parent")
    cache.put("b", "child", parent_key="a")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="max_bytes"):
        DiskKVCache(tmp_path, max_bytes=cache.nbytes - 1)
    assert snapshot(tmp_path) == before
    assert DiskKVCache(tmp_path, max_bytes=cache.nbytes).nbytes == disk_bytes(tmp_path)


def test_recovery_counts_only_surviving_entry_files(chain):
    cache = DiskKVCache(chain.directory)
    cache.put("keep", "valid")
    cache._path("a").unlink()
    (cache.directory / "interrupted.tmp").write_bytes(b"x" * 1000)
    (cache.directory / ("0" * 64 + ".entry")).write_bytes(b"invalid")
    (cache.directory / "notes.txt").write_bytes(b"not an entry" * 100)
    budget = cache._path("keep").stat().st_size

    reopened = DiskKVCache(cache.directory, max_bytes=budget)

    assert len(reopened) == 1
    assert reopened.get("keep") == "valid"
    assert reopened.nbytes == disk_bytes(cache.directory) == budget


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_writes_preserve_prefixes_and_bytes_across_restarts(tmp_path, seed):
    rng = random.Random(seed)
    cache = DiskKVCache(tmp_path, max_entries=8, max_bytes=480)
    parents = {
        f"{root}-{depth}": None if depth == 0 else f"{root}-{depth - 1}"
        for root in range(4)
        for depth in range(4)
    }
    keys = list(parents)
    for step in range(180):
        key = rng.choice(keys)
        cache.put(key, "x" * rng.randrange(220), parent_key=parents[key])
        cache.get(rng.choice(keys))
        stored = {key for key in keys if key in cache}
        assert len(stored) == len(list(tmp_path.glob("*.entry"))) == len(cache) <= 8
        assert cache.nbytes == disk_bytes(tmp_path) <= 480
        for key in stored:
            parent = cache.parent_of(key)
            assert parent is None or parent in stored
        if step % 37 == 36:
            size = cache.nbytes
            cache = DiskKVCache(tmp_path, max_entries=8, max_bytes=480)
            assert cache.nbytes == size
