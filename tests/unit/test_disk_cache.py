import json
import os

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.serializers import JsonSerializer


def entry_files(directory):
    return sorted(p for p in directory.iterdir() if p.suffix == ".entry")


# ---- basic contract ---------------------------------------------------


def test_satisfies_storage_protocol(tmp_path):
    assert isinstance(DiskKVCache(tmp_path), CacheStorage)


def test_put_then_get(tmp_path):
    cache = DiskKVCache(tmp_path)

    cache.put("a", "kv-a")

    assert cache.get("a") == "kv-a"
    assert cache.get("missing") is None
    assert len(cache) == 1


def test_creates_the_directory(tmp_path):
    cache_dir = tmp_path / "nested" / "cache"

    DiskKVCache(cache_dir).put("a", "kv-a")

    assert cache_dir.is_dir()


def test_one_file_per_entry(tmp_path):
    cache = DiskKVCache(tmp_path)

    cache.put("a", 1)
    cache.put("b", 2, parent_key="a")

    assert len(entry_files(tmp_path)) == 2


def test_overwrite_replaces_the_payload(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put("a", "old")

    cache.put("a", "new")

    assert cache.get("a") == "new"
    assert len(entry_files(tmp_path)) == 1


def test_keys_cannot_escape_the_directory(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = DiskKVCache(cache_dir)

    cache.put("../../escape", "x")

    assert cache.get("../../escape") == "x"
    assert list(tmp_path.iterdir()) == [cache_dir]


def test_missing_parent_is_rejected(tmp_path):
    cache = DiskKVCache(tmp_path)

    cache.put("orphan", 1, parent_key="never-stored")

    assert cache.get("orphan") is None
    assert entry_files(tmp_path) == []
    assert cache.stats.rejected_missing_parent == 1


# ---- persistence across restarts --------------------------------------


def test_entries_survive_a_restart(tmp_path):
    DiskKVCache(tmp_path).put("a", {"layers": [1, 2, 3]})

    reopened = DiskKVCache(tmp_path)

    assert reopened.get("a") == {"layers": [1, 2, 3]}
    assert len(reopened) == 1


def test_parent_links_survive_a_restart(tmp_path):
    first = DiskKVCache(tmp_path)
    first.put("a", 1)
    first.put("b", 2, parent_key="a")

    reopened = DiskKVCache(tmp_path)

    assert reopened.parent_of("b") == "a"
    assert reopened.parent_of("a") is None


def test_prefix_cache_hits_after_a_restart(tmp_path):
    tokens = [101, 205, 307, 409, 511, 613]
    writer = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=DiskKVCache(tmp_path)
    )
    writer.store(tokens, ["kv-0", "kv-1", "kv-2"])

    # A new process: fresh objects, same directory.
    reader = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=DiskKVCache(tmp_path)
    )
    result = reader.lookup(tokens)

    assert result.matched_chunks == 3
    assert result.values == ("kv-0", "kv-1", "kv-2")


def test_recency_survives_a_restart(tmp_path):
    names = ["a", "b", "c", "d", "e"]
    first = DiskKVCache(tmp_path, max_entries=5)
    for name in names:
        first.put(name, name)
    # Use them in a scrambled order: c, a, e, b, d (d most recent).
    for name in ["c", "a", "e", "b", "d"]:
        first.get(name)

    reopened = DiskKVCache(tmp_path, max_entries=5)
    evicted = []
    for i in range(5):
        before = {n for n in names if n in reopened}
        reopened.put(f"new-{i}", i)
        evicted.extend(sorted(before - {n for n in names if n in reopened}))

    # Eviction order must be the usage order from before the restart.
    assert evicted == ["c", "a", "e", "b", "d"]


def test_parent_protection_survives_a_restart(tmp_path):
    first = DiskKVCache(tmp_path, max_entries=3)
    first.put("a", 1)
    first.put("b", 2, parent_key="a")
    first.put("c", 3, parent_key="b")
    # A later hit on the root makes a newer than its descendants.
    first.get("a")

    reopened = DiskKVCache(tmp_path, max_entries=3)
    reopened.put("x", 0)

    # c is the only leaf, so it goes, even though a and b are older.
    assert reopened.get("c") is None
    assert reopened.get("a") == 1
    assert reopened.get("b") == 2


def test_reopening_with_less_capacity_than_stored_is_an_error(tmp_path):
    first = DiskKVCache(tmp_path)
    for name in "abc":
        first.put(name, name)

    with pytest.raises(ValueError, match="max_entries"):
        DiskKVCache(tmp_path, max_entries=2)


# ---- crash safety -----------------------------------------------------


def test_leftover_temp_files_are_removed_on_open(tmp_path):
    DiskKVCache(tmp_path).put("a", 1)
    # What a crash in the middle of a write leaves behind.
    (tmp_path / "half-written.tmp").write_bytes(b"garbage")

    reopened = DiskKVCache(tmp_path)

    assert not (tmp_path / "half-written.tmp").exists()
    assert reopened.get("a") == 1


def test_writes_go_through_a_temp_file(tmp_path, monkeypatch):
    replaced = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((os.fspath(src), os.fspath(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    DiskKVCache(tmp_path).put("a", 1)

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src.endswith(".tmp")
    assert dst.endswith(".entry")


def test_orphaned_entries_are_removed_on_open(tmp_path):
    first = DiskKVCache(tmp_path)
    first.put("a", 1)
    first.put("b", 2, parent_key="a")
    # Simulate losing the parent's file outside the cache's control.
    parent_file = next(
        p for p in entry_files(tmp_path) if b'"key":"a"' in p.read_bytes()
    )
    parent_file.unlink()

    reopened = DiskKVCache(tmp_path)

    assert reopened.get("b") is None
    assert entry_files(tmp_path) == []


def test_unreadable_entries_are_removed_on_open(tmp_path):
    DiskKVCache(tmp_path).put("a", 1)
    (tmp_path / ("0" * 64 + ".entry")).write_bytes(b"not a valid entry")

    reopened = DiskKVCache(tmp_path)

    assert len(reopened) == 1
    assert reopened.get("a") == 1


# ---- eviction ---------------------------------------------------------


def test_eviction_deletes_files(tmp_path):
    cache = DiskKVCache(tmp_path, max_entries=2)

    for name in "abcd":
        cache.put(name, name)

    assert len(entry_files(tmp_path)) == 2
    assert cache.stats.evictions == 2
    assert cache.get("a") is None
    assert cache.get("d") == "d"


def test_long_chain_keeps_the_prefix_that_fits(tmp_path):
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=DiskKVCache(tmp_path, max_entries=2),
    )
    tokens = [1, 2, 3, 4, 5, 6]

    cache.store(tokens, ["kv-0", "kv-1", "kv-2"])

    assert cache.lookup(tokens).matched_chunks == 2


# ---- serialization ----------------------------------------------------


def test_default_serializer_is_json(tmp_path):
    cache = DiskKVCache(tmp_path)

    assert isinstance(cache.serializer, JsonSerializer)


def test_default_serializer_rejects_non_json_values(tmp_path):
    cache = DiskKVCache(tmp_path)

    with pytest.raises(TypeError):
        cache.put("a", object())

    assert cache.get("a") is None
    assert len(cache) == 0
    assert entry_files(tmp_path) == []


def test_failed_encoding_evicts_nothing(tmp_path):
    cache = DiskKVCache(tmp_path, max_entries=1)
    cache.put("a", 1)

    with pytest.raises(TypeError):
        cache.put("b", object())

    assert cache.get("a") == 1
    assert cache.stats.evictions == 0


def test_custom_serializer_is_used(tmp_path):
    class UpperSerializer:
        def dumps(self, value: object) -> bytes:
            return json.dumps(str(value).upper()).encode()

        def loads(self, data: bytes) -> object:
            return json.loads(data)

    cache = DiskKVCache(tmp_path, serializer=UpperSerializer())
    cache.put("a", "hello")

    assert cache.get("a") == "HELLO"


def test_entry_whose_name_does_not_match_its_key_is_removed(tmp_path):
    DiskKVCache(tmp_path).put("a", 1)
    original = entry_files(tmp_path)[0]
    # A copy under another name claims the same key.
    copy = tmp_path / ("f" * 64 + ".entry")
    copy.write_bytes(original.read_bytes())

    reopened = DiskKVCache(tmp_path)

    assert not copy.exists()
    assert reopened.get("a") == 1


def test_contains_does_not_change_recency(tmp_path):
    cache = DiskKVCache(tmp_path, max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)

    assert "a" in cache
    assert "missing" not in cache
    cache.put("c", 3)

    # Checking membership is not a use: a is still the oldest.
    assert "a" not in cache
