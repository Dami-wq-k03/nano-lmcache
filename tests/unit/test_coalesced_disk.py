import json
import os
import subprocess
import sys
import zlib

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.experimental.dataplane.layout import KVLayout
from nano_lmcache.storage.coalesced import CoalescedDiskKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.serializers import JsonSerializer


def write_index(directory, rows):
    payload = json.dumps({"version": 1, "entries": rows}).encode()
    (directory / "pages.index").write_bytes(
        str(zlib.crc32(payload)).encode() + b"\n" + payload
    )


def test_packed_pages_match_experimental_layout_and_storage_contract(tmp_path):
    layout = KVLayout("test-model", 2, 2, 4, 8, "float16")
    page = bytes(range(256)) * (layout.page_nbytes // 256)
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert isinstance(disk, CacheStorage)
        assert disk.put_many([("a", page, None), ("b", page, "a")]) == [True, True]
        assert (tmp_path / "pages.data").read_bytes() == page + page
        assert disk.get_many(["b", "missing", "a", "b"]) == [page, None, page, page]
        assert disk.read_ops == 1
        assert disk.read_bytes == len(page) * 2
        assert disk.parent_of("b") == "a"
        assert disk.nbytes == len(page) * 2
    with CoalescedDiskKVCache(tmp_path) as reopened:
        assert reopened.get_many(["a", "b"]) == [page, page]


def test_json_prefix_cache_survives_restart(tmp_path):
    with CoalescedDiskKVCache(tmp_path, serializer=JsonSerializer()) as disk:
        cache = ChunkedPrefixCache(chunk_size=2, namespace="model", storage=disk)
        cache.store([1, 2, 3, 4, 5, 6], ["a", "b", "c"])
    with CoalescedDiskKVCache(tmp_path, serializer=JsonSerializer()) as disk:
        cache = ChunkedPrefixCache(chunk_size=2, namespace="model", storage=disk)
        assert cache.lookup([1, 2, 3, 4, 5, 6]).values == ("a", "b", "c")


def test_prefix_safe_eviction_byte_capacity_and_overwrite(tmp_path):
    with CoalescedDiskKVCache(tmp_path, max_bytes=6) as disk:
        disk.put_many([("a", b"aa", None), ("b", b"bb", "a"), ("c", b"cc", "b")])
        assert not disk.try_put("d", b"dd", parent_key="c")
        assert not disk.try_put("orphan", b"x", parent_key="missing")
        assert list(disk.keys()) == ["a", "b", "c"]
        disk.put("x", b"xx")
        assert disk.get_many(["a", "b", "c", "x"]) == [b"aa", b"bb", None, b"xx"]
        disk.put("b", b"BBB", parent_key="x")
        assert disk.parent_of("b") == "a"  # overwrite cannot reparent
        assert disk.get("x") is None
        assert disk.nbytes == 5
        assert (tmp_path / "pages.data").stat().st_size == 11  # no compaction
        assert disk.stats.evictions == 2
    with CoalescedDiskKVCache(tmp_path, max_bytes=6) as disk:
        assert disk.get_many(["a", "b", "c", "x"]) == [b"aa", b"BBB", None, None]


def test_rejected_overwrite_preserves_old_payload(tmp_path):
    with CoalescedDiskKVCache(tmp_path, max_bytes=2) as disk:
        disk.put("a", b"a")
        disk.put("b", b"b", parent_key="a")
        assert not disk.try_put("b", b"bbb")
        assert disk.get_many(["a", "b"]) == [b"a", b"b"]


@pytest.mark.parametrize("value", [None, object(), "not packed bytes"])
def test_batch_encoding_failure_evicts_nothing(tmp_path, value):
    with CoalescedDiskKVCache(tmp_path, max_entries=1) as disk:
        disk.put("a", b"a")
        with pytest.raises((TypeError, ValueError)):
            disk.put_many([("b", b"b", None), ("c", value, None)])
        assert disk.get("a") == b"a"
        assert disk.stats.evictions == 0
        assert (tmp_path / "pages.data").read_bytes() == b"a"


def test_batch_admissions_do_not_claim_all_entries_survive(tmp_path):
    with CoalescedDiskKVCache(tmp_path, max_entries=1) as disk:
        assert disk.put_many([("a", b"a", None), ("b", b"b", None)]) == [True, True]
        assert disk.get_many(["a", "b"]) == [None, b"b"]
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get("a") is None  # dead append bytes cannot resurrect


@pytest.mark.parametrize("policy", ["lru", "lfu"])
def test_recovery_restores_read_recency_without_fake_hits(tmp_path, policy):
    with CoalescedDiskKVCache(tmp_path, max_entries=3, eviction_policy=policy) as disk:
        disk.put_many([(key, key.encode(), None) for key in "abc"])
        disk.get("a")
        disk.get("a")
    with CoalescedDiskKVCache(tmp_path, max_entries=3, eviction_policy=policy) as disk:
        disk.put("d", b"d")
        assert "b" not in disk
        assert "a" in disk


def test_coalescing_respects_gaps_and_read_size(tmp_path):
    with CoalescedDiskKVCache(tmp_path, max_read_bytes=4) as disk:
        disk.put_many([(key, key.encode() * 2, None) for key in "abcd"])
        assert disk.get_many(["d", "a", "b", "c"]) == [b"dd", b"aa", b"bb", b"cc"]
        assert disk.read_ops == 2
        disk.read_ops = disk.read_bytes = 0
        assert disk.get_many(["a", "c"]) == [b"aa", b"cc"]
        assert disk.read_ops == 2
        assert disk.read_bytes == 4  # don't read unrequested b
        assert disk.get_many([]) == []
        assert disk.get_many(["absent"]) == [None]
        assert disk.read_ops == 2


def test_single_page_may_exceed_coalesced_read_limit_and_be_empty(tmp_path):
    with CoalescedDiskKVCache(tmp_path, max_read_bytes=1) as disk:
        disk.put("a", b"long")
        disk.put("empty", b"")
        assert disk.get_many(["empty", "a"]) == [b"", b"long"]


@pytest.mark.parametrize("damage", ["flip", "truncate", "read-error"])
def test_live_corruption_is_a_miss_and_persists_subtree_removal(
    tmp_path, monkeypatch, damage
):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put_many([("a", b"aa", None), ("b", b"bb", "a"), ("c", b"cc", "b")])
        if damage == "read-error":

            def fail(*args):
                raise OSError("disk read failed")

            with monkeypatch.context() as patch:
                patch.setattr(os, "pread", fail)
                assert disk.get("b") is None
        else:
            (tmp_path / "pages.data").write_bytes(
                b"aaBc" if damage == "flip" else b"aab"
            )
            assert disk.get_many(["c", "b", "a"]) == [None, None, b"aa"]
        assert list(disk.keys()) == ["a"]
        assert disk.stats.corrupted_entries >= 1
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get_many(["b", "c"]) == [None, None]


def test_short_pread_keeps_complete_pages_in_same_range(tmp_path, monkeypatch):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put_many([("a", b"aa", None), ("b", b"bb", "a"), ("c", b"cc", "b")])
        real_pread = os.pread
        monkeypatch.setattr(
            os, "pread", lambda fd, n, offset: real_pread(fd, 3, offset)
        )
        assert disk.get_many(["a", "b", "c"]) == [b"aa", None, None]
        assert disk.stats.corrupted_entries == 1


def test_child_is_not_returned_before_corrupt_overwritten_parent(tmp_path):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put_many([("a", b"a", None), ("b", b"b", "a")])
        disk.put("a", b"new")  # parent now physically AFTER child
        (tmp_path / "pages.data").write_bytes(b"abBAD")
        assert disk.get_many(["b", "a", "b"]) == [None, None, None]
        assert len(disk) == 0


def test_decode_failure_is_a_miss(tmp_path):
    class BadDecoder(JsonSerializer):
        def loads(self, data):
            raise ValueError("invalid serialized page")

    with CoalescedDiskKVCache(tmp_path, serializer=BadDecoder()) as disk:
        disk.put("a", {"page": 1})
        assert disk.get("a") is None
        assert disk.stats.corrupted_entries == 1


@pytest.mark.parametrize("damage", ["checksum", "truncated", "missing"])
def test_bad_metadata_fails_closed_without_resurrecting_payloads(tmp_path, damage):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put("old", b"old")
    path = tmp_path / "pages.index"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        data = path.read_bytes()
        path.write_bytes(b"0\n" + data.partition(b"\n")[2])
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get("old") is None
        disk.put("new", b"new")
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get_many(["old", "new"]) == [None, b"new"]


@pytest.mark.parametrize(
    "rows",
    [
        [["bad", None, -1, 1, 0]],
        [["bad", None, True, 1, 0]],
        [["bad", None, 0, 1, -1]],
        [["bad", [], 0, 1, 0]],
        [["bad", None, 0, 1, 0], ["bad", None, 0, 1, 0]],
        [["bad"]],
    ],
)
def test_malformed_index_records_are_misses(tmp_path, rows):
    write_index(tmp_path, rows)
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert len(disk) == 0


def test_missing_parent_cycles_and_short_extents_not_restored(tmp_path):
    (tmp_path / "pages.data").write_bytes(b"abc")
    write_index(
        tmp_path,
        [
            ["root", None, 0, 1, zlib.crc32(b"a")],
            ["short", "root", 2, 2, 0],
            ["child", "short", 1, 1, zlib.crc32(b"b")],
            ["orphan", "missing", 1, 1, zlib.crc32(b"b")],
            ["cycle-a", "cycle-b", 0, 1, 0],
            ["cycle-b", "cycle-a", 0, 1, 0],
        ],
    )
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert list(disk.keys()) == ["root"]
        assert disk.get("root") == b"a"


def test_checksum_verified_after_restart(tmp_path):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put_many([("a", b"a", None), ("b", b"b", "a")])
    (tmp_path / "pages.data").write_bytes(b"Xb")
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get_many(["b", "a"]) == [None, None]


def test_interrupted_publication_keeps_old_committed_prefix(tmp_path, monkeypatch):
    disk = CoalescedDiskKVCache(tmp_path, max_entries=2)
    disk.put_many([("a", b"old", None), ("b", b"child", "a")])
    real_replace = os.replace

    def fail(src, dst):
        raise OSError("interrupted rename")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="rename"):
        disk.put("x", b"uncommitted")  # would evict b
    with pytest.raises(RuntimeError, match="failed"):
        disk.get("a")
    disk.close()  # must not publish the mutated in-memory index
    monkeypatch.setattr(os, "replace", real_replace)
    with CoalescedDiskKVCache(tmp_path) as reopened:
        assert reopened.get_many(["a", "b", "x"]) == [b"old", b"child", None]
        assert not (tmp_path / "pages.index.tmp").exists()


def test_process_crash_after_data_fsync_ignores_unpublished_append(tmp_path):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put("a", b"old")
    script = """
import os, sys
from nano_lmcache.storage.coalesced import CoalescedDiskKVCache
storage = CoalescedDiskKVCache(sys.argv[1])
storage._checkpoint = lambda: os._exit(0)
storage.put('a', b'new-but-not-published')
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get("a") == b"old"
        disk.put("b", b"b", parent_key="a")
    with CoalescedDiskKVCache(tmp_path) as disk:
        assert disk.get_many(["a", "b"]) == [b"old", b"b"]


def test_commit_fsyncs_data_then_index_then_directory(tmp_path, monkeypatch):
    with CoalescedDiskKVCache(tmp_path) as disk:
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            events.append("fsync")
            real_fsync(fd)

        def replace(src, dst):
            events.append("replace")
            real_replace(src, dst)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)
        disk.put_many([("a", b"a", None), ("b", b"b", "a")])
        assert events == ["fsync", "fsync", "replace", "fsync"]


def test_restart_smaller_capacity_is_error_and_does_not_rewrite_index(tmp_path):
    with CoalescedDiskKVCache(tmp_path) as disk:
        disk.put_many([("a", b"aa", None), ("b", b"bb", "a")])
    original = (tmp_path / "pages.index").read_bytes()
    for limits in ({"max_entries": 1}, {"max_bytes": 3}):
        with pytest.raises(ValueError):
            CoalescedDiskKVCache(tmp_path, **limits)
        assert (tmp_path / "pages.index").read_bytes() == original


def test_close_is_idempotent_and_releases_descriptor(tmp_path):
    disk = CoalescedDiskKVCache(tmp_path)
    disk.put("../escape", b"safe")
    assert {path.name for path in tmp_path.iterdir()} == {"pages.data", "pages.index"}
    disk.close()
    disk.close()
    assert disk._file.closed
    with pytest.raises(RuntimeError, match="closed"):
        disk.get("../escape")
    with pytest.raises(RuntimeError, match="closed"):
        disk.put("x", b"x")


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_fsync_failure_requires_reopen_and_never_publishes_partial_pages(
    tmp_path, monkeypatch, stage
):
    disk = CoalescedDiskKVCache(tmp_path)
    disk.put("a", b"old")
    real_fsync = os.fsync
    calls = 0

    def fail_at_stage(fd):
        nonlocal calls
        calls += 1
        if calls == stage:
            raise OSError("fsync failed")
        real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_at_stage)
        with pytest.raises(OSError, match="fsync"):
            disk.put("a", b"new")
        with pytest.raises(RuntimeError, match="failed"):
            disk.put("b", b"b")
        disk.close()
    with CoalescedDiskKVCache(tmp_path) as reopened:
        # Only the directory fsync failure occurs AFTER the atomic rename.
        assert reopened.get("a") == (b"new" if stage == 3 else b"old")


def test_short_write_leaves_old_index_authoritative(tmp_path, monkeypatch):
    disk = CoalescedDiskKVCache(tmp_path)
    disk.put("a", b"old")
    real_write = disk._file.write

    def short_write(payload):
        return real_write(payload[:1])

    with monkeypatch.context() as patch:
        patch.setattr(disk._file, "write", short_write)
        with pytest.raises(OSError, match="short page write"):
            disk.put("a", b"new")
        disk.close()
    with CoalescedDiskKVCache(tmp_path) as reopened:
        assert reopened.get("a") == b"old"


def test_close_error_still_closes_descriptor(tmp_path, monkeypatch):
    disk = CoalescedDiskKVCache(tmp_path)
    disk.put("a", b"a")

    def fail(fd):
        raise OSError("close checkpoint failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="checkpoint"):
            disk.close()
        assert disk._file.closed
        disk.close()
    with CoalescedDiskKVCache(tmp_path) as reopened:
        assert reopened.get("a") == b"a"


@pytest.mark.parametrize(
    "limits", [{"max_read_bytes": 0}, {"max_entries": 0}, {"max_bytes": 0}]
)
def test_invalid_limits(tmp_path, limits):
    with pytest.raises(ValueError):
        CoalescedDiskKVCache(tmp_path, **limits)
