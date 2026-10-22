import json
import os
import subprocess
import sys

from nano_lmcache.inspect import inspect_directory
from nano_lmcache.storage.disk import DiskKVCache


def test_inspector_reports_entries_tree_bytes_and_lru(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put("root", "a")
    cache.put("child", "bb", parent_key="root")
    cache.put("other", "ccc")
    # Give stable, distinct recency timestamps without reading through the cache.
    os.utime(cache._path("root"), ns=(100, 100))
    os.utime(cache._path("child"), ns=(200, 200))
    os.utime(cache._path("other"), ns=(150, 150))

    report = inspect_directory(tmp_path)

    assert report["entries"] == 3
    assert report["total_bytes"] == sum(
        path.stat().st_size for path in tmp_path.glob("*.entry")
    )
    assert [node["key"] for node in report["tree"]] == ["other", "root"]
    root_node = next(node for node in report["tree"] if node["key"] == "root")
    assert root_node["depth"] == 0
    assert root_node["children"] == 1
    assert root_node["children_entries"][0]["key"] == "child"
    assert [item["key"] for item in report["leaves_lru"]] == [
        "other",
        "child",
    ]


def test_inspector_cli_json(tmp_path):
    DiskKVCache(tmp_path).put("root", "value")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nano_lmcache.inspect",
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["entries"] == 1
    assert report["tree"][0]["key"] == "root"


def test_inspector_does_not_modify_files_or_mtimes(tmp_path):
    cache = DiskKVCache(tmp_path)
    cache.put("root", "value")
    cache.put("child", "value-2", parent_key="root")
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
    }

    inspect_directory(tmp_path)

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.iterdir()
    }
    assert after == before


def test_inspector_reports_invalid_files_without_deleting_them(tmp_path):
    bad_file = tmp_path / ("bad.entry")
    bad_file.write_bytes(b"broken")
    before = bad_file.read_bytes()

    report = inspect_directory(tmp_path)

    assert report["invalid_files"] == ["bad.entry"]
    assert bad_file.read_bytes() == before
