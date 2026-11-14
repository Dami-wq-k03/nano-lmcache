import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_external_evidence_counts_payload_reads_not_presence_checks():
    server_type = runpy.run_path(str(ROOT / "benchmarks/bench_vllm_external.py"))[
        "ObservedServer"
    ]
    server = server_type()
    try:
        server.handle({"op": "contains", "key": "missing"})
        server.handle({"op": "get", "key": "missing"})
        assert server.reads == []
        server.storage.put("root", b"payload")
        server.handle({"op": "get", "key": "root"})
        assert server.reads == ["root"]
    finally:
        server.close()


def test_cold_advice_only_touches_experiment_files(tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "benchmarks/bench_disk_pages.py"))
    (tmp_path / "first").write_bytes(b"1234")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "second").write_bytes(b"12")
    advised = []

    def advise(fd, offset, length, advice):
        advised.append((os.fstat(fd).st_size, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    module["drop_cached_files"](tmp_path)
    assert sorted(advised) == [(2, 0, 0, 4), (4, 0, 0, 4)]


@pytest.mark.parametrize("script", ["bench_vllm_external.py", "bench_pd_handoff.py"])
def test_invalid_benchmark_counts_fail_before_loading_models(script):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / script),
            "--model",
            "does-not-exist",
            "--repeats",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "positive" in result.stderr
    assert "Traceback" not in result.stderr


def test_disk_measure_reports_physical_reads_separately_from_payload_reads(monkeypatch):
    module = runpy.run_path(str(ROOT / "benchmarks/bench_disk_pages.py"))
    measure = module["measure"]
    counters = iter([100, 120, 120, 150])
    monkeypatch.setitem(
        measure.__globals__, "process_read_bytes", lambda: next(counters)
    )
    reader = SimpleNamespace(read_ops=0)
    advised = []

    def read():
        reader.read_ops += 1
        return [b"value"]

    result = measure(read, reader, [b"value"], 2, lambda: advised.append(True))
    assert result["physical_read_bytes"] == [20, 30]
    assert result["payload_read_operations_per_lookup"] == 1
    assert len(result["samples_ms"]) == 2
    assert advised == [True, True]
