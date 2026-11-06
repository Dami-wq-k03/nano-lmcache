"""Streaming must bound retained detail without changing simulated results."""

import gc
import json
import subprocess
import sys
import tracemalloc
from dataclasses import replace

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.datasets import (
    WorkloadRequest,
    generate_synthetic,
    iter_workload_jsonl,
)
from nano_lmcache.datasets.adapters.freeinference import iter_freeinference_jsonl
from nano_lmcache.observability import TraceRecorder, TracingStorage
from nano_lmcache.replay import BlockTraceReplayRunner, WorkloadReplayRunner
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.visualization import render_html


def requests(count):
    for index in range(count):
        yield WorkloadRequest(
            request_id=f"r{index:08d}", session_id="s", arrival_time_ms=index,
            model_id="m", prompt_tokens=32, prefix_blocks=(10, 11),
            observed_cached_tokens=16,
        )


def test_recorder_samples_with_original_sequences_and_exact_counters():
    recorder = TraceRecorder(max_events=3, sample_every=2, request_id="r")
    for _ in range(10):
        recorder.record("storage_get", result="hit", metadata={"backend": "disk"})
    assert [event.sequence for event in recorder.events] == [0, 2, 4]
    assert recorder.total_events == 10
    assert recorder.event_counts == {"storage_get": 10}
    assert recorder.storage_hits == {"disk": 10}
    recorder.clear()
    assert recorder.total_events == 0
    assert recorder.event_counts == {}
    assert recorder.storage_hits == {}
    assert recorder.record("chunk_hit").sequence == 0


@pytest.mark.parametrize("kwargs", [
    {"max_events": -1}, {"max_events": True},
    {"sample_every": 0}, {"sample_every": False},
])
def test_recorder_rejects_invalid_retention_options(kwargs):
    with pytest.raises(ValueError):
        TraceRecorder(**kwargs)


def test_streaming_consumes_one_request_at_a_time_and_keeps_exact_summary():
    recorder = TraceRecorder(max_events=4, sample_every=3)

    def source():
        for index, request in enumerate(requests(20)):
            # Eager list()/sort() would request the next item before lookup.
            assert recorder.total_events >= index
            yield request

    report = BlockTraceReplayRunner(
        recorder=recorder, storage=MemoryKVCache(max_entries=2),
    ).run(source(), sort_by_arrival=False, max_requests=2)
    data = report.to_dict()
    assert len(report.requests) == 2
    assert len(report.events) == 4
    assert data["summary"]["requests"] == 20
    assert report.total_chunks == 40
    assert report.matched_chunks == 38
    assert report.observed_cached_tokens == 320
    assert data["summary"]["matched_tokens"] == 608
    assert data["retention"]["requests_omitted"] == 18
    assert data["retention"]["events_omitted"] == recorder.total_events - 4
    assert data["retention"]["sample_every"] == 3
    assert data["summary"]["event_counts"]["chunk_hit"] == 38


def test_sampling_and_detail_caps_do_not_change_token_replay_or_tier_counts():
    full = WorkloadReplayRunner(chunk_size=2).run(
        generate_synthetic("partial_prefix_hit")
    )
    recorder = TraceRecorder(max_events=0)
    storage = TracingStorage(MemoryKVCache(max_entries=5), recorder, name="memory")
    bounded = WorkloadReplayRunner(
        chunk_size=2, recorder=recorder, storage=storage,
    ).run(iter(generate_synthetic("partial_prefix_hit")),
          sort_by_arrival=False, max_requests=0)
    assert bounded.requests == bounded.events == ()
    assert bounded.total_chunks == full.total_chunks
    assert bounded.matched_chunks == full.matched_chunks
    assert bounded.hit_rate == full.hit_rate
    assert bounded.to_dict()["summary"]["storage_hits"] == {"memory": 2}
    assert bounded.to_dict()["summary"]["requests"] == 2


def test_streaming_rejects_unsorted_input_instead_of_silently_reordering():
    source = list(requests(3))
    with pytest.raises(ValueError, match="order"):
        BlockTraceReplayRunner().run(reversed(source), sort_by_arrival=False)
    report = BlockTraceReplayRunner().run(reversed(source))
    assert [r.request_id for r in report.requests] == [r.request_id for r in source]


def test_opaque_replay_uses_bounded_prefix_safe_storage_and_model_keys():
    source = list(requests(3))
    source[1] = WorkloadRequest(
        request_id="r00000001", session_id="s", arrival_time_ms=1,
        model_id="other-model", prompt_tokens=32, prefix_blocks=(10, 11),
    )
    storage = MemoryKVCache(max_entries=2)
    report = BlockTraceReplayRunner(storage=storage).run(source)
    assert [r.matched_chunks for r in report.requests] == [0, 0, 0]
    assert len(storage) == 2
    assert storage.stats.evictions == 4
    for key in storage.keys():
        parent = storage.parent_of(key)
        assert parent is None or parent in storage


def test_each_replay_report_describes_its_own_run_with_a_warm_cache():
    runner = BlockTraceReplayRunner(recorder=TraceRecorder(max_events=3))
    cold = runner.run(requests(1), max_requests=0)
    warm = runner.run(requests(1), max_requests=0)
    assert cold.matched_chunks == 0
    assert warm.matched_chunks == 2
    assert warm.to_dict()["summary"]["event_counts"].get("chunk_miss", 0) == 0
    assert cold.to_dict()["summary"]["event_counts"]["chunk_miss"] == 1


def test_retained_memory_does_not_grow_with_workload_length():
    def peak(count):
        gc.collect()
        tracemalloc.start()
        try:
            report = BlockTraceReplayRunner(
                storage=MemoryKVCache(max_entries=4),
                recorder=TraceRecorder(max_events=20),
            ).run(requests(count), sort_by_arrival=False, max_requests=3)
            assert report.to_dict()["summary"]["requests"] == count
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    small = peak(500)
    large = peak(5000)
    assert large < small + 256_000


def test_freeinference_limit_does_not_parse_a_later_record(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"session_id": "s", "requests": [{
        "step": 0, "start_ms": 0, "model": "m", "block_ids": [10, 11],
        "provider_tokens": {"prompt": 32},
    }]}) + "\nnot valid json\n")
    assert len(list(iter_freeinference_jsonl(path, limit=1))) == 1
    assert list(iter_freeinference_jsonl(path, limit=0)) == []


def test_cli_streams_workload_jsonl_and_exports_complete_summary(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(r.to_dict()) for r in requests(25)))
    output = tmp_path / "viewer.html"
    summary = tmp_path / "summary.json"
    result = subprocess.run([
        sys.executable, "-m", "nano_lmcache.visualize",
        "--workload-jsonl", str(path), "--stream", "--opaque-blocks",
        "--max-entries", "2", "--max-events", "5", "--max-requests", "2",
        "--sample-every", "2", "--summary-json", str(summary),
        "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(summary.read_text())
    assert data["summary"]["requests"] == 25
    assert data["summary"]["matched_chunks"] == 48
    assert data["summary"]["observed_cached_tokens"] == 400
    assert data["retention"]["events_retained"] == 5
    assert "events" not in data
    assert "25 requests" in result.stdout
    page = output.read_text()
    assert "Displayed detail" in page
    assert "sample_every" in page


def test_viewer_explains_that_sampled_tree_is_not_complete_cache_state():
    report = BlockTraceReplayRunner(recorder=TraceRecorder(max_events=0)).run(
        requests(3), max_requests=0,
    )
    page = render_html(report)
    assert "retained events" in page
    assert "observed_cached_tokens" in page


def test_token_replay_preserves_an_explicit_empty_recorder_on_a_supplied_cache():
    recorder = TraceRecorder(max_events=0)
    cache = ChunkedPrefixCache(chunk_size=2, namespace="m", trace_recorder=recorder)
    report = WorkloadReplayRunner(chunk_size=2, cache=cache, recorder=recorder).run(
        generate_synthetic("full_prefix_hit"), max_requests=0,
    )
    assert report.to_dict()["summary"]["event_counts"]["chunk_hit"] == 2


def test_token_workloads_isolate_models_in_a_shared_backend():
    storage = MemoryKVCache(max_entries=10)
    source = generate_synthetic("full_prefix_hit")
    first = WorkloadReplayRunner(chunk_size=2, storage=storage).run(source)
    other = [replace(r, model_id="other-model") for r in source]
    second = WorkloadReplayRunner(chunk_size=2, storage=storage).run(other)
    assert first.matched_chunks == second.matched_chunks == 2
    assert second.requests[0].matched_chunks == 0
    with pytest.raises(ValueError, match="one model"):
        WorkloadReplayRunner(chunk_size=2).run([source[0], other[1]])


def test_capacity_changes_hits_even_when_all_event_detail_is_omitted():
    reports = [WorkloadReplayRunner(
        chunk_size=2,
        storage=MemoryKVCache(max_entries=capacity),
        recorder=TraceRecorder(max_events=0),
    ).run(generate_synthetic("lru_eviction"), max_requests=0)
        for capacity in (2, 3)]
    assert [r.hit_rate for r in reports] == [0, 0.25]


def test_workload_jsonl_limits_and_contextual_validation(tmp_path):
    path = tmp_path / "workload.jsonl"
    path.write_text(json.dumps(next(requests(1)).to_dict()) + "\n{bad\n")
    assert len(list(iter_workload_jsonl(path, limit=1))) == 1
    assert list(iter_workload_jsonl(path, limit=0)) == []
    with pytest.raises(ValueError, match="line 2"):
        list(iter_workload_jsonl(path))
    with pytest.raises(ValueError, match="limit"):
        list(iter_workload_jsonl(path, limit=-1))


@pytest.mark.parametrize("option,value", [
    ("--max-events", "-1"), ("--max-requests", "-1"),
    ("--sample-every", "0"), ("--max-entries", "0"),
])
def test_cli_rejects_invalid_budgets(tmp_path, option, value):
    output = tmp_path / "invalid.html"
    result = subprocess.run([
        sys.executable, "-m", "nano_lmcache.visualize", "--scenario", "cold_cache",
        option, value, "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be" in result.stderr
    assert not output.exists()


def test_cli_engine_import_caps_detail_after_computing_exact_summary(tmp_path):
    recorder = TraceRecorder(request_id="r")
    recorder.record("request_started", metadata={"block_size": 2, "prompt_tokens": 4})
    recorder.record("l1_hit", chunk_index=0)
    recorder.record("request_completed")
    source = tmp_path / "engine.json"
    recorder.write_json(source)
    output = tmp_path / "engine.html"
    summary = tmp_path / "summary.json"
    result = subprocess.run([
        sys.executable, "-m", "nano_lmcache.visualize", "--trace-json", str(source),
        "--max-events", "0", "--max-requests", "0", "--output", str(output),
        "--summary-json", str(summary),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(summary.read_text())
    assert data["summary"]["l1_hit_blocks"] == 1
    assert data["summary"]["hit_rate"] == 0.5
    assert data["retention"]["events_retained"] == 0
    assert data["retention"]["events_total"] == 3
