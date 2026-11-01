import json

from nano_lmcache.datasets import generate_synthetic
from nano_lmcache.datasets.adapters.freeinference import FreeInferenceAdapter
from nano_lmcache.observability import TraceRecorder, TracingStorage
from nano_lmcache.replay import BlockTraceReplayRunner, WorkloadReplayRunner
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.visualization import render_html, write_html


def test_workload_replay_stores_missing_chunks_and_reuses_them():
    report = WorkloadReplayRunner(chunk_size=2).run(
        generate_synthetic("partial_prefix_hit")
    )

    assert [item.matched_chunks for item in report.requests] == [0, 2]
    assert report.requests[0].statuses == ("MISS", "NOT_CHECKED", "NOT_CHECKED")
    assert report.requests[1].statuses == ("HIT", "HIT", "MISS")
    assert report.matched_chunks == 2
    assert any(event.event_type == "chunk_not_checked" for event in report.events)


def test_prefix_branching_replay_keeps_parent_keys_for_the_tree():
    report = WorkloadReplayRunner(chunk_size=2).run(
        generate_synthetic("prefix_branching")
    )

    nodes = {
        event.key: event.parent_key
        for event in report.events
        if event.key is not None
    }
    child_keys = [key for key, parent in nodes.items() if parent is not None]
    assert child_keys
    assert all(nodes[key] in nodes for key in child_keys)


def test_html_escapes_external_request_text_and_merges_tree_nodes():
    report = WorkloadReplayRunner(chunk_size=2).run(
        generate_synthetic("full_prefix_hit")
    )
    payload = report.to_dict()
    payload["requests"][0]["request_id"] = '<img src=x onerror="alert(1)">'
    page = render_html(payload)

    assert "escapeText(r.request_id)" in page
    assert "escapeText(e.request_id)" in page
    assert "previous && previous.parent" in page
    assert "<img src=x onerror" not in page


def test_opaque_block_replay_does_not_treat_block_ids_as_token_ids():
    records = [
        {
            "session_id": "s",
            "requests": [
                {
                    "step": 0,
                    "start_ms": 0,
                    "model": "m",
                    "provider_tokens": {"prompt": 32, "completion": 1},
                    "block_ids": [10, 11],
                },
                {
                    "step": 1,
                    "start_ms": 1,
                    "model": "m",
                    "provider_tokens": {
                        "prompt": 48,
                        "completion": 1,
                        "cached": 32,
                    },
                    "block_ids": [10, 11, 12],
                },
            ],
        }
    ]
    requests = FreeInferenceAdapter().iter_records(records)
    report = BlockTraceReplayRunner(block_tokens=16).run(requests)

    assert report.requests[1].matched_tokens == 32
    assert report.requests[1].observed_cached_tokens == 32
    assert report.requests[1].source == "opaque_block_workload"


def test_html_is_self_contained_and_can_be_written(tmp_path):
    report = WorkloadReplayRunner(chunk_size=2).run(
        generate_synthetic("full_prefix_hit")
    )
    page = render_html(report)
    assert "const DATA =" in page
    assert "https://" not in page
    assert "Prefix Tree" in page

    path = write_html(tmp_path / "trace.html", report)
    assert json.loads(json.dumps({"html": path.read_text()}))["html"].startswith(
        "<!doctype html>"
    )


def test_tracing_storage_records_put_get_and_eviction():
    recorder = TraceRecorder()
    storage = TracingStorage(
        MemoryKVCache(max_entries=1),
        recorder,
        name="memory",
    )

    storage.put("a", "value-a")
    assert storage.get("a") == "value-a"
    storage.put("b", "value-b")

    event_types = [event.event_type for event in recorder.events]
    assert "storage_put" in event_types
    assert "storage_get" in event_types
    assert "eviction" in event_types
