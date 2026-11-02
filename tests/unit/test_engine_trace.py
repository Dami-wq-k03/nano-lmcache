"""The engine Trace JSON -> HTML path must work without torch installed."""

import json
import subprocess
import sys

import pytest

from nano_lmcache.observability import TraceRecorder
from nano_lmcache.visualization import load_trace_json, report_from_events, write_html


def engine_events():
    recorder = TraceRecorder(request_id="request-a")
    recorder.record("request_started", metadata={
        "prompt_tokens": 9, "block_size": 4, "model_id": "test-model",
    })
    # Scheduling probes may repeat. They must never inflate committed hits.
    for _ in range(3):
        recorder.record("l2_lookup", chunk_index=1, key="child", result="available")
    recorder.record("l1_hit", chunk_index=0, key="root", metadata={"blocks": 1})
    recorder.record("l2_hit", chunk_index=1, key="child", parent_key="root")
    recorder.record("l2_restore", chunk_index=1, key="child", parent_key="root")
    recorder.record("gpu_import", chunk_index=1, key="child", parent_key="root")
    recorder.record("prefill_step", metadata={"computed_tokens": 1})
    recorder.record("request_completed", metadata={"ttft_ms": 12.5})
    return recorder


@pytest.mark.parametrize("envelope", [False, True])
def test_raw_recorder_and_engine_envelope_normalize_for_html(tmp_path, envelope):
    recorder = engine_events()
    data = recorder.to_list()
    if envelope:
        data = {"format_version": 1, "source": "nanovllm", "events": data}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    report = load_trace_json(path)

    assert report["summary"]["l1_hit_blocks"] == 1
    assert report["summary"]["l2_hit_blocks"] == 1
    assert report["summary"]["hit_rate"] == 2 / 3
    (request,) = report["requests"]
    assert request["statuses"] == ["L1_HIT", "L2_HIT", "COMPUTE"]
    assert request["computed_tokens"] == 1
    assert request["ttft_ms"] == 12.5
    html = write_html(tmp_path / "trace.html", report).read_text()
    assert "GPU prefix cache reused" in html
    assert "KV copy into engine cache enqueued" in html
    assert report["events"] == recorder.to_list()


def test_trace_json_cli_runs_without_loading_engine_dependencies(tmp_path):
    source = tmp_path / "engine.json"
    engine_events().write_json(source)
    output = tmp_path / "engine.html"
    result = subprocess.run(
        [sys.executable, "-m", "nano_lmcache.visualize",
         "--trace-json", str(source), "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert "1 requests" in result.stdout
    assert output.exists()


@pytest.mark.parametrize("data, message", [
    ({"format_version": 2, "events": []}, "format_version"),
    ({"events": {}}, "event array"),
    ([{}], "invalid event"),
    (["bad"], "must be an object"),
])
def test_trace_import_rejects_malformed_input(tmp_path, data, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_trace_json(path)


def test_trace_import_rejects_out_of_order_events(tmp_path):
    events = engine_events().to_list()
    events.reverse()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_trace_json(path)


def test_probe_without_actual_fetch_is_not_a_hit_or_ttft():
    recorder = TraceRecorder(request_id="request-a")
    recorder.record("l2_lookup", metadata={"hit_blocks": 3})
    report = report_from_events(recorder.events)
    assert report["summary"]["matched_chunks"] == 0
    assert report["requests"][0]["ttft_ms"] is None


def test_unknown_event_and_external_text_survive_as_data(tmp_path):
    recorder = TraceRecorder(request_id='external <text> & "quotes"')
    recorder.record("custom_event", metadata={"label": "a<b"})
    path = tmp_path / "trace.json"
    recorder.write_json(path)
    report = load_trace_json(path)
    assert report["events"] == recorder.to_list()
    html = write_html(tmp_path / "trace.html", report).read_text()
    assert "external <text>" not in html
    assert "\\u003ctext\\u003e" in html
