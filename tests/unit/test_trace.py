import json

import pytest

from nano_lmcache.observability import TraceEvent, TraceEventType, TraceRecorder


def test_trace_event_contains_schema_fields_and_serializes():
    event = TraceEvent(
        sequence=3,
        request_id="request-a",
        event_type=TraceEventType.CHUNK_HIT,
        chunk_index=1,
        key="key-1",
        parent_key="key-0",
        result="hit",
        duration_ns=42,
        metadata={"backend": "memory"},
    )

    assert event.event_type == "chunk_hit"
    assert event.to_dict() == {
        "sequence": 3,
        "request_id": "request-a",
        "event_type": "chunk_hit",
        "chunk_index": 1,
        "key": "key-1",
        "parent_key": "key-0",
        "result": "hit",
        "duration_ns": 42,
        "metadata": {"backend": "memory"},
    }


def test_recorder_assigns_ordered_sequences_and_preserves_request_ids():
    recorder = TraceRecorder()

    first = recorder.record("lookup_started", request_id="request-a")
    second = recorder.record("lookup_completed", request_id="request-b")

    assert first is not None and first.sequence == 0
    assert second is not None and second.sequence == 1
    assert [event.request_id for event in recorder.events] == [
        "request-a",
        "request-b",
    ]


def test_recorder_can_bind_a_default_request_id():
    recorder = TraceRecorder(request_id="request-a")

    event = recorder.record("chunk_hit", chunk_index=0)

    assert event is not None
    assert event.request_id == "request-a"


def test_recorder_exports_json_and_writes_a_file(tmp_path):
    recorder = TraceRecorder(request_id="request-a")
    recorder.record("lookup_started")
    recorder.record("lookup_completed", metadata={"matched_chunks": 2})

    payload = json.loads(recorder.to_json())
    assert payload[0]["event_type"] == "lookup_started"
    assert payload[1]["metadata"] == {"matched_chunks": 2}

    output = tmp_path / "trace.json"
    recorder.write_json(output)
    assert json.loads(output.read_text()) == payload


def test_disabled_recorder_does_not_record_or_require_request_id():
    recorder = TraceRecorder(enabled=False)

    assert recorder.record("lookup_started") is None
    assert recorder.events == ()

    recorder.enabled = True
    with pytest.raises(ValueError, match="request_id"):
        recorder.record("lookup_started")


def test_disable_keeps_existing_events_and_clear_restarts_sequence():
    recorder = TraceRecorder(request_id="request-a")
    recorder.record("lookup_started")
    recorder.disable()
    recorder.record("lookup_completed")

    assert len(recorder) == 1
    recorder.clear()
    recorder.enable()
    event = recorder.record("lookup_started")
    assert event is not None and event.sequence == 0


def test_event_metadata_is_copied_and_invalid_values_are_rejected():
    metadata = {"backend": "memory"}
    event = TraceEvent(0, "request-a", "lookup_started", metadata=metadata)
    metadata["backend"] = "disk"
    assert event.metadata == {"backend": "memory"}

    with pytest.raises(ValueError):
        TraceEvent(-1, "request-a", "lookup_started")
    with pytest.raises(ValueError):
        TraceEvent(0, "request-a", "lookup_started", duration_ns=-1)
    with pytest.raises(ValueError):
        TraceEvent(0, "request-a", "lookup_started", chunk_index=-1)
