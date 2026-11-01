import pytest

from nano_lmcache.core.metrics import CacheMetrics
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.observability import TraceRecorder, TracingStorage
from nano_lmcache.storage.memory import MemoryKVCache


def test_store_then_lookup_returns_all_chunks():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    tokens = [101, 205, 307, 409]
    cache.store(tokens, ["kv-0", "kv-1"])

    result = cache.lookup(tokens)

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4
    assert result.values == ("kv-0", "kv-1")


def test_lookup_returns_longest_matching_prefix():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    cache.store(
        [101, 205, 307, 409, 511, 613],
        ["kv-0", "kv-1", "kv-2"],
    )

    result = cache.lookup(
        [101, 205, 307, 409, 777, 888]
    )

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4
    assert result.values == ("kv-0", "kv-1")


def test_lookup_stops_when_first_chunk_misses():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    cache.store(
        [101, 205, 307, 409],
        ["kv-0", "kv-1"],
    )

    result = cache.lookup([999, 205, 307, 409])

    assert result.matched_chunks == 0
    assert result.matched_tokens == 0
    assert result.values == ()


def test_incomplete_trailing_chunk_is_not_reused():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    cache.store(
        [101, 205, 307, 409, 511],
        ["kv-0", "kv-1"],
    )

    result = cache.lookup([101, 205, 307, 409, 511])

    assert result.matched_chunks == 2
    assert result.matched_tokens == 4


def test_store_rejects_wrong_number_of_values():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )

    with pytest.raises(ValueError):
        cache.store([101, 205, 307, 409], ["kv-0"])


def test_lookup_records_a_full_hit():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    tokens = [101, 205, 307, 409]
    cache.store(tokens, ["kv-0", "kv-1"])

    cache.lookup(tokens)

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 0
    assert cache.metrics.hit_tokens == 4
    assert cache.metrics.hit_rate == 1.0


def test_lookup_records_remaining_chunks_as_misses():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    cache.store(
        [101, 205, 307, 409, 511, 613],
        ["kv-0", "kv-1", "kv-2"],
    )

    # The first 2 chunks hit; the 3rd must be recomputed, so it is 1 miss.
    cache.lookup([101, 205, 307, 409, 777, 888])

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 1
    assert cache.metrics.hit_tokens == 4
    assert cache.metrics.hit_rate == 2 / 3


def test_lookup_on_cold_cache_records_only_misses():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )

    cache.lookup([101, 205, 307, 409])

    assert cache.metrics.hit_chunks == 0
    assert cache.metrics.miss_chunks == 2
    assert cache.metrics.hit_tokens == 0
    assert cache.metrics.hit_rate == 0.0


def test_metrics_ignore_incomplete_trailing_chunk():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    cache.store(
        [101, 205, 307, 409, 511],
        ["kv-0", "kv-1"],
    )

    # The incomplete tail chunk (5th token) is not reusable and not a miss.
    cache.lookup([101, 205, 307, 409, 511])

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 0


def test_metrics_accumulate_across_lookups():
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
    )
    tokens = [101, 205, 307, 409]

    cache.lookup(tokens)
    cache.store(tokens, ["kv-0", "kv-1"])
    cache.lookup(tokens)

    assert cache.metrics.hit_chunks == 2
    assert cache.metrics.miss_chunks == 2
    assert cache.metrics.hit_tokens == 4
    assert cache.metrics.hit_rate == 0.5


def test_injected_metrics_object_is_shared():
    metrics = CacheMetrics()
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        metrics=metrics,
    )
    tokens = [101, 205]
    cache.store(tokens, ["kv-0"])

    cache.lookup(tokens)

    assert metrics.hit_chunks == 1
    assert cache.metrics is metrics



def test_chunk_size_is_exposed():
    cache = ChunkedPrefixCache(chunk_size=4, namespace="model-a")

    assert cache.chunk_size == 4


def test_lookup_trace_distinguishes_hit_miss_and_not_checked():
    recorder = TraceRecorder()
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        trace_recorder=recorder,
    )
    cache.store([101, 205, 307, 409], ["kv-0", "kv-1"])

    result = cache.lookup(
        [101, 205, 307, 409, 777, 888, 999, 1001],
        request_id="request-b",
    )

    assert result.matched_chunks == 2
    assert [event.event_type for event in recorder.events] == [
        "lookup_started",
        "chunk_hit",
        "chunk_hit",
        "chunk_miss",
        "lookup_stopped",
        "chunk_not_checked",
        "lookup_completed",
    ]
    assert [event.chunk_index for event in recorder.events[1:6]] == [
        0,
        1,
        2,
        2,
        3,
    ]
    completed = recorder.events[-1]
    assert completed.metadata == {
        "matched_chunks": 2,
        "matched_tokens": 4,
        "miss_chunks": 2,
        "checked_chunks": 3,
        "unchecked_chunks": 1,
        "incomplete_chunks": 0,
    }
    assert all(event.request_id == "request-b" for event in recorder.events)


def test_lookup_trace_marks_incomplete_trailing_chunk():
    recorder = TraceRecorder(request_id="request-a")
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        trace_recorder=recorder,
    )
    cache.store([101, 205, 307, 409], ["kv-0", "kv-1"])

    cache.lookup([101, 205, 307, 409, 511])

    incomplete = recorder.events[-2]
    assert incomplete.event_type == "chunk_incomplete"
    assert incomplete.chunk_index == 2
    assert incomplete.key is None
    assert incomplete.metadata == {"token_count": 1, "chunk_size": 2}
    assert recorder.events[-1].metadata["incomplete_chunks"] == 1


def test_lookup_without_request_id_gets_anonymous_trace_id():
    recorder = TraceRecorder()
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        trace_recorder=recorder,
    )

    cache.lookup([101, 205])

    assert recorder.events[0].request_id == "lookup-0"


def test_storage_events_inherit_the_lookup_request_context():
    recorder = TraceRecorder()
    storage = TracingStorage(MemoryKVCache(), recorder, name="memory")
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="model-a",
        storage=storage,
        trace_recorder=recorder,
    )

    cache.lookup([101, 205], request_id="request-a")
    cache.store([101, 205], ["kv-0"], request_id="request-a")

    storage_events = [
        event for event in recorder.events if event.event_type.startswith("storage_")
    ]
    assert storage_events
    assert {event.request_id for event in storage_events} == {"request-a"}
    assert storage_events[-1].parent_key is None
