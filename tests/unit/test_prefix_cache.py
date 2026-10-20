import pytest

from nano_lmcache.core.metrics import CacheMetrics
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache


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
