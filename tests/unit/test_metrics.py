from nano_lmcache.core.metrics import CacheMetrics


def test_new_metrics_start_at_zero():
    metrics = CacheMetrics()

    assert metrics.hit_chunks == 0
    assert metrics.miss_chunks == 0
    assert metrics.hit_tokens == 0
    assert metrics.total_chunks == 0


def test_hit_rate_is_zero_without_any_lookup():
    # Must not divide by zero before any lookup.
    assert CacheMetrics().hit_rate == 0.0


def test_record_lookup_accumulates_counters():
    metrics = CacheMetrics()

    metrics.record_lookup(hit_chunks=2, miss_chunks=1, hit_tokens=4)
    metrics.record_lookup(hit_chunks=3, miss_chunks=0, hit_tokens=6)

    assert metrics.hit_chunks == 5
    assert metrics.miss_chunks == 1
    assert metrics.hit_tokens == 10
    assert metrics.total_chunks == 6


def test_hit_rate_is_hit_chunks_over_total_chunks():
    metrics = CacheMetrics()

    metrics.record_lookup(hit_chunks=2, miss_chunks=1, hit_tokens=4)

    assert metrics.hit_rate == 2 / 3


def test_hit_rate_is_one_when_everything_hits():
    metrics = CacheMetrics()

    metrics.record_lookup(hit_chunks=2, miss_chunks=0, hit_tokens=4)

    assert metrics.hit_rate == 1.0
