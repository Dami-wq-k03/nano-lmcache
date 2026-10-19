import pytest

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
