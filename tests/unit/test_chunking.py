import pytest

from nano_lmcache.core.chunking import (
    chunk_tokens,
    make_chunk_key,
)


def test_chunk_tokens():
    assert chunk_tokens(
        [10, 20, 30, 40, 50],
        chunk_size=2,
    ) == [
        (10, 20),
        (30, 40),
        (50,),
    ]


def test_empty_tokens():
    assert chunk_tokens([], chunk_size=2) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_tokens([10, 20], chunk_size=0)


def test_chunk_key_is_stable():
    assert make_chunk_key(
        [10, 20],
        namespace="model-a",
    ) == make_chunk_key(
        [10, 20],
        namespace="model-a",
    )


def test_chunk_key_depends_on_namespace():
    assert make_chunk_key(
        [10, 20],
        namespace="model-a",
    ) != make_chunk_key(
        [10, 20],
        namespace="model-b",
    )


def test_chunk_key_depends_on_prefix():
    prefix_a = make_chunk_key(
        [101, 205],
        namespace="model-a",
    )
    prefix_b = make_chunk_key(
        [999, 205],
        namespace="model-a",
    )
    suffix_a = make_chunk_key(
        [307, 409],
        namespace="model-a",
        prefix_key=prefix_a,
    )
    suffix_b = make_chunk_key(
        [307, 409],
        namespace="model-a",
        prefix_key=prefix_b,
    )
    assert suffix_a != suffix_b


def test_chunk_key_is_stable_with_same_prefix():
    prefix_a = make_chunk_key(
        [101, 205],
        namespace="model-a",
    )
    prefix_b = make_chunk_key(
        [101, 205],
        namespace="model-a",
    )
    suffix_a = make_chunk_key(
        [307, 409],
        namespace="model-a",
        prefix_key=prefix_a,
    )
    suffix_b = make_chunk_key(
        [307, 409],
        namespace="model-a",
        prefix_key=prefix_b,
    )
    assert suffix_a == suffix_b
