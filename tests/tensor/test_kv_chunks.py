import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.tensor.kv_chunks import join_kv, split_kv  # noqa: E402

pytestmark = pytest.mark.torch

NUM_LAYERS = 3
NUM_KV_HEADS = 2
HEAD_DIM = 4


def make_layers(num_tokens: int):
    """Random per-layer (K, V) pairs shaped [kv_heads, tokens, head_dim]."""
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(NUM_KV_HEADS, num_tokens, HEAD_DIM, generator=generator),
            torch.randn(NUM_KV_HEADS, num_tokens, HEAD_DIM, generator=generator),
        )
        for _ in range(NUM_LAYERS)
    ]


def test_split_produces_one_tensor_per_complete_chunk():
    chunks = split_kv(make_layers(8), chunk_size=4)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.shape == (NUM_LAYERS, 2, NUM_KV_HEADS, 4, HEAD_DIM)


def test_split_ignores_the_incomplete_tail():
    chunks = split_kv(make_layers(5), chunk_size=2)

    assert len(chunks) == 2


def test_split_of_fewer_tokens_than_one_chunk_is_empty():
    assert split_kv(make_layers(3), chunk_size=4) == []


def test_split_then_join_restores_the_complete_part_exactly():
    layers = make_layers(6)

    joined = join_kv(split_kv(layers, chunk_size=2))

    assert len(joined) == NUM_LAYERS
    for (key, value), (orig_key, orig_value) in zip(joined, layers, strict=True):
        assert torch.equal(key, orig_key)
        assert torch.equal(value, orig_value)


def test_chunks_do_not_share_memory_with_the_source():
    layers = make_layers(4)
    chunks = split_kv(layers, chunk_size=2)

    layers[0][0].zero_()

    assert chunks[0][0, 0].abs().sum() > 0


def test_join_of_a_prefix_of_chunks_gives_a_prefix_of_tokens():
    layers = make_layers(6)
    chunks = split_kv(layers, chunk_size=2)

    joined = join_kv(chunks[:2])

    assert joined[0][0].shape == (NUM_KV_HEADS, 4, HEAD_DIM)
    assert torch.equal(joined[0][0], layers[0][0][:, :4])


def test_join_requires_at_least_one_chunk():
    with pytest.raises(ValueError):
        join_kv([])


def test_split_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        split_kv(make_layers(4), chunk_size=0)


def test_split_rejects_layers_with_different_shapes():
    layers = make_layers(4)
    layers[1] = (layers[1][0][:, :3], layers[1][1][:, :3])

    with pytest.raises(ValueError):
        split_kv(layers, chunk_size=2)
