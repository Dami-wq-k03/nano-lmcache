import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.tensor.hf import (  # noqa: E402
    cache_to_layers,
    layers_to_cache,
    run_with_prefix_cache,
)
from nano_lmcache.tensor.kv_chunks import split_kv  # noqa: E402

pytestmark = pytest.mark.torch

CHUNK_SIZE = 4


@pytest.fixture(scope="module")
def model():
    """A tiny randomly initialised Qwen3: no download, runs in milliseconds.

    KV reuse correctness depends on the architecture (attention, rotary
    positions), not on trained weights, so random weights are enough.
    """
    config = transformers.Qwen3Config(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config).eval()


@pytest.fixture
def cache():
    return ChunkedPrefixCache(chunk_size=CHUNK_SIZE, namespace="tiny-qwen3")


def full_forward(model, token_ids):
    """Reference: compute the whole prompt from scratch, no cache at all."""
    with torch.no_grad():
        output = model(torch.tensor([token_ids]), use_cache=True)
    return output.logits, output.past_key_values


def test_cache_layers_round_trip(model):
    _, past = full_forward(model, list(range(1, 9)))

    layers = cache_to_layers(past)
    rebuilt = layers_to_cache(layers, model.config)

    assert rebuilt.get_seq_length() == 8
    for (key, value), (new_key, new_value) in zip(
        layers, cache_to_layers(rebuilt), strict=True
    ):
        assert torch.equal(key, new_key)
        assert torch.equal(value, new_value)


def test_cache_layers_drop_the_batch_dimension(model):
    _, past = full_forward(model, list(range(1, 9)))

    key, _ = cache_to_layers(past)[0]

    # [kv_heads, tokens, head_dim], without the batch dimension.
    assert key.shape == (2, 8, 8)


def test_cold_run_matches_a_plain_forward_pass(model, cache):
    prompt = list(range(1, 14))

    result = run_with_prefix_cache(model, cache, prompt)

    expected, _ = full_forward(model, prompt)
    assert result.matched_tokens == 0
    assert result.computed_tokens == len(prompt)
    assert torch.allclose(result.logits, expected, atol=1e-5)


def test_reused_prefix_gives_the_same_logits_as_full_computation(model, cache):
    shared = list(range(1, 9))  # two full chunks
    prompt_a = shared + [50, 51, 52, 53, 54]
    prompt_b = shared + [90, 91, 92, 93, 94, 95]
    run_with_prefix_cache(model, cache, prompt_a)

    result = run_with_prefix_cache(model, cache, prompt_b)

    expected, _ = full_forward(model, prompt_b)
    assert result.matched_tokens == 8
    assert result.computed_tokens == len(prompt_b) - 8
    assert torch.allclose(result.logits, expected[:, 8:], atol=1e-4)


def test_reuse_saves_computation(model, cache):
    shared = list(range(1, 13))
    run_with_prefix_cache(model, cache, shared + [100, 101])

    result = run_with_prefix_cache(model, cache, shared + [110, 111, 112])

    assert result.computed_tokens < len(shared) + 3
    assert result.matched_tokens == 12


def test_identical_prompt_still_recomputes_the_last_token(model, cache):
    prompt = list(range(1, 9))
    run_with_prefix_cache(model, cache, prompt)

    result = run_with_prefix_cache(model, cache, prompt)

    # Logits for the next token need one forward step, so one token is
    # always recomputed even when the whole prompt is cached.
    expected, _ = full_forward(model, prompt)
    assert result.computed_tokens == 1
    assert result.matched_tokens == 7
    assert torch.allclose(result.logits, expected[:, -1:], atol=1e-4)


def test_different_first_chunk_is_never_reused(model, cache):
    tail = [60, 61, 62, 63]
    prompt_x = [1, 2, 3, 4] + tail
    prompt_y = [9, 8, 7, 6] + tail  # same second chunk, different prefix
    run_with_prefix_cache(model, cache, prompt_x)

    result = run_with_prefix_cache(model, cache, prompt_y)

    expected, _ = full_forward(model, prompt_y)
    assert result.matched_tokens == 0
    assert torch.allclose(result.logits, expected, atol=1e-5)


def test_same_tokens_after_a_different_prefix_have_different_kv(model):
    """Why chunk keys must include the prefix: the KV itself differs."""
    tail = [60, 61, 62, 63]
    _, past_x = full_forward(model, [1, 2, 3, 4] + tail)
    _, past_y = full_forward(model, [9, 8, 7, 6] + tail)

    chunk_x = split_kv(cache_to_layers(past_x), CHUNK_SIZE)[1]
    chunk_y = split_kv(cache_to_layers(past_y), CHUNK_SIZE)[1]

    assert not torch.allclose(chunk_x, chunk_y, atol=1e-3)


def test_stored_chunks_have_the_documented_layout(model, cache):
    run_with_prefix_cache(model, cache, list(range(1, 10)))

    result = cache.lookup(list(range(1, 10)))

    assert result.matched_chunks == 2
    chunk = result.values[0]
    # [layers, K/V, kv_heads, chunk_tokens, head_dim]
    assert chunk.shape == (3, 2, 2, CHUNK_SIZE, 8)
    assert chunk.dtype == torch.float32


def test_prompt_shorter_than_one_chunk_is_computed_and_not_stored(model, cache):
    result = run_with_prefix_cache(model, cache, [1, 2, 3])

    assert result.computed_tokens == 3
    assert cache.lookup([1, 2, 3]).matched_chunks == 0


def test_empty_prompt_is_rejected(model, cache):
    with pytest.raises(ValueError):
        run_with_prefix_cache(model, cache, [])


def test_layers_without_kv_tensors_are_rejected():
    from types import SimpleNamespace

    # Stands in for a linear-attention or otherwise non-KV cache layer.
    fake_cache = SimpleNamespace(layers=[SimpleNamespace(keys=None, values=None)])

    with pytest.raises(ValueError, match="attention"):
        cache_to_layers(fake_cache)
