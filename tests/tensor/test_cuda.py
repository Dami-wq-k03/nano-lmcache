import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.tensor.hf import run_with_prefix_cache  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda_model():
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
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    return model.to("cuda")


def test_cuda_is_usable():
    assert torch.cuda.is_available()
    assert torch.zeros(1, device="cuda").is_cuda


def test_reused_prefix_matches_full_computation_on_gpu(cuda_model):
    cache = ChunkedPrefixCache(chunk_size=4, namespace="tiny-qwen3-cuda")
    shared = list(range(1, 9))
    prompt_a = shared + [50, 51, 52, 53, 54]
    prompt_b = shared + [90, 91, 92, 93, 94, 95]
    run_with_prefix_cache(cuda_model, cache, prompt_a)

    result = run_with_prefix_cache(cuda_model, cache, prompt_b)

    with torch.no_grad():
        expected = cuda_model(torch.tensor([prompt_b], device="cuda")).logits
    assert result.matched_tokens == 8
    assert torch.allclose(result.logits, expected[:, 8:], atol=1e-3)


def test_cached_chunks_stay_on_the_gpu(cuda_model):
    cache = ChunkedPrefixCache(chunk_size=4, namespace="tiny-qwen3-cuda-2")
    prompt = list(range(1, 10))
    run_with_prefix_cache(cuda_model, cache, prompt)

    chunk = cache.lookup(prompt).values[0]

    assert chunk.is_cuda
