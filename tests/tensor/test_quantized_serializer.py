import io
import pickle

import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.storage.serializers import Serializer  # noqa: E402
from nano_lmcache.tensor.serializers import (  # noqa: E402
    QuantizedTensorSerializer,
    TensorSerializer,
)

pytestmark = pytest.mark.torch
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_round_trip(dtype, device):
    torch.manual_seed(0)
    original = torch.randn(3, 2, 2, 16, 128, device=device).to(dtype)
    original = original.transpose(-1, -2).requires_grad_()
    before = original.detach().clone()
    serializer = QuantizedTensorSerializer()
    assert isinstance(serializer, Serializer)

    restored = serializer.loads(serializer.dumps(original))

    assert restored.shape == original.shape
    assert restored.dtype == dtype
    assert restored.device.type == "cpu"
    assert not restored.requires_grad
    assert torch.equal(original, before)
    reference = before.cpu().float()
    error = (restored.float() - reference).abs()
    bound = reference.abs().amax(dim=-2, keepdim=True) / 254
    bound = bound + torch.finfo(dtype).eps * reference.abs() + 1e-6
    assert torch.all(error <= bound)
    assert error.norm() / reference.norm() < 0.015


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_vectors(dtype):
    original = torch.zeros(2, 2, 2, 4, 8, dtype=dtype)
    serializer = QuantizedTensorSerializer()
    for value in (0, 1):
        original[0, 0, 0, 0] = value
        assert torch.equal(serializer.loads(serializer.dumps(original)), original)


def test_payload_is_compact_and_preserves_shape():
    original = torch.ones(2, 2, 4, 32, 128, dtype=torch.float16)
    data = QuantizedTensorSerializer().dumps(original)
    payload = torch.load(io.BytesIO(data), weights_only=True)
    assert payload["values"].dtype == torch.int8
    assert payload["values"].shape == original.shape
    assert payload["scales"].dtype == torch.float32
    assert payload["scales"].shape == (*original.shape[:-2], 1, original.shape[-1])
    assert len(data) < original.nbytes * 0.6


@pytest.mark.parametrize("value", ["not a tensor", torch.ones(2, dtype=torch.int64)])
def test_rejects_unsupported_values(value):
    with pytest.raises(TypeError):
        QuantizedTensorSerializer().dumps(value)


def test_restricted_loading():
    buffer = io.BytesIO()
    torch.save({"payload": print}, buffer)
    with pytest.raises(pickle.UnpicklingError):
        QuantizedTensorSerializer().loads(buffer.getvalue())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_g1_disk_reuse_matches_full_prefill(tmp_path, dtype, device):
    transformers = pytest.importorskip("transformers")
    from nano_lmcache.tensor.hf import run_with_prefix_cache

    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(
        transformers.Qwen3Config(
            vocab_size=200, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
        )
    ).to(device=device, dtype=dtype).eval()
    shared = list(range(1, 13))
    prompt = shared + [90, 91, 92]
    with torch.no_grad():
        expected = model(torch.tensor([prompt], device=device)).logits[:, 12:].float()

    for serializer in (TensorSerializer(), QuantizedTensorSerializer()):
        directory = tmp_path / type(serializer).__name__
        cache = ChunkedPrefixCache(
            chunk_size=4, namespace="tiny",
            storage=DiskKVCache(directory, serializer=serializer),
        )
        run_with_prefix_cache(model, cache, shared + [50, 51])
        cache = ChunkedPrefixCache(
            chunk_size=4, namespace="tiny",
            storage=DiskKVCache(directory, serializer=serializer),
        )
        result = run_with_prefix_cache(model, cache, prompt)
        assert result.matched_tokens == 12
        assert result.computed_tokens == 3
        error = result.logits.float() - expected
        assert error.abs().max() < 0.01
        assert error.norm() / expected.norm() < 0.02
        assert torch.equal(result.logits.argmax(-1), expected.argmax(-1))
