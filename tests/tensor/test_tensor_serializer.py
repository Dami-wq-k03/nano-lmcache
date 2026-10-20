import io
import pickle

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.storage.serializers import Serializer  # noqa: E402
from nano_lmcache.storage.tiered import TieredStorage  # noqa: E402
from nano_lmcache.tensor.hf import run_with_prefix_cache  # noqa: E402
from nano_lmcache.tensor.serializers import TensorSerializer  # noqa: E402

pytestmark = pytest.mark.torch


def test_satisfies_serializer_protocol():
    assert isinstance(TensorSerializer(), Serializer)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_round_trip_is_exact(dtype):
    original = torch.randn(3, 2, 2, 4, 8).to(dtype)
    serializer = TensorSerializer()

    restored = serializer.loads(serializer.dumps(original))

    assert restored.dtype == dtype
    assert torch.equal(restored, original)


def test_loads_to_cpu_even_from_a_gpu_tensor():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    serializer = TensorSerializer()

    restored = serializer.loads(serializer.dumps(torch.ones(2, device="cuda")))

    assert restored.device.type == "cpu"


def test_only_stores_the_tensor_view_not_its_whole_storage():
    big = torch.zeros(1000, 1000)
    serializer = TensorSerializer()

    data = serializer.dumps(big[:2])

    assert len(data) < 100_000


def test_rejects_non_tensor_values():
    with pytest.raises(TypeError):
        TensorSerializer().dumps("not a tensor")


class Exploit:
    def __reduce__(self):
        return (print, ("this must never run",))


def test_refuses_to_unpickle_arbitrary_objects():
    buffer = io.BytesIO()
    torch.save({"payload": Exploit()}, buffer)

    with pytest.raises(pickle.UnpicklingError):
        TensorSerializer().loads(buffer.getvalue())


def test_disk_cache_stores_tensors(tmp_path):
    cache = DiskKVCache(tmp_path, serializer=TensorSerializer())
    original = torch.randn(3, 2, 2, 4, 8)

    cache.put("a", original)

    reopened = DiskKVCache(tmp_path, serializer=TensorSerializer())
    assert torch.equal(reopened.get("a"), original)


# ---- the full hierarchy with a real (tiny) model ----------------------


@pytest.fixture(scope="module")
def model():
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


def test_model_reuses_kv_from_disk_after_a_restart(model, tmp_path):
    from nano_lmcache.storage.memory import MemoryKVCache

    def open_cache():
        storage = TieredStorage(
            fast=MemoryKVCache(max_entries=1),
            slow=DiskKVCache(tmp_path, serializer=TensorSerializer()),
        )
        return ChunkedPrefixCache(chunk_size=4, namespace="tiny", storage=storage)

    shared = list(range(1, 13))
    run_with_prefix_cache(model, open_cache(), shared + [50, 51])

    # New process: memory is empty, KV comes back from disk.
    cache = open_cache()
    prompt = shared + [90, 91, 92]
    result = run_with_prefix_cache(model, cache, prompt)

    with torch.no_grad():
        expected = model(torch.tensor([prompt])).logits
    assert result.matched_tokens == 12
    assert cache.storage.tier_stats.slow_hits == 3
    assert torch.allclose(result.logits, expected[:, 12:], atol=1e-4)
