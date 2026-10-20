import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.storage.protocol import CacheStorage  # noqa: E402
from nano_lmcache.tensor.hf import run_with_prefix_cache  # noqa: E402
from nano_lmcache.tensor.offload import CPUOffloadKVCache  # noqa: E402

pytestmark = pytest.mark.gpu


def gpu_tensor(seed: int = 0, shape=(3, 2, 2, 4, 8)):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=generator)


@pytest.fixture
def offload():
    return CPUOffloadKVCache(device="cuda")


def test_satisfies_storage_protocol(offload):
    assert isinstance(offload, CacheStorage)


def test_get_returns_an_exact_copy_on_the_gpu(offload):
    original = gpu_tensor()

    offload.put("a", original)
    restored = offload.get("a")

    assert restored.is_cuda
    assert torch.equal(restored, original)


def test_payload_is_held_in_pinned_cpu_memory(offload):
    offload.put("a", gpu_tensor())

    held = offload.peek("a")

    assert held.device.type == "cpu"
    assert held.is_pinned()


def test_offloading_frees_gpu_memory(offload):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    big = gpu_tensor(shape=(8, 2, 8, 256, 64))  # 16 MiB

    offload.put("a", big)
    del big
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == baseline
    assert offload.nbytes == 8 * 2 * 8 * 256 * 64 * 4


def test_returned_tensor_is_independent_of_the_cache(offload):
    offload.put("a", gpu_tensor())

    offload.get("a").zero_()

    assert offload.get("a").abs().sum() > 0


def test_source_can_change_after_put(offload):
    original = gpu_tensor()
    expected = original.clone()
    offload.put("a", original)

    original.zero_()

    assert torch.equal(offload.get("a"), expected)


def test_missing_key_is_none(offload):
    assert offload.get("missing") is None


def test_non_tensor_payload_is_rejected_without_evicting():
    offload = CPUOffloadKVCache(device="cuda", max_entries=1)
    offload.put("a", gpu_tensor())

    with pytest.raises(TypeError):
        offload.put("b", "not a tensor")

    assert "a" in offload
    assert offload.stats.evictions == 0


def test_capacity_uses_prefix_safe_eviction():
    offload = CPUOffloadKVCache(device="cuda", max_entries=2)
    offload.put("a", gpu_tensor(1))
    offload.put("b", gpu_tensor(2), parent_key="a")

    # The only leaf is b, the new entry's parent, so c is rejected.
    offload.put("c", gpu_tensor(3), parent_key="b")
    # An unrelated root evicts the leaf b, never its parent a.
    offload.put("x", gpu_tensor(4))

    assert "c" not in offload
    assert "b" not in offload
    assert "a" in offload
    assert offload.stats.rejected_no_room == 1
    assert offload.stats.evictions == 1
    assert offload.nbytes == 2 * gpu_tensor().nelement() * 4


def test_eviction_releases_cpu_memory_accounting():
    offload = CPUOffloadKVCache(device="cuda", max_entries=1)
    offload.put("a", gpu_tensor(1))

    offload.put("b", gpu_tensor(2))

    assert len(offload) == 1
    assert offload.nbytes == gpu_tensor().nelement() * 4


@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("non_blocking", [True, False])
def test_copy_modes_are_all_exact(pin_memory, non_blocking):
    offload = CPUOffloadKVCache(
        device="cuda", pin_memory=pin_memory, non_blocking=non_blocking
    )
    original = gpu_tensor()

    offload.put("a", original)

    assert offload.peek("a").is_pinned() == pin_memory
    assert torch.equal(offload.get("a"), original)


def test_parent_of_and_contains(offload):
    offload.put("a", gpu_tensor())
    offload.put("b", gpu_tensor(), parent_key="a")

    assert "b" in offload
    assert offload.parent_of("b") == "a"
    with pytest.raises(KeyError):
        offload.parent_of("missing")


# ---- with a real (tiny) model -----------------------------------------


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


def test_model_reuses_kv_offloaded_to_cpu(cuda_model):
    storage = CPUOffloadKVCache(device="cuda")
    cache = ChunkedPrefixCache(chunk_size=4, namespace="tiny", storage=storage)
    shared = list(range(1, 9))
    run_with_prefix_cache(cuda_model, cache, shared + [50, 51, 52])

    result = run_with_prefix_cache(cuda_model, cache, shared + [90, 91, 92, 93])

    with torch.no_grad():
        expected = cuda_model(
            torch.tensor([shared + [90, 91, 92, 93]], device="cuda")
        ).logits
    assert result.matched_tokens == 8
    assert torch.allclose(result.logits, expected[:, 8:], atol=1e-3)
    # Every cached chunk lives in CPU memory, not on the GPU.
    assert len(storage) > 0
    assert all(storage.peek(k).device.type == "cpu" for k in storage.keys())


class CPUOnlyStorage:
    """Hands back chunks on the CPU, like a disk tier would."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def put(self, key, value, *, parent_key=None) -> None:
        self.data[key] = value.cpu()

    def get(self, key):
        return self.data.get(key)


def test_model_accepts_cached_chunks_from_another_device(cuda_model):
    cache = ChunkedPrefixCache(
        chunk_size=4, namespace="tiny-cpu", storage=CPUOnlyStorage()
    )
    shared = list(range(1, 9))
    run_with_prefix_cache(cuda_model, cache, shared + [50])

    result = run_with_prefix_cache(cuda_model, cache, shared + [60, 61])

    assert result.matched_tokens == 8
    assert result.logits.is_cuda
