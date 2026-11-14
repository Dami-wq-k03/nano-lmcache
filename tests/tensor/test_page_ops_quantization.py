import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.experimental.dataplane import KVLayout  # noqa: E402
from nano_lmcache.tensor.page_ops import gather_pages, scatter_pages  # noqa: E402
from nano_lmcache.tensor.quantization import (  # noqa: E402
    dequantize_kv,
    quantize_kv,
)

pytestmark = pytest.mark.torch


def layout():
    return KVLayout(
        model_namespace="test",
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        page_tokens=3,
        dtype="float16",
        key_layout="HND",
        value_layout="HND",
    )


def test_gather_and_scatter_pages_match_reference():
    descriptor = layout()
    pages = torch.arange(
        5 * 2 * 2 * 2 * 3 * 4, dtype=torch.float16
    ).reshape(5, 2, 2, 2, 3, 4)
    indices = torch.tensor([3, 1, 3], dtype=torch.long)

    gathered = gather_pages(pages, indices, layout=descriptor)
    destination = torch.zeros_like(pages)
    restored = scatter_pages(destination, indices, gathered, layout=descriptor)

    assert torch.equal(gathered[0], pages[3])
    assert torch.equal(gathered[1], pages[1])
    assert torch.equal(restored[3], pages[3])
    assert torch.equal(restored[1], pages[1])
    assert torch.equal(destination, torch.zeros_like(destination))


def test_page_ops_reject_wrong_dtype_and_indices():
    descriptor = layout()
    pages = torch.zeros(2, 2, 2, 2, 3, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        gather_pages(pages, torch.tensor([0], dtype=torch.long), layout=descriptor)
    with pytest.raises(ValueError, match="one-dimensional"):
        gather_pages(pages, torch.tensor([[0]], dtype=torch.long))


def test_int8_quantization_is_per_head_and_restores_dtype():
    value = torch.randn(2, 2, 4, 16, 8, dtype=torch.float16)
    encoded = quantize_kv(value)
    restored = dequantize_kv(encoded)

    assert encoded.values.dtype is torch.int8
    assert encoded.scales.shape == (2, 2, 4, 1, 8)
    assert restored.dtype is value.dtype
    assert restored.shape == value.shape
    assert (restored.float() - value.float()).norm() / value.float().norm() < 0.02


def test_int8_quantization_rejects_non_floating_input():
    with pytest.raises(TypeError):
        quantize_kv(torch.ones(4, dtype=torch.int64))
