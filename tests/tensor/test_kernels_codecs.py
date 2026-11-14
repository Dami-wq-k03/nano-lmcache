import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.tensor.codecs import (  # noqa: E402
    dequantize_blocks,
    dequantize_kv_gpu,
    quantize_blocks,
    quantize_kv_asymmetric,
    quantize_kv_gpu,
)
from nano_lmcache.tensor.kernels import (  # noqa: E402
    gather_pages_gpu,
    scatter_pages_gpu_,
)
from nano_lmcache.tensor.quantization import (  # noqa: E402
    dequantize_kv,
    quantize_kv,
)

pytestmark = pytest.mark.torch


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [7, 32])
def test_kernels_match_random_page_table(backend, dtype, width):
    if backend == "triton":
        pytest.importorskip("triton")
    pages = torch.randn(37, 2, 3, width, dtype=dtype, device="cuda")
    ids = torch.tensor([31, 3, 31, 0, 36], device="cuda")
    packed = gather_pages_gpu(pages, ids, backend=backend)
    assert torch.equal(packed, pages.index_select(0, ids))
    targets = torch.randperm(37, device="cuda")[:5]
    dest = torch.zeros_like(pages)
    expected = dest.clone().index_copy_(0, targets, packed)
    scatter_pages_gpu_(dest, targets, packed, backend=backend)
    assert torch.equal(dest, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_kernels_empty_and_invalid_tables(backend):
    pages = torch.zeros(3, 16, device="cuda")
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    assert gather_pages_gpu(pages, empty, backend=backend).shape == (0, 16)
    with pytest.raises(IndexError):
        gather_pages_gpu(pages, torch.tensor([3], device="cuda"), backend=backend)
    with pytest.raises(ValueError, match="unique"):
        scatter_pages_gpu_(
            pages, torch.tensor([1, 1], device="cuda"), pages[:2], backend=backend
        )
    with pytest.raises(ValueError, match="alias"):
        gather_pages_gpu(pages, torch.arange(3, device="cuda"), out=pages)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [3, 16, 64])
def test_triton_int8_matches_reference(dtype, tokens):
    pytest.importorskip("triton")
    torch.manual_seed(42)
    value = torch.randn(2, 2, 3, tokens, 33, device="cuda", dtype=dtype)
    value[0, 0].zero_()
    reference = quantize_kv(value)
    encoded = quantize_kv_gpu(value)
    torch.testing.assert_close(encoded.scales, reference.scales)
    assert (encoded.values.int() - reference.values.int()).abs().max() <= 1
    restored = dequantize_kv_gpu(encoded)
    torch.testing.assert_close(restored, dequantize_kv(encoded), rtol=0, atol=0)
    assert (restored.float() - value.float()).norm() / value.float().norm() < 0.02


@pytest.mark.parametrize("format,limit", [("int4", 0.12), ("fp8", 0.05)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_block_codecs_padding_size_and_quality(format, limit, dtype):
    torch.manual_seed(3)
    value = torch.randn(2, 3, 17, dtype=dtype)
    encoded = quantize_blocks(value, format=format)
    restored = dequantize_blocks(encoded)
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert (restored.float() - value.float()).norm() / value.float().norm() < limit
    assert encoded.nbytes < value.nbytes
    zero = torch.zeros(3, dtype=dtype)
    assert torch.equal(dequantize_blocks(quantize_blocks(zero, format=format)), zero)


def test_asymmetric_codecs_and_invalid_input():
    value = torch.randn(2, 2, 3, 16, 8)
    key, val = quantize_kv_asymmetric(value)
    assert key.values.dtype == torch.int8
    assert val.format == "int4"
    with pytest.raises(ValueError):
        quantize_blocks(torch.tensor([float("nan")]))
    with pytest.raises(TypeError):
        quantize_blocks(torch.ones(2, dtype=torch.int32))
