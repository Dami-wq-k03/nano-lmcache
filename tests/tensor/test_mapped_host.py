import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.tensor.mapped_host import (  # noqa: E402
    gather_to_host_gpu,
    quantize_to_host_gpu,
)
from nano_lmcache.tensor.quantization import dequantize_kv, quantize_kv  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_mapped_pack_and_quantize(dtype):
    pages = torch.randn(17, 2, 3, 16, 32, device="cuda", dtype=dtype)
    ids = torch.tensor([13, 2, 16, 0, 13], device="cuda")
    handle = gather_to_host_gpu(pages, ids)
    assert torch.equal(handle.wait(), pages[ids].cpu())
    encoded, handle = quantize_to_host_gpu(pages)
    handle.wait()
    expected = quantize_kv(pages.cpu())
    torch.testing.assert_close(encoded.scales, expected.scales)
    assert (encoded.values.int() - expected.values.int()).abs().max() <= 1
    restored = dequantize_kv(encoded)
    error = (restored.float() - pages.cpu().float()).norm() / pages.cpu().float().norm()
    assert error < 0.02


def test_mapped_pack_empty_and_validation():
    pages = torch.randn(2, 16, device="cuda")
    empty = torch.empty(0, dtype=torch.long, device="cuda")
    assert gather_to_host_gpu(pages, empty).wait().shape == (0, 16)
    with pytest.raises(ValueError, match="pinned"):
        gather_to_host_gpu(
            pages, torch.tensor([0], device="cuda"), out=torch.empty(1, 16)
        )
