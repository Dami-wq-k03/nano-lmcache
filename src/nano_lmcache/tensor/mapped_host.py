"""Experimental CUDA kernels writing directly to mapped pinned host memory.

These use GPU stores over PCIe, not the DMA copy engine. A single launch can
remove an intermediate GPU buffer, but can be slower than pack + cudaMemcpy.
Unsupported host mapping raises a CUDA error rather than using a hidden fallback.
"""

import torch

from nano_lmcache.tensor.kernels import _cuda_extension, _validate
from nano_lmcache.tensor.quantization import QuantizedKV
from nano_lmcache.tensor.transfer import TransferHandle


def gather_to_host_gpu(
    pages: torch.Tensor,
    ids: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    validate: bool = True,
) -> TransferHandle:
    """Gather non-contiguous GPU pages directly into a pinned CPU output."""
    _validate(pages, ids, None, scatter=False, validate=validate)
    shape = (ids.numel(), *pages.shape[1:])
    if out is None:
        out = torch.empty(shape, dtype=pages.dtype, device="cpu", pin_memory=True)
    if (
        out.shape != shape
        or out.dtype != pages.dtype
        or (out.numel() > 0 and not out.is_pinned())
        or out.device.type != "cpu"
        or not out.is_contiguous()
    ):
        raise ValueError("out must be matching contiguous pinned CPU memory")
    with torch.cuda.device(pages.device):
        if out.numel():
            _cuda_extension().pack_host(pages, ids, out)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(pages.device))
    return TransferHandle(out, event, (pages, ids, out))


def quantize_to_host_gpu(
    value: torch.Tensor,
    *,
    out: QuantizedKV | None = None,
) -> tuple[QuantizedKV, TransferHandle]:
    """Fuse per-channel INT8 quantization and mapped-host export in one launch."""
    if (
        not value.is_cuda
        or not value.is_contiguous()
        or value.ndim < 2
        or not value.numel()
        or not 0 < value.shape[-2] <= 256
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("requires contiguous nonempty CUDA float KV with <=256 tokens")
    scales_shape = (*value.shape[:-2], 1, value.shape[-1])
    if out is None:
        out = QuantizedKV(
            torch.empty_like(value, device="cpu", dtype=torch.int8, pin_memory=True),
            torch.empty(
                scales_shape, device="cpu", dtype=torch.float32, pin_memory=True
            ),
            value.dtype,
            value.ndim - 2,
        )
    if (
        out.values.shape != value.shape
        or out.scales.shape != scales_shape
        or out.values.dtype != torch.int8
        or out.scales.dtype != torch.float32
        or not out.values.is_pinned()
        or not out.scales.is_pinned()
        or not out.values.is_contiguous()
        or not out.scales.is_contiguous()
        or out.dtype != value.dtype
        or out.token_axis != value.ndim - 2
    ):
        raise ValueError("out must be matching pinned INT8 values and FP32 scales")
    with torch.cuda.device(value.device):
        _cuda_extension().quantize_host(value, out.values, out.scales)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(value.device))
    return out, TransferHandle(out.values, event, (value, out.values, out.scales))
