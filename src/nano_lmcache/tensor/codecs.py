"""GPU INT8 adapter and blockwise INT4/FP8 research codecs.

Low-bit formats are opt-in and lossy. The block codecs flatten a contiguous
logical tensor; their scales and padding are part of the stored payload.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from nano_lmcache.tensor.quantization import QuantizedKV


def quantize_kv_gpu(value: torch.Tensor) -> QuantizedKV:
    """Fused per-channel INT8 reduction and quantization for [..., tokens, dim]."""
    if (
        not value.is_cuda
        or not value.is_contiguous()
        or value.ndim < 2
        or not value.numel()
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not 0 < value.shape[-2] <= 256
        or value.shape[-1] == 0
    ):
        raise ValueError("requires contiguous CUDA float KV with 1--256 tokens")
    from nano_lmcache.tensor._triton_kernels import quantize_rows

    with torch.cuda.device(value.device):
        values, scales = quantize_rows(value)
    return QuantizedKV(values, scales, value.dtype, value.ndim - 2)


def dequantize_kv_gpu(value: QuantizedKV) -> torch.Tensor:
    """Single-launch INT8 dequantization on the values' CUDA device."""
    if (
        not value.values.is_cuda
        or not value.values.is_contiguous()
        or not value.scales.is_contiguous()
        or value.scales.device != value.values.device
        or value.token_axis != value.values.ndim - 2
        or value.values.dtype != torch.int8
        or value.scales.shape != (*value.values.shape[:-2], 1, value.values.shape[-1])
    ):
        raise ValueError("requires contiguous CUDA INT8 values and per-channel scales")
    from nano_lmcache.tensor._triton_kernels import dequantize_rows

    with torch.cuda.device(value.values.device):
        return dequantize_rows(value.values, value.scales, value.dtype)


@dataclass(frozen=True)
class BlockQuantizedKV:
    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    format: Literal["int4", "fp8"]
    block_size: int

    @property
    def nbytes(self) -> int:
        return int(self.codes.nbytes + self.scales.nbytes)


def quantize_blocks(
    value: torch.Tensor,
    *,
    format: Literal["int4", "fp8"] = "int4",
    block_size: int = 32,
) -> BlockQuantizedKV:
    """Reference INT4 nibble packing or scaled E4M3FN FP8, in fixed blocks."""
    if format not in ("int4", "fp8") or block_size <= 0 or block_size % 2:
        raise ValueError("format must be int4/fp8 and block_size a positive even int")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("expected FP16, BF16 or FP32")
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError("expected nonempty finite values")
    count = math.ceil(value.numel() / block_size) * block_size
    padded = torch.zeros(count, dtype=torch.float32, device=value.device)
    padded[: value.numel()] = value.detach().reshape(-1).float()
    blocks = padded.reshape(-1, block_size)
    limit = 7 if format == "int4" else 448
    scales = blocks.abs().amax(dim=1, keepdim=True) / limit
    scales = torch.where(scales == 0, 1, scales)
    normalized = blocks / scales
    if format == "int4":
        codes = (normalized.round().clamp(-7, 7).to(torch.int8) + 8).to(torch.uint8)
        codes = codes[:, 0::2] | (codes[:, 1::2] << 4)
    else:
        codes = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    return BlockQuantizedKV(
        codes, scales, tuple(value.shape), value.dtype, format, block_size
    )


def dequantize_blocks(value: BlockQuantizedKV) -> torch.Tensor:
    """Restore the logical shape and original floating-point dtype."""
    if value.format == "int4":
        low = (value.codes & 15).to(torch.int8) - 8
        high = (value.codes >> 4).to(torch.int8) - 8
        blocks = torch.stack((low, high), dim=-1).flatten(1).float()
    else:
        blocks = value.codes.view(torch.float8_e4m3fn).float()
    flat = (blocks * value.scales).flatten()[: math.prod(value.shape)]
    return flat.reshape(value.shape).to(value.dtype)


def quantize_kv_asymmetric(
    value: torch.Tensor, *, key_format: Literal["int8", "int4"] = "int8"
) -> tuple[QuantizedKV | BlockQuantizedKV, BlockQuantizedKV]:
    """K8V4 or K4V4 reference for canonical [layers, 2, heads, tokens, dim]."""
    from nano_lmcache.tensor.quantization import quantize_kv

    if value.ndim != 5 or value.shape[1] != 2:
        raise ValueError("expected canonical [layers, 2, heads, tokens, dim]")
    if key_format not in ("int8", "int4"):
        raise ValueError("key_format must be int8 or int4")
    key, val = value.unbind(dim=1)
    encoded_key = quantize_kv(key) if key_format == "int8" else quantize_blocks(key)
    return encoded_key, quantize_blocks(val)
