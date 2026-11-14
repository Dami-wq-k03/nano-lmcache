"""Reference KV quantization primitives used by storage and GPU experiments."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class QuantizedKV:
    """INT8 values with per-head/channel scales and original dtype."""

    values: torch.Tensor
    scales: torch.Tensor
    dtype: torch.dtype
    token_axis: int

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.scales.nbytes)


def quantize_kv(value: torch.Tensor, *, token_axis: int = -2) -> QuantizedKV:
    """Quantize a floating KV tensor symmetrically along its token axis."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("quantize_kv needs a floating-point tensor")
    axis = token_axis if token_axis >= 0 else value.ndim + token_axis
    if not 0 <= axis < value.ndim:
        raise ValueError(
            "token_axis must address an existing dimension, "
            f"got {token_axis}"
        )

    source = value.detach().float()
    scales = source.abs().amax(dim=axis, keepdim=True) / 127
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    values = (source / scales).round().clamp(-127, 127).to(torch.int8)
    return QuantizedKV(values, scales, value.dtype, axis)


def dequantize_kv(
    value: QuantizedKV,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Restore a quantized KV tensor to its original dtype."""
    restored = value.values.float() * value.scales
    if device is not None:
        restored = restored.to(device)
    return restored.to(value.dtype)
