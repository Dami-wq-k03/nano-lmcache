"""Triton DSL kernels, loaded only by the optional CUDA adapter."""

import torch
import triton
import triton.language as tl


@triton.jit
def _page_move(
    source,
    indices,
    destination,
    elements: tl.constexpr,
    page_elements: tl.constexpr,
    scatter: tl.constexpr,
    block: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    page = offsets // page_elements
    within = offsets % page_elements
    selected = tl.load(indices + page, offsets < elements, other=0)
    physical = selected * page_elements + within
    if scatter:
        values = tl.load(source + offsets, offsets < elements, other=0)
        tl.store(destination + physical, values, offsets < elements)
    else:
        values = tl.load(source + physical, offsets < elements, other=0)
        tl.store(destination + offsets, values, offsets < elements)


def move_pages(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    *,
    scatter: bool,
) -> None:
    page_elements = source[0].numel()
    elements = indices.numel() * page_elements
    _page_move[(triton.cdiv(elements, 1024),)](
        source, indices, destination, elements, page_elements, scatter, 1024
    )


@triton.jit
def _quantize_rows(
    source,
    values,
    scales,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    block_tokens: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    token = tl.arange(0, block_tokens)
    offsets = (group * tokens + token[:, None]) * channels + channel[None, :]
    mask = (token[:, None] < tokens) & (channel[None, :] < channels)
    x = tl.load(source + offsets, mask, other=0).to(tl.float32)
    scale = tl.max(tl.abs(x), axis=0) / 127.0
    scale = tl.where(scale == 0, 1.0, scale)
    q = tl.extra.cuda.libdevice.nearbyint(x / scale[None, :])
    q = tl.minimum(127.0, tl.maximum(-127.0, q)).to(tl.int8)
    tl.store(values + offsets, q, mask)
    tl.store(scales + group * channels + channel, scale, channel < channels)


@triton.jit
def _dequantize(
    values,
    scales,
    output,
    elements: tl.constexpr,
    tokens: tl.constexpr,
    channels: tl.constexpr,
    block: tl.constexpr,
) -> None:
    offset = tl.program_id(0) * block + tl.arange(0, block)
    scale_id = offset // (tokens * channels) * channels + offset % channels
    q = tl.load(values + offset, offset < elements, other=0).to(tl.float32)
    scale = tl.load(scales + scale_id, offset < elements, other=1)
    tl.store(output + offset, q * scale, offset < elements)


def quantize_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, channels = source.shape[-2:]
    groups = source.numel() // (tokens * channels)
    values = torch.empty_like(source, dtype=torch.int8)
    scales = torch.empty(
        (*source.shape[:-2], 1, channels), device=source.device, dtype=torch.float32
    )
    _quantize_rows[(groups, triton.cdiv(channels, 32))](
        source,
        values,
        scales,
        tokens,
        channels,
        triton.next_power_of_2(tokens),
        32,
    )
    return values, scales


def dequantize_rows(
    values: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    tokens, channels = values.shape[-2:]
    output = torch.empty_like(values, dtype=dtype)
    _dequantize[(triton.cdiv(values.numel(), 1024),)](
        values, scales, output, values.numel(), tokens, channels, 1024
    )
    return output
