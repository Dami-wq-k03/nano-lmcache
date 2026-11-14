"""Torch reference gather/scatter operations for batched KV pages."""

import torch

from nano_lmcache.experimental.dataplane import KVLayout


def _check_page_batch(pages: torch.Tensor, layout: KVLayout) -> None:
    expected = (layout.num_layers, 2, *layout.tensor_shape)
    if layout.layer_layout == "page-major":
        expected = (2, layout.num_layers, *layout.tensor_shape)
    if tuple(pages.shape[1:]) != expected:
        raise ValueError(
            f"pages have shape {tuple(pages.shape)}; expected [pages, {expected}]"
        )
    if pages.dtype != _torch_dtype(layout.dtype):
        raise ValueError(
            f"pages use dtype {pages.dtype}; layout requires "
            f"{_torch_dtype(layout.dtype)}"
        )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.").lower()
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    normalized = aliases.get(normalized, normalized)
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype {name!r}")
    return dtype


def gather_pages(
    pages: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    layout: KVLayout | None = None,
) -> torch.Tensor:
    """Gather pages in ``page_indices`` order without modifying the source."""
    if pages.ndim < 2:
        raise ValueError("pages must have a page axis and at least one payload axis")
    if page_indices.ndim != 1 or page_indices.dtype != torch.long:
        raise ValueError("page_indices must be a one-dimensional torch.long tensor")
    if page_indices.device != pages.device:
        raise ValueError("page_indices and pages must be on the same device")
    if page_indices.numel() and (
        page_indices.min().item() < 0 or page_indices.max().item() >= pages.shape[0]
    ):
        raise IndexError("page_indices contains an out-of-range page")
    if layout is not None:
        _check_page_batch(pages, layout)
    return pages.index_select(0, page_indices)


def scatter_pages(
    destination: torch.Tensor,
    page_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    layout: KVLayout | None = None,
) -> torch.Tensor:
    """Scatter page values into a cloned destination using ``index_copy``."""
    if destination.ndim < 2 or values.ndim != destination.ndim:
        raise ValueError("destination and values must have matching payload rank")
    if page_indices.ndim != 1 or page_indices.dtype != torch.long:
        raise ValueError("page_indices must be a one-dimensional torch.long tensor")
    if page_indices.device != destination.device or values.device != destination.device:
        raise ValueError("pages, values, and page_indices must share a device")
    if (
        values.shape[0] != page_indices.shape[0]
        or values.shape[1:] != destination.shape[1:]
    ):
        raise ValueError(
            "values shape must match page_indices and destination payload shape"
        )
    if page_indices.numel() and (
        page_indices.min().item() < 0
        or page_indices.max().item() >= destination.shape[0]
    ):
        raise IndexError("page_indices contains an out-of-range page")
    if layout is not None:
        _check_page_batch(destination, layout)
        _check_page_batch(values, layout)
    return destination.clone().index_copy_(0, page_indices, values)
