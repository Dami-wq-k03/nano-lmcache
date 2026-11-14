"""Optional GPU page kernels; imports/builds happen only on first use.

The fast path (``validate=False``) requires a caller-validated page table.
Scatter IDs must be unique: duplicate destinations have no GPU ordering.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

from nano_lmcache.experimental.dataplane import KVLayout
from nano_lmcache.tensor.page_ops import _check_page_batch

Backend = Literal["triton", "cuda"]


@lru_cache(maxsize=1)
def _cuda_extension() -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name="nano_lmcache_pages_v1",
        sources=[str(Path(__file__).with_name("page_kernel.cu"))],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _validate(
    pages: torch.Tensor,
    indices: torch.Tensor,
    layout: KVLayout | None,
    *,
    scatter: bool,
    validate: bool,
) -> None:
    if not pages.is_cuda or pages.ndim < 2 or not pages.is_contiguous():
        raise ValueError("kernels require contiguous CUDA pages with a payload axis")
    if (
        indices.ndim != 1
        or indices.dtype != torch.long
        or indices.device != pages.device
        or not indices.is_contiguous()
    ):
        raise ValueError("indices must be contiguous CUDA int64 on the pages device")
    if layout is not None:
        _check_page_batch(pages, layout)
    if validate and indices.numel():
        if indices.min().item() < 0 or indices.max().item() >= pages.shape[0]:
            raise IndexError("out-of-range page ID")
        if scatter and torch.unique(indices).numel() != indices.numel():
            raise ValueError("scatter requires unique destination page IDs")


def _move(
    source: torch.Tensor,
    indices: torch.Tensor,
    destination: torch.Tensor,
    *,
    scatter: bool,
    backend: Backend,
) -> None:
    if not indices.numel() or not source.numel():
        return
    with torch.cuda.device(source.device):
        if backend == "cuda":
            _cuda_extension().move(source, indices, destination, scatter)
        elif backend == "triton":
            from nano_lmcache.tensor._triton_kernels import move_pages

            move_pages(source, indices, destination, scatter=scatter)
        else:
            raise ValueError(f"unknown kernel backend {backend!r}")


def gather_pages_gpu(
    pages: torch.Tensor,
    indices: torch.Tensor,
    *,
    layout: KVLayout | None = None,
    backend: Backend = "triton",
    validate: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack non-contiguous page IDs in a single GPU kernel launch."""
    _validate(pages, indices, layout, scatter=False, validate=validate)
    shape = (indices.numel(), *pages.shape[1:])
    if out is None:
        out = torch.empty(shape, device=pages.device, dtype=pages.dtype)
    if (
        out.shape != shape
        or out.dtype != pages.dtype
        or out.device != pages.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must match the gathered shape, dtype and device")
    if (
        out.numel()
        and out.untyped_storage().data_ptr() == pages.untyped_storage().data_ptr()
    ):
        raise ValueError("gather input and output must not alias")
    _move(pages, indices, out, scatter=False, backend=backend)
    return out


def scatter_pages_gpu_(
    destination: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    layout: KVLayout | None = None,
    backend: Backend = "triton",
    validate: bool = True,
) -> torch.Tensor:
    """Unpack pages into unique destinations without cloning the full pool."""
    _validate(destination, indices, layout, scatter=True, validate=validate)
    if (
        values.shape != (indices.numel(), *destination.shape[1:])
        or values.dtype != destination.dtype
        or values.device != destination.device
        or not values.is_contiguous()
    ):
        raise ValueError("values must match page count, payload, dtype and device")
    if (
        values.numel()
        and values.untyped_storage().data_ptr()
        == destination.untyped_storage().data_ptr()
    ):
        raise ValueError("scatter input and output must not alias")
    _move(values, indices, destination, scatter=True, backend=backend)
    return destination
