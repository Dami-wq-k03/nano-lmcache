"""Shared pytest configuration: skip tests whose heavy dependencies are absent.

The core package needs nothing but Python, and CI runs CPU-only. Tests that
need PyTorch, a GPU, or vLLM carry a marker and are skipped, never failed,
when the environment cannot run them:

- `torch`: needs PyTorch installed.
- `gpu`: needs PyTorch and a CUDA device.
- `vllm`: needs vLLM and a CUDA device.
"""

import importlib.util
from collections.abc import Collection

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_cuda() -> bool:
    if not _has_module("torch"):
        return False

    import torch

    return bool(torch.cuda.is_available())


def skip_reason(
    markers: Collection[str],
    *,
    has_torch: bool,
    has_cuda: bool,
    has_vllm: bool,
) -> str | None:
    """Return why a test with these markers must be skipped, or None."""
    needs_torch = bool(set(markers) & {"torch", "gpu", "vllm"})
    needs_cuda = bool(set(markers) & {"gpu", "vllm"})

    if needs_torch and not has_torch:
        return "requires torch (install with `uv sync --extra torch`)"
    if needs_cuda and not has_cuda:
        return "requires a CUDA GPU"
    if "vllm" in markers and not has_vllm:
        return "requires vllm"

    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    has_torch = _has_module("torch")
    has_vllm = _has_module("vllm")
    # Only touch CUDA if some collected test could need it.
    needs_cuda = any(
        {"gpu", "vllm"} & {marker.name for marker in item.iter_markers()}
        for item in items
    )
    has_cuda = _has_cuda() if needs_cuda else False

    for item in items:
        names = {marker.name for marker in item.iter_markers()}
        reason = skip_reason(
            names, has_torch=has_torch, has_cuda=has_cuda, has_vllm=has_vllm
        )
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
