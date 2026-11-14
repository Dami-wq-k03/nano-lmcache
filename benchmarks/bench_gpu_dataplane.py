"""Reproducible GPU copy and page gather/scatter measurements.

Examples::

    uv run python benchmarks/bench_gpu_dataplane.py --sizes 4K 1M 32M 256M \
        --json /tmp/nano-lmcache-gpu.json

The script records environment metadata with every result. It requires a CUDA
PyTorch installation and is intentionally separate from CI benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from nano_lmcache.tensor.page_ops import gather_pages, scatter_pages

MODES = (
    ("pageable", False, False),
    ("pinned", True, False),
    ("pinned_async", True, True),
)


def parse_size(value: str) -> int:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    suffix = value[-1].upper()
    if suffix in units:
        number = value[:-1]
        return int(float(number) * units[suffix])
    return int(value)


def _git_sha() -> str | None:
    if os.environ.get("NANO_LMCACHE_GIT_SHA"):
        return os.environ["NANO_LMCACHE_GIT_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device: torch.device) -> dict[str, Any]:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "src").rglob("*")):
        if path.suffix in (".py", ".cu"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    result: dict[str, Any] = {
        "source_sha256": digest.hexdigest(),
        "command": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_sha": _git_sha(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_count": torch.cuda.device_count(),
    }
    try:
        result["driver"] = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        result["driver"] = None
    try:
        result["gpu_clock_power"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.sm,clocks.mem,power.limit",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        result["gpu_clock_power"] = None
    return result


def _copy_once(
    source: torch.Tensor, destination: torch.Tensor, *, non_blocking: bool
) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call_start = time.perf_counter()
    destination.copy_(source, non_blocking=non_blocking)
    call_ms = (time.perf_counter() - call_start) * 1000
    end.record()
    end.synchronize()
    return start.elapsed_time(end), call_ms


def benchmark_copy(
    device: torch.device, sizes: list[int], warmup: int, repeats: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for nbytes in sizes:
        numel = (nbytes // 2) or 1
        source = torch.empty(numel, dtype=torch.bfloat16, device=device).normal_()
        for name, pinned, non_blocking in MODES:
            destination = torch.empty(
                numel, dtype=torch.bfloat16, device="cpu", pin_memory=pinned
            )
            samples: list[tuple[float, float, float]] = []
            for index in range(warmup + repeats):
                d2h_ms, call_ms = _copy_once(
                    source, destination, non_blocking=non_blocking
                )
                h2d_ms, _ = _copy_once(destination, source, non_blocking=non_blocking)
                if index >= warmup:
                    samples.append((d2h_ms, h2d_ms, call_ms))
            columns = [sorted(sample[j] for sample in samples) for j in range(3)]
            d2h_ms, h2d_ms, call_ms = [col[len(col) // 2] for col in columns]
            p95 = [col[int((len(col) - 1) * 0.95)] for col in columns]
            results.append(
                {
                    "kind": "copy",
                    "bytes": numel * 2,
                    "mode": name,
                    "d2h_ms": d2h_ms,
                    "h2d_ms": h2d_ms,
                    "d2h_gbps": numel * 2 / (d2h_ms / 1000) / 1e9,
                    "h2d_gbps": numel * 2 / (h2d_ms / 1000) / 1e9,
                    "call_ms": call_ms,
                    "d2h_p95_ms": p95[0],
                    "h2d_p95_ms": p95[1],
                    "call_p95_ms": p95[2],
                    "warmup": warmup,
                    "repeats": repeats,
                }
            )
    return results


def benchmark_page_ops(
    device: torch.device,
    *,
    page_count: int,
    page_tokens: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    shape = (page_count, 2, 2, 4, page_tokens, 16)
    pages = torch.randn(shape, device=device, dtype=torch.float16)
    indices = torch.arange(page_count - 1, -1, -1, device=device, dtype=torch.long)
    destination = torch.empty_like(pages)
    results: list[dict[str, Any]] = []
    for operation in ("gather", "scatter"):
        samples: list[float] = []
        for index in range(warmup + repeats):
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if operation == "gather":
                output = gather_pages(pages, indices)
            else:
                output = scatter_pages(destination, indices, pages)
            end.record()
            end.synchronize()
            if index >= warmup:
                samples.append(start.elapsed_time(end))
        samples.sort()
        milliseconds = samples[len(samples) // 2]
        bytes_moved = pages.nbytes + (output.nbytes if operation == "gather" else 0)
        results.append(
            {
                "kind": "page_ops",
                "operation": operation,
                "pages": page_count,
                "page_shape": list(shape[1:]),
                "bytes_moved": bytes_moved,
                "median_ms": milliseconds,
                "effective_gbps": bytes_moved / (milliseconds / 1000) / 1e9,
                "warmup": warmup,
                "repeats": repeats,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[4 * 1024, 1 * 1024**2, 32 * 1024**2, 256 * 1024**2],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--page-count", type=int, default=1024)
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report = {
        "environment": environment(device),
        "copy": benchmark_copy(device, args.sizes, args.warmup, args.repeats),
        "page_ops": benchmark_page_ops(
            device,
            page_count=args.page_count,
            page_tokens=args.page_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    encoded = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
