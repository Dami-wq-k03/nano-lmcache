"""Measure GPU <-> CPU copy speed for KV chunks, by copy mode.

Needs a CUDA GPU and the `torch` extra. Not run in CI.

    uv run python benchmarks/bench_offload.py
    uv run python benchmarks/bench_offload.py --device cuda:1 --megabytes 256

Three modes are compared, matching `CPUOffloadKVCache` options:

- pageable, blocking: ordinary CPU memory. The driver stages every copy
  through a hidden pinned buffer, so bandwidth suffers.
- pinned, blocking: page-locked memory the GPU can DMA into directly.
- pinned, non_blocking: same bandwidth, but the Python call returns as soon
  as the copy is queued, so the CPU can do other work while it runs.

Bandwidth is timed with CUDA events around the copy itself. `call ms` is
how long the Python call took to return, which shows the non-blocking
effect: the copy still takes as long, but the caller is not stuck waiting.
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

MODES = ((False, False), (True, False), (True, True))  # (pinned, non_blocking)


@dataclass(frozen=True)
class CopyResult:
    """Best-of-N timings for one copy mode."""

    pinned: bool
    non_blocking: bool
    megabytes: float
    to_cpu_gbps: float
    to_gpu_gbps: float
    # Time for the `copy_` call to return to Python, GPU -> CPU direction.
    call_ms: float


def chunk_megabytes(
    *, layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per_value: int
) -> float:
    """Size of one chunk payload: [layers, 2 (K/V), kv_heads, tokens, head_dim]."""
    return layers * 2 * kv_heads * tokens * head_dim * bytes_per_value / 2**20


def _time_copy(
    dst: torch.Tensor, src: torch.Tensor, non_blocking: bool
) -> tuple[float, float]:
    """Return (GPU-measured seconds, seconds until the call returned)."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()
    start.record()
    call_start = time.perf_counter()
    dst.copy_(src, non_blocking=non_blocking)
    call_seconds = time.perf_counter() - call_start
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000, call_seconds


def run_mode(
    device: str, megabytes: float, pinned: bool, non_blocking: bool, repeats: int
) -> CopyResult:
    numel = int(megabytes * 2**20 / 2)  # bfloat16: 2 bytes per value
    gpu = torch.randn(numel, device=device).to(torch.bfloat16)
    cpu = torch.empty(numel, dtype=torch.bfloat16, pin_memory=pinned)
    nbytes = numel * 2

    best_to_cpu = best_to_gpu = best_call = float("inf")
    with torch.cuda.device(device):
        for _ in range(repeats):
            seconds, call = _time_copy(cpu, gpu, non_blocking)
            best_to_cpu, best_call = min(best_to_cpu, seconds), min(best_call, call)
            seconds, _ = _time_copy(gpu, cpu, non_blocking)
            best_to_gpu = min(best_to_gpu, seconds)

    return CopyResult(
        pinned=pinned,
        non_blocking=non_blocking,
        megabytes=nbytes / 2**20,
        to_cpu_gbps=nbytes / best_to_cpu / 1e9,
        to_gpu_gbps=nbytes / best_to_gpu / 1e9,
        call_ms=best_call * 1000,
    )


def run_benchmark(
    device: str = "cuda", megabytes: float = 64, repeats: int = 5
) -> list[CopyResult]:
    """Measure every copy mode once, after one warm-up copy."""
    run_mode(device, min(megabytes, 1), True, False, 1)  # warm up CUDA
    return [
        run_mode(device, megabytes, pinned, non_blocking, repeats)
        for pinned, non_blocking in MODES
    ]


def format_table(results: Sequence[CopyResult]) -> str:
    header = (
        f"{'memory':>9} {'non_blocking':>13} {'GPU->CPU GB/s':>14} "
        f"{'CPU->GPU GB/s':>14} {'call ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        memory = "pinned" if r.pinned else "pageable"
        lines.append(
            f"{memory:>9} {str(r.non_blocking):>13} {r.to_cpu_gbps:>14.1f} "
            f"{r.to_gpu_gbps:>14.1f} {r.call_ms:>9.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--megabytes", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    # Default: one 256-token chunk of Qwen3-0.6B in bfloat16 (~28 MiB).
    megabytes = args.megabytes or chunk_megabytes(
        layers=28, kv_heads=8, head_dim=128, tokens=256, bytes_per_value=2
    )
    print(
        f"{torch.cuda.get_device_name(args.device)}, "
        f"{megabytes:.1f} MiB per copy, best of {args.repeats}\n"
    )
    print(format_table(run_benchmark(args.device, megabytes, args.repeats)))


if __name__ == "__main__":
    main()
