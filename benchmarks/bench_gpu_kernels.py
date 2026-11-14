"""Matched GPU kernel, quantization, stream-overlap and peer-copy baselines."""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from bench_gpu_dataplane import environment

from nano_lmcache.tensor.codecs import dequantize_kv_gpu, quantize_kv_gpu
from nano_lmcache.tensor.kernels import gather_pages_gpu, scatter_pages_gpu_
from nano_lmcache.tensor.peer import PeerTransferEngine
from nano_lmcache.tensor.quantization import dequantize_kv, quantize_kv


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def measure(function: Callable[[], Any], warmup: int, repeats: int) -> dict[str, float]:
    device_ms, wall_ms = [], []
    for i in range(warmup + repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        begin = time.perf_counter()
        function()
        call_ms = (time.perf_counter() - begin) * 1000
        end.record()
        end.synchronize()
        if i >= warmup:
            device_ms.append(start.elapsed_time(end))
            wall_ms.append(call_ms)
    return {
        "p50_ms": statistics.median(device_ms),
        "p95_ms": percentile(device_ms, 0.95),
        "p99_ms": percentile(device_ms, 0.99),
        "host_call_p50_ms": statistics.median(wall_ms),
    }


def page_sweep(counts: list[int], tokens: list[int], warmup: int, repeats: int) -> list:
    rows = []
    for count in counts:
        for size in tokens:
            source = torch.randn(
                count, 2, 2, 4, size, 64, device="cuda", dtype=torch.float16
            )
            ids = torch.randperm(count, device="cuda")
            out = torch.empty_like(source)
            for operation in ("gather", "scatter"):

                def loop(ids=ids, source=source, out=out, operation=operation):
                    for i, page in enumerate(ids.tolist()):
                        if operation == "gather":
                            out[i].copy_(source[page])
                        else:
                            out[page].copy_(source[i])

                def torch_op(source=source, ids=ids, out=out, operation=operation):
                    if operation == "gather":
                        return torch.index_select(source, 0, ids, out=out)
                    return out.index_copy_(0, ids, source)

                functions = {"python_loop": loop, "torch_batched": torch_op}
                for backend in ("triton", "cuda"):
                    if operation == "gather":
                        functions[backend] = partial(
                            gather_pages_gpu,
                            source,
                            ids,
                            backend=backend,
                            validate=False,
                            out=out,
                        )
                    else:
                        functions[backend] = partial(
                            scatter_pages_gpu_,
                            out,
                            ids,
                            source,
                            backend=backend,
                            validate=False,
                        )
                for backend, fn in functions.items():
                    fn()
                    expected = source[ids] if operation == "gather" else source
                    actual = out if operation == "gather" else out[ids]
                    assert torch.equal(actual, expected)
                    result = measure(fn, warmup, repeats)
                    result.update(
                        kind="pages",
                        backend=backend,
                        operation=operation,
                        pages=count,
                        page_tokens=size,
                        traffic_bytes=2 * source.nbytes,
                        effective_gbps=2 * source.nbytes / result["p50_ms"] / 1e6,
                    )
                    rows.append(result)
    return rows


def quant_sweep(warmup: int, repeats: int) -> list:
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        source = torch.randn(28, 2, 8, 64, 128, dtype=dtype, device="cuda")
        for backend, encode, decode in (
            ("torch", quantize_kv, dequantize_kv),
            ("triton", quantize_kv_gpu, dequantize_kv_gpu),
        ):
            encoded = encode(source)
            restored = decode(encoded)
            error = (
                (restored.float() - source.float()).norm() / source.float().norm()
            ).item()
            for op, fn in (
                ("encode", partial(encode, source)),
                ("decode", partial(decode, encoded)),
            ):
                result = measure(fn, warmup, repeats)
                result.update(
                    kind="int8",
                    backend=backend,
                    operation=op,
                    dtype=str(dtype),
                    raw_bytes=source.nbytes,
                    encoded_bytes=encoded.nbytes,
                    relative_l2=error,
                )
                rows.append(result)
    return rows


def overlap(warmup: int, repeats: int) -> list:
    source = torch.randn(32 * 1024**2 // 2, device="cuda", dtype=torch.float16)
    host = torch.empty_like(source, device="cpu", pin_memory=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    copy_stream = torch.cuda.Stream()
    rows = []
    for mode in ("copy_only", "gemm_only", "serial", "overlap"):

        def run(mode=mode):
            current = torch.cuda.current_stream()
            copy_stream.wait_stream(current)
            if mode in ("copy_only", "serial"):
                host.copy_(source, non_blocking=True)
            elif mode == "overlap":
                with torch.cuda.stream(copy_stream):
                    host.copy_(source, non_blocking=True)
            if mode != "copy_only":
                torch.mm(a, b, out=output)
            if mode == "overlap":
                current.wait_stream(copy_stream)

        rows.append({"kind": "overlap", "mode": mode, **measure(run, warmup, repeats)})
    return rows


def peer_sweep(warmup: int, repeats: int) -> list:
    rows = []
    if torch.cuda.device_count() < 2:
        return [{"kind": "peer", "status": "requires_two_visible_gpus"}]
    for src, dst in ((0, 1), (1, 0)):
        torch.cuda.set_device(src)
        with PeerTransferEngine(src, dst) as engine:
            for nbytes in (4096, 1024**2, 32 * 1024**2):
                source = torch.randn(
                    nbytes // 2, device=f"cuda:{src}", dtype=torch.float16
                )
                modes = ["host"] + (["peer"] if engine.peer_available else [])
                for mode in modes:
                    samples = []
                    for i in range(warmup + repeats):
                        begin = time.perf_counter()
                        transfer = engine.copy(source, mode=mode)
                        value = transfer.wait()
                        elapsed = (time.perf_counter() - begin) * 1000
                        if i >= warmup:
                            samples.append(elapsed)
                    assert torch.equal(value.cpu(), source.cpu())
                    median = statistics.median(samples)
                    rows.append(
                        dict(
                            kind="peer",
                            source=src,
                            target=dst,
                            bytes=nbytes,
                            transport=transfer.transport,
                            peer_available=engine.peer_available,
                            p50_wall_ms=median,
                            p95_wall_ms=percentile(samples, 0.95),
                            effective_gbps=nbytes / median / 1e6,
                        )
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", type=int, default=[16, 256, 1024])
    parser.add_argument("--tokens", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--peer", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if (
        args.repeats <= 0
        or args.warmup < 0
        or any(x <= 0 for x in args.pages + args.tokens)
    ):
        parser.error("repeats, pages and tokens must be positive; warmup nonnegative")
    torch.cuda.set_device(0)
    rows = (
        peer_sweep(args.warmup, args.repeats)
        if args.peer
        else (
            page_sweep(args.pages, args.tokens, args.warmup, args.repeats)
            + quant_sweep(args.warmup, args.repeats)
            + overlap(args.warmup, args.repeats)
        )
    )
    result = dict(
        environment=environment(torch.device("cuda:0")),
        command=sys.argv,
        warmup=args.warmup,
        repeats=args.repeats,
        results=rows,
    )
    encoded = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
