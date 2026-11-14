"""Mapped-host fused kernels versus GPU packing/quantization plus DMA."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from bench_gpu_dataplane import environment
from bench_gpu_kernels import measure

from nano_lmcache.tensor.codecs import quantize_kv_gpu
from nano_lmcache.tensor.kernels import gather_pages_gpu
from nano_lmcache.tensor.mapped_host import gather_to_host_gpu, quantize_to_host_gpu
from nano_lmcache.tensor.quantization import QuantizedKV, quantize_kv


def benchmark(warmup: int, repeats: int) -> list:
    rows = []
    for count in (16, 256, 1024):
        pages = torch.randn(count, 2, 2, 4, 16, 64, device="cuda", dtype=torch.float16)
        ids = torch.randperm(count, device="cuda")
        gpu = torch.empty_like(pages)
        host = torch.empty_like(pages, device="cpu", pin_memory=True)

        def pack_dma(backend="torch", pages=pages, ids=ids, gpu=gpu, host=host):
            if backend == "torch":
                torch.index_select(pages, 0, ids, out=gpu)
            else:
                gather_pages_gpu(pages, ids, backend=backend, validate=False, out=gpu)
            host.copy_(gpu, non_blocking=True)

        functions = {
            "torch_pack_dma": pack_dma,
            "cuda_pack_dma": partial(pack_dma, "cuda"),
            "cuda_mapped_pack": partial(
                gather_to_host_gpu, pages, ids, out=host, validate=False
            ),
        }
        for name, fn in functions.items():
            fn()
            torch.cuda.current_stream().synchronize()
            assert torch.equal(host, pages[ids].cpu())
            rows.append(
                dict(
                    kind="pack_export",
                    backend=name,
                    bytes=host.nbytes,
                    pages=count,
                    **measure(fn, warmup, repeats),
                )
            )
    for tokens in (16, 64, 256):
        value = torch.randn(28, 2, 8, tokens, 128, device="cuda", dtype=torch.float16)
        host = QuantizedKV(
            torch.empty_like(value, device="cpu", dtype=torch.int8, pin_memory=True),
            torch.empty(
                28, 2, 8, 1, 128, device="cpu", dtype=torch.float32, pin_memory=True
            ),
            value.dtype,
            value.ndim - 2,
        )

        def quant_dma(encode, value=value, host=host):
            encoded = encode(value)
            host.values.copy_(encoded.values, non_blocking=True)
            host.scales.copy_(encoded.scales, non_blocking=True)

        for name, fn in (
            ("torch_quant_dma", partial(quant_dma, quantize_kv)),
            ("triton_quant_dma", partial(quant_dma, quantize_kv_gpu)),
            ("cuda_mapped_quant", partial(quantize_to_host_gpu, value, out=host)),
        ):
            fn()
            torch.cuda.current_stream().synchronize()
            expected = quantize_kv(value.cpu())
            torch.testing.assert_close(host.scales, expected.scales)
            assert (host.values.int() - expected.values.int()).abs().max() <= 1
            rows.append(
                dict(
                    kind="quant_export",
                    backend=name,
                    tokens=tokens,
                    raw_bytes=value.nbytes,
                    host_bytes=host.nbytes,
                    **measure(fn, warmup, repeats),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative and repeats positive")
    torch.cuda.set_device(0)
    result = dict(
        environment=environment(torch.device("cuda:0")),
        command=sys.argv,
        warmup=args.warmup,
        repeats=args.repeats,
        allocation_scope=(
            "preallocated host outputs; quant_dma allocates GPU codec outputs"
        ),
        results=benchmark(args.warmup, args.repeats),
    )
    encoded = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
