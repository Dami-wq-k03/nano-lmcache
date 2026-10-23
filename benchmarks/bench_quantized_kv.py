"""Measure INT8 serialization and G1 full/lossless/quantized KV reuse."""

import argparse
import json
import runpy
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.tensor.hf import run_with_prefix_cache
from nano_lmcache.tensor.serializers import QuantizedTensorSerializer, TensorSerializer


def errors(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm()).item(),
    }


def median_ms(operation, device, warmup, repeats):
    samples = []
    for index in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if index >= warmup:
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def bench_tensor(dtype, args):
    torch.manual_seed(0)
    value = torch.randn(28, 2, 8, 16, 128, device=args.device).to(dtype)
    serializer = QuantizedTensorSerializer()
    data = serializer.dumps(value)
    restored = serializer.loads(data)
    return {
        "shape": list(value.shape),
        "raw_bytes": value.nbytes,
        "lossless_bytes": len(TensorSerializer().dumps(value)),
        "int8_bytes": len(data),
        "raw_over_int8": value.nbytes / len(data),
        "encode_ms": median_ms(
            lambda: serializer.dumps(value), args.device, args.warmup, args.repeats
        ),
        "decode_ms": median_ms(
            lambda: serializer.loads(data), torch.device("cpu"),
            args.warmup, args.repeats,
        ),
        **errors(restored, value.cpu()),
    }


def bench_model(dtype, args):
    torch.manual_seed(0)
    if args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        demo = Path(__file__).resolve().parents[1] / "examples/demo_kv_reuse.py"
        document = runpy.run_path(str(demo))["DOCUMENT"]
        warm_ids = tokenizer(document + "Question: what does the plan add?")[
            "input_ids"
        ]
        ids = tokenizer(document + "Question: why do critics object?")["input_ids"]
        chunk_size = 16
    else:
        model = AutoModelForCausalLM.from_config(Qwen3Config(
            vocab_size=200, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256,
        )).to(dtype=dtype)
        warm_ids = list(range(1, 13)) + [50, 51]
        ids = list(range(1, 13)) + [90, 91, 92]
        chunk_size = 4
    model = model.to(args.device).eval()
    with torch.no_grad():
        full = model(torch.tensor([ids], device=args.device)).logits
    source = ChunkedPrefixCache(chunk_size=chunk_size, namespace="g5")
    run_with_prefix_cache(model, source, warm_ids)
    match = source.lookup(ids)
    kv = torch.stack(match.values).cpu()
    report = {"prompt_tokens": len(ids), "cached_tokens": match.matched_tokens}
    for name, serializer in (
        ("B_lossless", TensorSerializer()), ("C_int8", QuantizedTensorSerializer()),
    ):
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkedPrefixCache(
                chunk_size=chunk_size, namespace="g5",
                storage=DiskKVCache(directory, serializer=serializer),
            )
            cache.store(ids, match.values)
            cache = ChunkedPrefixCache(
                chunk_size=chunk_size, namespace="g5",
                storage=DiskKVCache(directory, serializer=serializer),
            )
            decoded = torch.stack(cache.lookup(ids).values)
            result = run_with_prefix_cache(model, cache, ids)
        assert result.matched_tokens == match.matched_tokens > 0
        expected = full[:, -result.computed_tokens:]
        agreement = result.logits.argmax(-1) == expected.argmax(-1)
        report[name] = {
            "logits_vs_A": errors(result.logits, expected),
            "greedy_matches": agreement.sum().item(),
            "greedy_positions": agreement.numel(),
            "next_token_match": agreement[0, -1].item(),
            "K_error": errors(decoded[:, :, 0], kv[:, :, 0]),
            "V_error": errors(decoded[:, :, 1], kv[:, :, 1]),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--model", help="Local HuggingFace model; default: tiny Qwen3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps({
        "torch": torch.__version__, "device": str(args.device),
        "gpu": torch.cuda.get_device_name(args.device)
        if args.device.type == "cuda" else None,
        "model": args.model or "tiny Qwen3 (random seed 0)",
        "threads": args.threads, "warmup": args.warmup, "repeats": args.repeats,
    }))
    for dtype in (torch.float16, torch.bfloat16):
        print(json.dumps({
            "dtype": str(dtype), "tensor": bench_tensor(dtype, args),
            "model": bench_model(dtype, args),
        }), flush=True)


if __name__ == "__main__":
    main()
