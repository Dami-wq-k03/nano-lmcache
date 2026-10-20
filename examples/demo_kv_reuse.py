"""Reuse prefix KV with a real model and check the result is exact.

Needs the `hf` extra:

    uv sync --extra hf
    uv run python examples/demo_kv_reuse.py
    uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B --device cuda
    uv run python examples/demo_kv_reuse.py --device cuda --offload  # KV in CPU RAM
    # GPU -> CPU -> disk. Run twice: the second process reuses KV from disk.
    uv run python examples/demo_kv_reuse.py --device cuda --offload --disk /tmp/kv

Two prompts share a long prefix (think: the same document, different
questions). The second one reuses the first one's KV for that prefix, and
we compare its logits with a from-scratch computation to show that reuse
changes the cost, not the answer.

The logits always match. Wall-clock time is another story: reuse only wins
when the prefix is long enough that skipped computation outweighs the cost
of rebuilding and storing the cache. On a fast GPU with a small model and a
~270 token prompt it can even be slower than recomputing; try a longer
document, a bigger model, or CPU to see the saving (roadmap task G3 studies
this trade-off).
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.tiered import TieredStorage
from nano_lmcache.tensor.hf import run_with_prefix_cache
from nano_lmcache.tensor.offload import CPUOffloadKVCache
from nano_lmcache.tensor.serializers import TensorSerializer

DOCUMENT = (
    "The city council met on Tuesday to discuss the new transit plan. "
    "The plan adds two bus lines, extends the tram route to the airport, "
    "and replaces the oldest vehicles with electric ones. Supporters say it "
    "will cut commute times, while critics worry about the cost to taxpayers. "
    "The council will vote after a month of public comment. "
) * 4


def timed(function):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--offload",
        action="store_true",
        help="keep cached KV in pinned CPU memory instead of on the device",
    )
    parser.add_argument(
        "--disk",
        type=Path,
        help="also write KV through to this directory, reusable across runs",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).to(args.device).eval()

    prompts = {
        "A": DOCUMENT + "Question: what does the plan add?",
        "B": DOCUMENT + "Question: why do critics object?",
    }
    token_ids = {
        name: tokenizer(text)["input_ids"] for name, text in prompts.items()
    }

    fast: CacheStorage | None = (
        CPUOffloadKVCache(device=args.device) if args.offload else None
    )
    storage: CacheStorage | None = fast
    if args.disk is not None:
        disk = DiskKVCache(args.disk, serializer=TensorSerializer())
        print(f"Disk tier {args.disk}: {len(disk)} chunks from earlier runs")
        # Not `fast or ...`: an empty cache has len() 0, which is falsy.
        storage = TieredStorage(
            fast=fast if fast is not None else MemoryKVCache(), slow=disk
        )
    cache = ChunkedPrefixCache(
        chunk_size=args.chunk_size,
        namespace=args.model.replace("/", "-"),
        storage=storage,
    )

    # Warm up so one-time setup (CUDA context, kernel loading) is not timed.
    with torch.no_grad():
        model(torch.tensor([token_ids["A"][:8]], device=args.device))

    print(f"Model: {args.model} on {args.device}, chunk size {args.chunk_size}\n")

    for name in ("A", "B"):
        ids = token_ids[name]
        result, reuse_seconds = timed(
            lambda ids=ids: run_with_prefix_cache(model, cache, ids)
        )

        with torch.no_grad():
            reference, full_seconds = timed(
                lambda ids=ids: model(torch.tensor([ids], device=args.device))
            )

        # Compare the logits of the positions the cached run computed.
        expected = reference.logits[:, -result.computed_tokens :]
        max_diff = (result.logits - expected).abs().max().item()

        print(f"Prompt {name}: {len(ids)} tokens")
        print(f"  reused from cache: {result.matched_tokens} tokens")
        print(f"  computed:          {result.computed_tokens} tokens")
        print(f"  time with cache:   {reuse_seconds * 1000:8.1f} ms")
        print(f"  time from scratch: {full_seconds * 1000:8.1f} ms")
        print(f"  max |logit diff|:  {max_diff:.2e}\n")

    print(f"Cache metrics: hit rate {cache.metrics.hit_rate:.1%}")
    if isinstance(fast, CPUOffloadKVCache):
        print(f"KV held in CPU memory: {fast.nbytes / 2**20:.1f} MiB")
    if isinstance(storage, TieredStorage):
        stats = storage.tier_stats
        print(
            f"Chunks served: {stats.fast_hits} from memory, "
            f"{stats.slow_hits} from disk"
        )


if __name__ == "__main__":
    main()
