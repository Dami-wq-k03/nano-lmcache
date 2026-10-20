"""Reuse prefix KV with a real model and check the result is exact.

Needs the `hf` extra:

    uv sync --extra hf
    uv run python examples/demo_kv_reuse.py
    uv run python examples/demo_kv_reuse.py --model Qwen/Qwen3-0.6B --device cuda

Two prompts share a long prefix (think: the same document, different
questions). The second one reuses the first one's KV for that prefix, and
we compare its logits with a from-scratch computation to show that reuse
changes the cost, not the answer.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.tensor.hf import run_with_prefix_cache

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

    cache = ChunkedPrefixCache(
        chunk_size=args.chunk_size, namespace=args.model.replace("/", "-")
    )

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


if __name__ == "__main__":
    main()
