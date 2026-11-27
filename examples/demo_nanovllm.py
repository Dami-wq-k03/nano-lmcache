"""nano-vLLM with nano-lmcache as its external L2 KV cache.

Completed KV blocks are handed to a bounded background writer and stored in
a CPU-offload (pinned memory) tier. The first question computes the shared
document and stores its blocks; the second question restores them from L2,
so only the question suffix is prefilled. Each request reports its time to
first token (TTFT) and how many prompt tokens came from L2.

    uv sync --extra hf --group nanovllm
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B
"""

import argparse
import atexit
import os
import time
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams

from nano_lmcache.integrations.nanovllm import KVBlockLayout, NanoVLLMExternalCache
from nano_lmcache.integrations.nanovllm.engine import attach, detach
from nano_lmcache.storage.async_write import AsyncWriteBehindStorage
from nano_lmcache.tensor.offload import CPUOffloadKVCache

MAX_PENDING_WRITES = 64

DOCUMENT = (
    "The city council met on Tuesday to discuss the new transit plan. The "
    "plan adds two bus lines, extends the tram route to the airport, and "
    "replaces the oldest vehicles with electric ones. Supporters say it will "
    "cut commute times, while critics worry about the cost to taxpayers. "
) * 48
QUESTIONS = [
    "Question: what does the plan add? Answer:",
    "Question: why do critics object? Answer:",
]


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def generate(llm, prompt, max_tokens):
    if isinstance(prompt, str):
        prompt = llm.tokenizer.encode(prompt)
    torch.manual_seed(0)
    params = SamplingParams(temperature=1e-6, max_tokens=max_tokens)

    torch.cuda.synchronize()
    start = time.perf_counter()
    llm.add_request(prompt, params)
    seq = llm.scheduler.waiting[-1]
    ttft_ms = None
    while not llm.is_finished():
        llm.step()
        torch.cuda.current_stream().synchronize()
        if ttft_ms is None and seq.num_completion_tokens:
            ttft_ms = (time.perf_counter() - start) * 1000
    assert ttft_ms is not None
    return llm.tokenizer.decode(seq.completion_token_ids), ttft_ms


def engine_counters(manager):
    stats = manager.stats
    return (
        stats.l1_hit_blocks,
        stats.l2_hit_blocks,
        stats.prefill_tokens,
        stats.stored_blocks,
    )


def ask(llm, manager, token_ids, label):
    before = engine_counters(manager)

    torch.cuda.synchronize()
    start = time.perf_counter()
    answer, ttft_ms = generate(llm, token_ids, 1)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    l1, l2, computed, stored = (
        now - then
        for now, then in zip(engine_counters(manager), before, strict=True)
    )
    l2_tokens = l2 * manager.block_size
    print(label)
    print(f"  prompt {len(token_ids)} tokens: {l1} blocks from L1 (GPU), "
          f"{l2} blocks ({l2_tokens} tokens) from L2 (CPU), "
          f"{computed} tokens computed, {stored} blocks stored")
    print(f"  TTFT: {ttft_ms:.1f} ms (request time: {elapsed_ms:.1f} ms)")
    print(f"  answer: {answer.strip()!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nano-vLLM with nano-lmcache as its external L2 KV cache"
    )
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.5,
        help="fraction of GPU memory nano-vLLM may use (default: 0.5)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="maximum context length (default: 4096)",
    )
    args = parser.parse_args()
    model = os.path.expanduser(args.model)

    llm = LLM(model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    storage = None
    try:
        layout = KVBlockLayout.from_hf_config(
            llm.model_runner.config.hf_config,
            model=Path(model).name,
            revision="local",
        )
        backend = CPUOffloadKVCache(device="cuda")
        storage = AsyncWriteBehindStorage(
            backend,
            max_pending=MAX_PENDING_WRITES,
            prepare=to_cpu,
        )

        for prompt in ("Hello", "warm up " * 650, "warm up " * 650):
            generate(llm, prompt, 2)
        manager = attach(llm, NanoVLLMExternalCache(layout, storage))

        for index, question in enumerate(QUESTIONS, start=1):
            if index > 1:
                detach(llm)
                manager = attach(llm, NanoVLLMExternalCache(layout, storage))
            ask(llm, manager, llm.tokenizer.encode(DOCUMENT + question),
                f"question-{index}")

        storage.flush()
        cached_blocks = len(backend)
        print(f"L2 now holds {cached_blocks} blocks "
              f"({cached_blocks * manager.block_size} tokens) in CPU memory.")
    finally:
        try:
            if storage is not None:
                storage.close()
        finally:
            atexit.unregister(llm.exit)
            llm.exit()


if __name__ == "__main__":
    main()
