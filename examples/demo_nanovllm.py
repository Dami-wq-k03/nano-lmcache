"""nano-vLLM with nano-lmcache as its external L2 KV cache.

The disk cache persists between runs. Completed KV blocks are written
through a CPU-offloaded memory tier in front of disk, so running the
script twice restores the shared document from L2 in the second process.

    uv sync --extra hf --group nanovllm
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B
"""

import argparse
import atexit
import os
import tempfile
import time
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams

from nano_lmcache.integrations.nanovllm import KVBlockLayout, NanoVLLMExternalCache
from nano_lmcache.integrations.nanovllm.engine import attach
from nano_lmcache.storage.async_write import AsyncWriteBehindStorage
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.tiered import TieredStorage
from nano_lmcache.tensor.offload import CPUOffloadKVCache
from nano_lmcache.tensor.serializers import TensorSerializer

MAX_PENDING_WRITES = 64

CACHE_DIR = Path(tempfile.gettempdir()) / "nano-lmcache-nanovllm"

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
    """Run one request with a fixed seed so outputs are repeatable."""
    torch.manual_seed(0)
    (output,) = llm.generate(
        [prompt],
        SamplingParams(temperature=1e-6, max_tokens=max_tokens),
        use_tqdm=False,
    )
    return output


def engine_counters(manager):
    """Return the four engine counters in the order `ask` prints them."""
    stats = manager.stats
    return (
        stats.l1_hit_blocks,
        stats.l2_hit_blocks,
        stats.prefill_tokens,
        stats.stored_blocks,
    )


def ask(llm, manager, token_ids, label):
    """Run one request and time it from start to finish."""
    before = engine_counters(manager)

    torch.cuda.synchronize()
    start = time.perf_counter()
    answer = generate(llm, token_ids, 24)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    l1, l2, computed, stored = (
        now - then
        for now, then in zip(engine_counters(manager), before, strict=True)
    )
    print(label)
    print(f"  prompt {len(token_ids)} tokens: {l1} blocks from L1 (GPU), "
          f"{l2} from L2 (CPU/disk), {computed} tokens computed, "
          f"{stored} blocks stored")
    print(f"  request time: {elapsed_ms:.1f} ms")
    print(f"  answer: {answer['text'].strip()!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nano-vLLM with nano-lmcache as its external L2 KV cache"
    )
    parser.add_argument("--model", required=True, help="local model directory")
    args = parser.parse_args()
    model = os.path.expanduser(args.model)

    llm = LLM(model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.5, max_model_len=4096)
    storage = None
    try:
        layout = KVBlockLayout.from_hf_config(
            llm.model_runner.config.hf_config,
            model=Path(model).name,
            revision="local",
        )
        disk = DiskKVCache(CACHE_DIR, serializer=TensorSerializer())
        tiered = TieredStorage(
            fast=CPUOffloadKVCache(device="cuda"),
            slow=disk
        )
        storage = AsyncWriteBehindStorage(
            tiered,
            max_pending=MAX_PENDING_WRITES,
            prepare=to_cpu,
        )

        print(f"L2 on disk: {CACHE_DIR} ({len(disk)} blocks from earlier runs)\n")
        for prompt in ("Hello", "warm up " * 650, "warm up " * 650):
            generate(llm, prompt, 2)
        manager = attach(llm, NanoVLLMExternalCache(layout, storage))

        for index, question in enumerate(QUESTIONS, start=1):
            ask(llm, manager, llm.tokenizer.encode(DOCUMENT + question),
                f"question-{index}")

        storage.flush()
        print(f"L2 now holds {len(disk)} blocks on disk.")
        if manager.stats.l2_hit_blocks == 0:
            print("Run the script again: the next process restores them from disk.")
    finally:
        try:
            if storage is not None:
                storage.close()
        finally:
            atexit.unregister(llm.exit)
            llm.exit()


if __name__ == "__main__":
    main()
