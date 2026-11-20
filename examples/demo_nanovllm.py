"""Run nano-vLLM twice with nano-lmcache as its external L2 KV cache.

The first run stores the shared document's complete blocks on disk. The
second run is a new process, so nano-vLLM's GPU prefix cache (L1) starts
empty and the prompt blocks come from nano-lmcache (L2).

    uv sync --extra hf --group nanovllm
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B
"""

import argparse
import atexit
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams

from nano_lmcache.integrations.nanovllm import KVBlockLayout, NanoVLLMExternalCache
from nano_lmcache.integrations.nanovllm.engine import attach
from nano_lmcache.observability import TraceRecorder, TracingStorage, request_context
from nano_lmcache.storage.async_write import AsyncWriteBehindStorage
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.tiered import TieredStorage
from nano_lmcache.tensor.serializers import TensorSerializer

DEFAULT_DIR = Path(tempfile.gettempdir()) / "nano-lmcache-nanovllm"

DOCUMENT = (
    "The city council met on Tuesday to discuss the new transit plan. The "
    "plan adds two bus lines, extends the tram route to the airport, and "
    "replaces the oldest vehicles with electric ones. Supporters say it will "
    "cut commute times, while critics worry about the cost to taxpayers. "
) * 24
QUESTIONS = [
    "Question: what does the plan add? Answer:",
    "Question: why do critics object? Answer:",
]


def generate(llm, prompt, max_tokens):
    """Run one request with a fixed seed so outputs are repeatable.

    nano-vLLM forbids temperature 0 and breaks ties with random noise;
    bfloat16 logits tie often, so the seed keeps reruns identical.
    """
    torch.manual_seed(0)
    (output,) = llm.generate(
        [prompt],
        SamplingParams(temperature=1e-6, max_tokens=max_tokens),
        use_tqdm=False,
    )
    return output


def finish_trace(recorder, **metadata):
    """Close every request still open in the trace."""
    if recorder is None:
        return
    events = recorder.events
    completed = {
        event.request_id
        for event in events
        if event.event_type == "request_completed"
    }
    for event in events:
        if event.event_type == "request_started" and event.request_id not in completed:
            recorder.record(
                "request_completed",
                request_id=event.request_id,
                result="completed",
                metadata=metadata,
            )


def engine_counters(manager):
    """Return the four engine counters in the order `ask` prints them."""
    stats = manager.stats
    return (
        stats.l1_hit_blocks,
        stats.l2_hit_blocks,
        stats.prefill_tokens,
        stats.stored_blocks,
    )


def build_storage(disk, args, recorder):
    """Build memory in front of disk, with optional tracing and async writes.

    The tiered storage is write-through, so every stored block reaches disk.
    """
    fast = MemoryKVCache(max_entries=args.memory_max_entries)
    if recorder is None:
        slow = disk
    else:
        fast = TracingStorage(fast, recorder, name="memory")
        slow = TracingStorage(disk, recorder, name="disk")
    storage = TieredStorage(fast=fast, slow=slow, trace_recorder=recorder)
    if not args.async_store:
        return storage, None
    write_behind = AsyncWriteBehindStorage(storage, max_pending=args.write_queue)
    return write_behind, write_behind


def ask(llm, manager, token_ids, label, recorder=None):
    """Generate one token for TTFT, then a short answer, and print the stats."""
    before = engine_counters(manager)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with request_context(label + "/ttft"):
        generate(llm, token_ids, 1)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - start) * 1000
    finish_trace(recorder, ttft_ms=ttft_ms)

    with request_context(label + "/answer"):
        answer = generate(llm, token_ids, 24)
    finish_trace(recorder)

    l1, l2, computed, stored = (
        now - then
        for now, then in zip(engine_counters(manager), before, strict=True)
    )
    print(label)
    print(f"  prompt {len(token_ids)} tokens: {l1} blocks from L1 (GPU), "
          f"{l2} from L2 (disk), {computed} tokens computed, {stored} blocks stored")
    print(f"  time to first token: {ttft_ms:.1f} ms")
    print(f"  answer: {answer['text'].strip()!r}\n")


def main() -> None:
    """Parse arguments, attach L2 to nano-vLLM, and run the questions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--trace-output", type=Path, help="metadata-only Trace JSON")
    parser.add_argument(
        "--memory-max-entries", type=int, default=None,
        help="limit the memory L2 tier to observe disk restores and eviction",
    )
    parser.add_argument(
        "--async-store", action="store_true",
        help="queue L2 writes in a bounded background worker",
    )
    parser.add_argument(
        "--write-queue", type=int, default=32,
        help="maximum pending L2 writes with --async-store",
    )
    args = parser.parse_args()

    if args.clear:
        shutil.rmtree(args.dir, ignore_errors=True)
        print(f"Cleared {args.dir}")
        return

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.5, max_model_len=4096)
    write_behind = None
    try:
        layout = KVBlockLayout.from_hf_config(
            llm.model_runner.config.hf_config,
            model=Path(args.model).name,
            revision="local",
        )
        disk = DiskKVCache(args.dir, serializer=TensorSerializer())
        print(f"L2 on disk: {args.dir} ({len(disk)} blocks from earlier runs)\n")
        recorder = TraceRecorder() if args.trace_output is not None else None
        storage, write_behind = build_storage(disk, args, recorder)
        for prompt in ("Hello", "warm up " * 650, "warm up " * 650):
            generate(llm, prompt, 2)
        manager = attach(llm, NanoVLLMExternalCache(layout, storage, recorder))

        for index, question in enumerate(QUESTIONS, start=1):
            ask(llm, manager, llm.tokenizer.encode(DOCUMENT + question),
                f"question-{index}", recorder)

        if write_behind is not None:
            write_behind.flush()
        print(f"L2 now holds {len(disk)} blocks on disk.")
        if manager.stats.l2_hit_blocks == 0:
            print("Run the script again: the next process restores them from disk.")
        if recorder is not None:
            trace = {
                "format_version": 1,
                "source": "nanovllm",
                "events": recorder.to_list(),
            }
            args.trace_output.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            print(f"Trace written to {args.trace_output}")
    finally:
        if write_behind is not None:
            write_behind.close()
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()
