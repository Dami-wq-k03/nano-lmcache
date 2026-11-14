"""nano-lmcache as the L2 KV cache of nano-vLLM: run it twice.

Needs Linux, an NVIDIA GPU, and the nanovllm dependency group:

    uv sync --extra hf --group nanovllm
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B   # 1st
    uv run python examples/demo_nanovllm.py --model /path/to/Qwen3-0.6B   # 2nd

The first run computes the shared document's KV and nano-lmcache stores
every complete block on disk. The second run is a new process: nano-vLLM's
own GPU prefix cache (L1) starts empty, so its prompt blocks can only come
from nano-lmcache (L2). Inside each run, a second question about the same
document hits L1, the way nano-vLLM already works on its own.

nano-vLLM always uses the first visible GPU; pick one with
CUDA_VISIBLE_DEVICES. Pass --clear to empty the disk cache.

Read the timings honestly. Storing is synchronous in this version: the
first run's first request also waits for its blocks to be written to disk
(with an fsync per block). And with a small model on a fast GPU,
recomputing a prompt can be cheaper than reading its blocks back from
disk. What L2 buys here is reuse across processes; whether it is also
faster depends on the model, the prompt length, and the storage tier,
which roadmap task N4 measures.
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


def ask(llm, manager, token_ids, label, recorder=None):  # type: ignore[no-untyped-def]
    """Generate one token (time to first token), then a short answer."""
    before = (manager.stats.l1_hit_blocks, manager.stats.l2_hit_blocks,
              manager.stats.prefill_tokens, manager.stats.stored_blocks)

    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.manual_seed(0)
    trace_start = len(recorder) if recorder is not None else 0
    with request_context(label + "/ttft"):
        llm.generate([token_ids], SamplingParams(temperature=1e-6, max_tokens=1),
                     use_tqdm=False)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start
    finish_trace(recorder, trace_start, ttft_ms=ttft * 1000)

    l1, l2, computed, stored = (
        now - then
        for now, then in zip(
            (manager.stats.l1_hit_blocks, manager.stats.l2_hit_blocks,
             manager.stats.prefill_tokens, manager.stats.stored_blocks),
            before,
            strict=True,
        )
    )
    # nano-vLLM forbids temperature 0 and breaks ties between equally likely
    # tokens with random noise; bfloat16 logits tie often. A fixed seed makes
    # the answer repeatable across runs.
    torch.manual_seed(0)
    trace_start = len(recorder) if recorder is not None else 0
    with request_context(label + "/answer"):
        (answer,) = llm.generate(
            [token_ids], SamplingParams(temperature=1e-6, max_tokens=24),
            use_tqdm=False,
        )
    finish_trace(recorder, trace_start)
    print(f"{label}")
    print(f"  prompt {len(token_ids)} tokens: {l1} blocks from L1 (GPU), "
          f"{l2} from L2 (disk), {computed} tokens computed, {stored} blocks stored")
    print(f"  time to first token: {ttft * 1000:.1f} ms")
    print(f"  answer: {answer['text'].strip()!r}\n")


def finish_trace(recorder, start, **metadata):  # type: ignore[no-untyped-def]
    """Complete this generate call's requests without capturing text/tensors."""
    if recorder is None:
        return
    for event in recorder.events[start:]:
        if event.event_type == "request_started":
            recorder.record(
                "request_completed",
                request_id=event.request_id,
                result="completed",
                metadata=metadata,
            )


def main() -> None:
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
        "--async-store",
        action="store_true",
        help="queue L2 writes in a bounded background worker",
    )
    parser.add_argument(
        "--write-queue",
        type=int,
        default=32,
        help="maximum pending L2 writes with --async-store",
    )
    args = parser.parse_args()

    if args.clear:
        shutil.rmtree(args.dir, ignore_errors=True)
        print(f"Cleared {args.dir}")
        return

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=0.5, max_model_len=4096)
    write_behind: AsyncWriteBehindStorage | None = None
    try:
        layout = KVBlockLayout.from_hf_config(
            llm.model_runner.config.hf_config,
            model=Path(args.model).name,
            revision="local",  # use a commit or checksum of the weights in practice
        )
        disk = DiskKVCache(args.dir, serializer=TensorSerializer())
        print(f"L2 on disk: {args.dir} ({len(disk)} blocks from earlier runs)\n")
        # Memory in front of disk; every stored block is written through to disk.
        recorder = TraceRecorder() if args.trace_output is not None else None
        memory = MemoryKVCache(max_entries=args.memory_max_entries)
        if recorder is None:
            storage = TieredStorage(fast=memory, slow=disk)
        else:
            storage = TieredStorage(
                fast=TracingStorage(memory, recorder, name="memory"),
                slow=TracingStorage(disk, recorder, name="disk"),
                trace_recorder=recorder,
            )
        if args.async_store:
            write_behind = AsyncWriteBehindStorage(
                storage, max_pending=args.write_queue
            )
            storage = write_behind
        # Warm up first. nano-vLLM compiles several layers and its sampler with
        # torch.compile, and the first new sequence length or batch size
        # triggers a recompile. That costs hundreds of milliseconds, has nothing
        # to do with caching, and would otherwise land on the first measured
        # request. One request at a time (batch size 1, like the measured ones),
        # with a long prompt twice so both the cold and the prefix-cache
        # attention paths are compiled. Done before attach(), so L2 stays clean.
        for warmup in ["Hello", "warm up " * 650, "warm up " * 650]:
            llm.generate([warmup], SamplingParams(temperature=1e-6, max_tokens=2),
                         use_tqdm=False)
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
            args.trace_output.write_text(json.dumps({
                "format_version": 1,
                "source": "nanovllm",
                "events": recorder.to_list(),
            }, indent=2), encoding="utf-8")
            print(f"Trace written to {args.trace_output}")
    finally:
        try:
            if write_behind is not None:
                write_behind.close()
        finally:
            atexit.unregister(llm.exit)
            llm.exit()


if __name__ == "__main__":
    main()
