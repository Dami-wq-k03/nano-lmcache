"""Time to first token with nano-vLLM: no cache, L1, L2 in CPU memory, L2 on disk.

Roadmap task N4. Needs Linux, an NVIDIA GPU, and the `nanovllm` dependency
group; not run in CI:

    uv sync --extra hf --group nanovllm
    CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/bench_nanovllm.py \\
        --model /path/to/Qwen3-0.6B
    # more sizes, more repeats, and a JSON copy of the results:
    uv run python benchmarks/bench_nanovllm.py --model ... \\
        --blocks 1 2 4 8 16 32 --repeats 7 --json results.json

For each prompt size (a number of reusable 256-token blocks plus a short
tail), one engine measures the time to first token of the same prompt in
four setups:

- no cache: plain nano-vLLM with an empty L1 (`detach`), so the whole
  prompt is computed;
- L1: nano-vLLM's own GPU prefix cache already holds the blocks;
- L2, CPU memory: L1 is empty, the blocks are restored from
  `CPUOffloadKVCache` (pinned CPU memory);
- L2, disk: L1 is empty, the blocks are restored from `DiskKVCache` after
  reopening it (as after a restart) and dropping the files from the OS page
  cache, so they are really read from disk. A second disk column keeps the
  page cache, which is what you see right after writing.

It also records the first, storing request for each L2 backend, because
version 1 stores synchronously, inside the request.

The setups are interleaved, one of each per round, and medians reported.
Loading is also timed on its own (fetch from L2, copy into the GPU KV
cache), because inside a request it overlaps with kernel launches.
Recomputing one block is derived as `(no cache - L1) / blocks`. The
crossover is the smallest size from which restoring from L2 is at least
`--min-gain` (default 10%) faster than computing the whole prompt, at that
size and every larger one measured. See docs/nanovllm-benchmark.md.
"""

import argparse
import atexit
import json
import os
import random
import shutil
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BLOCK_SIZE = 256
DEFAULT_BLOCKS = [1, 2, 4, 8, 16, 24, 32]
DEFAULT_MIN_GAIN = 0.10


@dataclass(frozen=True)
class Row:
    """Median times, in milliseconds, for one prompt size."""

    blocks: int
    prompt_tokens: int
    reused_tokens: int
    no_cache_ms: float
    l1_ms: float
    l2_memory_ms: float
    l2_disk_ms: float
    l2_disk_page_cache_ms: float
    # The first request with each L2 backend attached: compute the whole
    # prompt, then store its blocks before the request returns.
    store_memory_ms: float
    store_disk_ms: float
    # Fetching all reused blocks from L2 and copying them into the GPU KV
    # cache, timed on its own, outside generation.
    load_memory_ms: float
    load_disk_ms: float
    load_disk_page_cache_ms: float


_TTFT_COLUMNS = (
    "no_cache_ms", "l1_ms", "l2_memory_ms", "l2_disk_ms",
    "l2_disk_page_cache_ms", "store_memory_ms", "store_disk_ms",
)

_COMPARED = {
    "l1_ms", "l2_memory_ms", "l2_disk_ms", "l2_disk_page_cache_ms",
}


def make_prompt(blocks: int, tail: int, vocab_size: int, seed: int) -> list[int]:
    """Random token IDs: `blocks` reusable blocks, then a `tail`.

    nano-vLLM never reuses a prompt's last block, so the tail (shorter than
    a block) is always computed and all `blocks` blocks can be reused. A
    different seed per size keeps prompts of different sizes from sharing
    a prefix. Token values do not change the cost of computing them.
    """
    rng = random.Random(seed)
    return [rng.randrange(1, vocab_size) for _ in range(blocks * BLOCK_SIZE + tail)]


def crossover(
    rows: Sequence[Row], column: str, *, min_gain: float = DEFAULT_MIN_GAIN
) -> int | None:
    """Smallest block count from which `column` is clearly faster than no cache.

    "Clearly" means by at least `min_gain` (a fraction of the no-cache time):
    run to run, the same setup varies by a few milliseconds, so a smaller
    win says nothing. "From which" means at that size and every larger one
    measured, so a single lucky win at a small size does not count either.
    """
    if column not in _COMPARED:
        raise ValueError(f"unknown column {column!r}; use one of {sorted(_COMPARED)}")
    if not 0 <= min_gain < 1:
        raise ValueError("min_gain must be a fraction in [0, 1)")

    found: int | None = None
    for row in sorted(rows, key=lambda r: r.blocks, reverse=True):
        if getattr(row, column) <= row.no_cache_ms * (1 - min_gain):
            found = row.blocks
        else:
            break
    return found


def per_block_costs(row: Row) -> dict[str, float]:
    """Milliseconds per reused block.

    Recomputing is derived: whatever the no-cache request takes above the
    L1 request, spread over the reused blocks. While a prefill step is
    dominated by fixed overhead (see the write-up), this is close to zero
    and noisy. Loading is measured directly.
    """
    return {
        "recompute_ms": (row.no_cache_ms - row.l1_ms) / row.blocks,
        "load_memory_ms": row.load_memory_ms / row.blocks,
        "load_disk_ms": row.load_disk_ms / row.blocks,
        "load_disk_page_cache_ms": row.load_disk_page_cache_ms / row.blocks,
    }


def _table(header: Sequence[str], lines: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(" ---: " for _ in header) + "|"]
    out += ["| " + " | ".join(cells) + " |" for cells in lines]
    return "\n".join(out)


def format_ttft_table(rows: Sequence[Row]) -> str:
    """Median time to first token per setup, as a Markdown table."""
    return _table(
        ["blocks", "prompt tokens", "no cache", "L1", "L2 memory", "L2 disk",
         "L2 disk (page cache)", "store, memory", "store, disk"],
        [[str(r.blocks), str(r.prompt_tokens)]
         + [f"{getattr(r, name):.1f}" for name in _TTFT_COLUMNS]
         for r in rows],
    )


def format_cost_table(rows: Sequence[Row]) -> str:
    """Per-block recompute and load costs, as a Markdown table."""
    lines = []
    for r in rows:
        costs = per_block_costs(r)
        lines.append([str(r.blocks), f"{costs['recompute_ms']:.2f}",
                      f"{costs['load_memory_ms']:.2f}",
                      f"{costs['load_disk_ms']:.2f}",
                      f"{costs['load_disk_page_cache_ms']:.2f}"])
    return _table(["blocks", "recompute / block", "load from memory / block",
                   "load from disk / block", "load from disk (page cache) / block"],
                  lines)


def format_summary(rows: Sequence[Row], min_gain: float = DEFAULT_MIN_GAIN) -> str:
    """One line per L2 backend saying where it starts to pay off."""
    largest = max(r.blocks for r in rows)
    out = []
    for column, name in (("l2_memory_ms", "L2 in CPU memory"),
                         ("l2_disk_ms", "L2 on disk (page cache dropped)"),
                         ("l2_disk_page_cache_ms", "L2 on disk (page cache warm)")):
        found = crossover(rows, column, min_gain=min_gain)
        if found is None:
            out.append(f"{name}: never at least {min_gain:.0%} faster than "
                       f"recomputing, up to {largest} blocks "
                       f"({largest * BLOCK_SIZE} tokens).")
        else:
            out.append(f"{name}: at least {min_gain:.0%} faster than recomputing "
                       f"from {found} blocks ({found * BLOCK_SIZE} reused tokens) on.")
    return "\n".join(out)


def drop_page_cache(directory: Path) -> int:
    """Ask the kernel to forget cached pages of every file under `directory`.

    Needs no root: `POSIX_FADV_DONTNEED` drops clean pages of a file the
    caller can open, and `DiskKVCache` fsyncs every entry, so its pages are
    clean. Returns the number of files.
    """
    count = 0
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        count += 1
    return count


# ---- the GPU part ------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> tuple[list[Row], dict[str, Any]]:
    """Measure every size in `args.blocks`; returns rows and an environment note."""
    import torch
    from nanovllm import LLM, SamplingParams

    from nano_lmcache.integrations.nanovllm import (
        KVBlockLayout,
        NanoVLLMExternalCache,
    )
    from nano_lmcache.integrations.nanovllm.engine import attach, detach
    from nano_lmcache.integrations.nanovllm.kv_blocks import import_block
    from nano_lmcache.storage.disk import DiskKVCache
    from nano_lmcache.tensor.offload import CPUOffloadKVCache
    from nano_lmcache.tensor.serializers import TensorSerializer

    longest = max(args.blocks) * BLOCK_SIZE + args.tail
    max_model_len = (longest // BLOCK_SIZE + 2) * BLOCK_SIZE
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_model_len,
              max_num_batched_tokens=max(16384, max_model_len))
    hf_config = llm.model_runner.config.hf_config
    layout = KVBlockLayout.from_hf_config(
        hf_config, model=Path(args.model).name, revision="benchmark"
    )
    params = SamplingParams(temperature=1e-6, max_tokens=1)

    def ttft(token_ids: list[int]) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.manual_seed(0)
        llm.generate([token_ids], params, use_tqdm=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    # nano-vLLM compiles several layers and its sampler with torch.compile;
    # the first new sequence lengths trigger recompiles that cost hundreds
    # of milliseconds. Warm up the cold and the prefix-cache attention path
    # at a few lengths before measuring anything.
    detach(llm)
    for blocks in (1, max(args.blocks) // 2 or 1, max(args.blocks)):
        warm = make_prompt(blocks, args.tail, hf_config.vocab_size, seed=-blocks)
        ttft(warm)
        ttft(warm)

    root = Path(args.dir or tempfile.mkdtemp(prefix="nano-lmcache-bench-"))

    def measure(blocks: int) -> Row:
        """Median times for one prompt size.

        The setups are interleaved: each round measures every setup once,
        so slow drift (clocks, other processes) affects them all alike
        instead of whichever setup happened to run during it.
        """
        token_ids = make_prompt(blocks, args.tail, hf_config.vocab_size, seed=blocks)
        directory = root / f"blocks-{blocks}"
        serializer = TensorSerializer()

        # First requests with each L2 backend: compute and store.
        memory = NanoVLLMExternalCache(layout, CPUOffloadKVCache(device="cuda"))
        attach(llm, memory)
        store_memory_ms = ttft(token_ids)
        attach(llm, NanoVLLMExternalCache(
            layout, DiskKVCache(directory, serializer=serializer)))
        store_disk_ms = ttft(token_ids)

        def restore(external: Any) -> float:
            manager = attach(llm, external)  # a fresh, empty L1
            elapsed = ttft(token_ids)
            if manager.stats.l2_hit_blocks != blocks:
                raise RuntimeError(
                    f"expected {blocks} L2 hits, got {manager.stats.l2_hit_blocks}"
                )
            return elapsed

        def reopened_disk(drop: bool) -> Any:
            disk = DiskKVCache(directory, serializer=serializer)  # a restart
            if drop:
                drop_page_cache(directory)
            return NanoVLLMExternalCache(layout, disk)

        def load(external: Any) -> float:
            """Fetch every reused block and copy it into the GPU, alone."""
            torch.cuda.synchronize()
            start = time.perf_counter()
            payloads = external.fetch(token_ids, 0, blocks)
            for block_id, payload in enumerate(payloads):
                import_block(llm.model_runner.kv_cache, block_id, payload)
            torch.cuda.synchronize()
            if len(payloads) != blocks:
                raise RuntimeError(f"expected {blocks} blocks, got {len(payloads)}")
            return (time.perf_counter() - start) * 1000

        samples: dict[str, list[float]] = {name: [] for name in (
            "no_cache", "l1", "l2_memory", "l2_disk", "l2_disk_page_cache",
            "load_memory", "load_disk", "load_disk_page_cache")}
        for _ in range(args.repeats):
            detach(llm)
            samples["no_cache"].append(ttft(token_ids))
            samples["l1"].append(ttft(token_ids))  # L1 now holds the prompt
            samples["l2_memory"].append(restore(memory))
            samples["l2_disk"].append(restore(reopened_disk(drop=True)))
            samples["l2_disk_page_cache"].append(restore(reopened_disk(drop=False)))
            # Direct loads last: they overwrite GPU blocks, and the next
            # round starts from an empty L1 anyway.
            detach(llm)
            samples["load_memory"].append(load(memory))
            samples["load_disk"].append(load(reopened_disk(drop=True)))
            samples["load_disk_page_cache"].append(load(reopened_disk(drop=False)))
        shutil.rmtree(directory)

        medians = {name: statistics.median(v) for name, v in samples.items()}
        return Row(
            blocks=blocks, prompt_tokens=len(token_ids),
            reused_tokens=blocks * BLOCK_SIZE,
            no_cache_ms=medians["no_cache"], l1_ms=medians["l1"],
            l2_memory_ms=medians["l2_memory"], l2_disk_ms=medians["l2_disk"],
            l2_disk_page_cache_ms=medians["l2_disk_page_cache"],
            store_memory_ms=store_memory_ms, store_disk_ms=store_disk_ms,
            load_memory_ms=medians["load_memory"],
            load_disk_ms=medians["load_disk"],
            load_disk_page_cache_ms=medians["load_disk_page_cache"],
        )

    rows = []
    for blocks in sorted(args.blocks):
        rows.append(measure(blocks))
        print(f"  measured {blocks} blocks", flush=True)

    if args.dir is None:
        shutil.rmtree(root, ignore_errors=True)
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model": Path(args.model).name,
        "dtype": layout.dtype,
        "block_mib": layout.chunk_nbytes / 2**20,
        "repeats": args.repeats,
        "tail_tokens": args.tail,
        "min_gain": args.min_gain,
    }
    # nano-vLLM also registers exit() with atexit; it must not run twice.
    atexit.unregister(llm.exit)
    llm.exit()
    return rows, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--blocks", type=int, nargs="+", default=DEFAULT_BLOCKS)
    parser.add_argument("--tail", type=int, default=32,
                        help="computed tokens after the reusable blocks (< 256)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--min-gain", type=float, default=DEFAULT_MIN_GAIN,
        help="how much faster than no cache counts as a win (default 0.10)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--dir", help="where to put the disk tier (default: a temp dir)"
    )
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args()
    if not 0 < args.tail < BLOCK_SIZE:
        parser.error("--tail must be between 1 and 255")
    if min(args.blocks) <= 0 or args.repeats <= 0:
        parser.error("--blocks and --repeats must be positive")

    rows, environment = run_benchmark(args)
    print(f"\n{environment['model']} ({environment['dtype']}, "
          f"{environment['block_mib']:.1f} MiB per block) on {environment['gpu']}, "
          f"median of {args.repeats}, times in ms\n")
    print("Time to first token:\n")
    print(format_ttft_table(rows))
    print("\nPer reused block (recompute derived, load measured directly):\n")
    print(format_cost_table(rows))
    print()
    print(format_summary(rows, args.min_gain))
    if args.json:
        args.json.write_text(json.dumps(
            {"environment": environment, "rows": [asdict(r) for r in rows]},
            indent=2,
        ))


if __name__ == "__main__":
    main()
