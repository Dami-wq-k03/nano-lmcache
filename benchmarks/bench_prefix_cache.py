"""Micro-benchmark for ChunkedPrefixCache store and lookup.

Not run in CI. Use it to compare numbers before and after a change, for
example when optimizing eviction:

    uv run python benchmarks/bench_prefix_cache.py
    uv run python benchmarks/bench_prefix_cache.py --chunks 100 1000 --capacity 64
    uv run python benchmarks/bench_prefix_cache.py --capacity 8192 \
        --churn-prompts 64 --churn-chunks 1024

Timings are wall-clock seconds for one store or lookup of a whole prompt,
taken as the best of several repeats to reduce noise. They are only
comparable on the same machine, so always record a baseline first.
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.memory import MemoryKVCache

CHUNK_SIZE = 16
DEFAULT_CHUNK_COUNTS = (10, 100, 1000)
DEFAULT_CAPACITIES: tuple[int | None, ...] = (None, 64)


@dataclass(frozen=True)
class BenchResult:
    """Timings and outcome for one (chunk_count, capacity) scenario."""

    chunk_count: int
    capacity: int | None
    store_seconds: float
    lookup_seconds: float
    matched_chunks: int


@dataclass(frozen=True)
class ChurnResult:
    """Outcome of storing many distinct prompts into one cache."""

    prompts: int
    chunks_per_prompt: int
    capacity: int | None
    seconds: float
    entries: int


def run_churn(
    prompts: int,
    chunks_per_prompt: int,
    capacity: int | None,
    repeats: int,
) -> ChurnResult:
    """Time storing `prompts` unrelated prompts into one bounded cache.

    Every prompt after the cache fills forces evictions, which is the
    workload that exercises victim selection. A single long prompt barely
    evicts at all, so `run_scenario` alone cannot show eviction cost.
    """
    best = float("inf")
    entries = 0

    for _ in range(repeats):
        storage = MemoryKVCache(max_entries=capacity)
        cache = ChunkedPrefixCache(
            chunk_size=CHUNK_SIZE, namespace="bench", storage=storage
        )
        payloads: list[object] = [f"kv-{i}" for i in range(chunks_per_prompt)]

        start = time.perf_counter()
        for prompt in range(prompts):
            # Distinct first token per prompt, so no prefix is shared.
            base = prompt * chunks_per_prompt * CHUNK_SIZE
            cache.store(
                list(range(base, base + chunks_per_prompt * CHUNK_SIZE)),
                payloads,
            )
        best = min(best, time.perf_counter() - start)

        entries = len(storage)

    return ChurnResult(
        prompts=prompts,
        chunks_per_prompt=chunks_per_prompt,
        capacity=capacity,
        seconds=best,
        entries=entries,
    )


def run_scenario(
    chunk_count: int,
    capacity: int | None,
    repeats: int,
) -> BenchResult:
    """Time store and lookup of one prompt of `chunk_count` full chunks."""
    token_ids = list(range(chunk_count * CHUNK_SIZE))
    payloads: list[object] = [f"kv-{i}" for i in range(chunk_count)]

    best_store = float("inf")
    best_lookup = float("inf")
    matched = 0

    for _ in range(repeats):
        # A fresh cache each repeat, so store always starts cold.
        cache = ChunkedPrefixCache(
            chunk_size=CHUNK_SIZE,
            namespace="bench",
            storage=MemoryKVCache(max_entries=capacity),
        )

        start = time.perf_counter()
        cache.store(token_ids, payloads)
        best_store = min(best_store, time.perf_counter() - start)

        start = time.perf_counter()
        matched = cache.lookup(token_ids).matched_chunks
        best_lookup = min(best_lookup, time.perf_counter() - start)

    return BenchResult(
        chunk_count=chunk_count,
        capacity=capacity,
        store_seconds=best_store,
        lookup_seconds=best_lookup,
        matched_chunks=matched,
    )


def run_benchmark(
    chunk_counts: Sequence[int] = DEFAULT_CHUNK_COUNTS,
    capacities: Sequence[int | None] = DEFAULT_CAPACITIES,
    repeats: int = 5,
) -> list[BenchResult]:
    """Run every (chunk_count, capacity) combination."""
    return [
        run_scenario(chunk_count, capacity, repeats)
        for chunk_count in chunk_counts
        for capacity in capacities
    ]


def format_table(results: Sequence[BenchResult]) -> str:
    """Render results as a fixed-width text table."""
    header = (
        f"{'chunks':>7} {'capacity':>10} {'store (ms)':>12} "
        f"{'lookup (ms)':>12} {'matched':>8}"
    )
    lines = [header, "-" * len(header)]

    for result in results:
        capacity = "unbounded" if result.capacity is None else str(result.capacity)
        lines.append(
            f"{result.chunk_count:>7} {capacity:>10} "
            f"{result.store_seconds * 1000:>12.3f} "
            f"{result.lookup_seconds * 1000:>12.3f} "
            f"{result.matched_chunks:>8}"
        )

    return "\n".join(lines)


def format_churn(result: ChurnResult) -> str:
    """Render one churn result as a single readable line."""
    capacity = "unbounded" if result.capacity is None else str(result.capacity)
    return (
        f"churn: {result.prompts} prompts x {result.chunks_per_prompt} chunks, "
        f"capacity {capacity}: {result.seconds * 1000:.3f} ms, "
        f"{result.entries} of {result.prompts * result.chunks_per_prompt} "
        "chunks kept"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNK_COUNTS)
    parser.add_argument(
        "--capacity",
        type=int,
        nargs="+",
        default=[c for c in DEFAULT_CAPACITIES if c is not None],
        help="entry limits to test; an unbounded run is always included",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--churn-prompts",
        type=int,
        default=2000,
        help="number of unrelated prompts in each churn run",
    )
    parser.add_argument(
        "--churn-chunks",
        type=int,
        default=8,
        help="chunks per churn prompt; increase to measure deep chains",
    )
    args = parser.parse_args()
    if args.churn_prompts <= 0 or args.churn_chunks <= 0:
        parser.error("--churn-prompts and --churn-chunks must be positive")

    capacities: list[int | None] = [None, *args.capacity]
    print(f"chunk_size={CHUNK_SIZE}, best of {args.repeats} repeats\n")
    print(format_table(run_benchmark(args.chunks, capacities, args.repeats)))

    print("\nEviction churn (many unrelated prompts):")
    for capacity in capacities:
        print(
            format_churn(
                run_churn(args.churn_prompts, args.churn_chunks, capacity, args.repeats)
            )
        )


if __name__ == "__main__":
    main()
