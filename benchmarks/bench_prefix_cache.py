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
Every invocation also runs a fixed deep-chain churn case: 64 prompts of
1024 chunks against an 8192-entry cache, forcing repeated leaf selection
through mostly internal nodes in the original linear scan.
"""

import argparse
import random
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
class PolicyWorkloadRequest:
    """One deterministic shared-system/user request."""

    token_ids: tuple[int, ...]
    chunk_values: tuple[object, ...]


@dataclass(frozen=True)
class PolicyWorkloadResult:
    """Policy comparison metrics for one workload and policy."""

    workload: str
    policy: str
    seed: int
    capacity: int
    requests: int
    hit_rate: float
    reused_tokens: int
    eviction_count: int
    occupancy: int
    lookup_seconds: float
    store_seconds: float


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


def run_deep_chain_churn(repeats: int = 5) -> ChurnResult:
    """Exercise victim selection with deep chains and a large working set.

    Eight chains fill the cache; the remaining prompts force eviction.
    Keep this case fixed so it can be compared across implementations,
    independently of the caller's short-chain churn parameters.
    """
    return run_churn(
        prompts=64, chunks_per_prompt=1024, capacity=8192, repeats=repeats
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


def build_policy_workload(
    workload: str,
    *,
    requests: int = 1_000,
    seed: int = 2026,
    system_families: int = 8,
    users_per_family: int = 16,
) -> tuple[PolicyWorkloadRequest, ...]:
    """Build a repeatable shared-system/user request sequence.

    ``uniform`` samples families and users evenly.  ``zipf`` samples a small
    set of popular request pairs with a long tail.  Both use the same prompt
    shape: two chunks of family-specific system tokens followed by two chunks
    of user-specific tokens.
    """
    if workload not in {"uniform", "zipf"}:
        raise ValueError("workload must be 'uniform' or 'zipf'")
    if requests <= 0 or system_families <= 0 or users_per_family <= 0:
        raise ValueError("workload sizes must be positive")

    rng = random.Random(seed)
    catalog = [
        (family, user)
        for family in range(system_families)
        for user in range(users_per_family)
    ]
    if workload == "uniform":
        choices = [rng.choice(catalog) for _ in range(requests)]
    else:
        # Rank is fixed before sampling, so changing policy cannot change the
        # sequence.  The exponent gives a visible hot head and long tail.
        weights = [1.0 / (rank + 1) ** 1.2 for rank in range(len(catalog))]
        choices = rng.choices(catalog, weights=weights, k=requests)

    result: list[PolicyWorkloadRequest] = []
    for family, user in choices:
        system_base = family * 1_000_000
        user_base = system_base + 100_000 + user * 1_000
        token_ids = tuple(
            [system_base + offset for offset in range(2 * CHUNK_SIZE)]
            + [user_base + offset for offset in range(2 * CHUNK_SIZE)]
        )
        chunk_values = tuple(
            f"system-{family}/user-{user}/chunk-{chunk}"
            for chunk in range(4)
        )
        result.append(
            PolicyWorkloadRequest(
                token_ids=token_ids,
                chunk_values=chunk_values,
            )
        )
    return tuple(result)


def run_policy_workload(
    workload: str,
    *,
    policy: str,
    requests: int = 1_000,
    seed: int = 2026,
    capacity: int = 48,
    repeats: int = 3,
) -> PolicyWorkloadResult:
    """Run one policy over a fixed workload and report cache metrics.

    Every repeat creates a fresh cache but consumes the exact same prebuilt
    sequence.  Timing uses the best total per operation; hit, eviction, and
    occupancy values are deterministic and are never selected by timing.
    """
    if policy not in {"lru", "lfu"}:
        raise ValueError("policy must be 'lru' or 'lfu'")
    if capacity <= 0 or repeats <= 0:
        raise ValueError("capacity and repeats must be positive")

    sequence = build_policy_workload(
        workload, requests=requests, seed=seed
    )
    best_lookup = float("inf")
    best_store = float("inf")
    final_cache: MemoryKVCache | None = None

    for _ in range(repeats):
        storage = MemoryKVCache(
            max_entries=capacity,
            eviction_policy=policy,
        )
        cache = ChunkedPrefixCache(
            chunk_size=CHUNK_SIZE,
            namespace=f"policy-benchmark-{workload}",
            storage=storage,
        )
        lookup_total = 0.0
        store_total = 0.0
        for request in sequence:
            lookup_start = time.perf_counter()
            hit = cache.lookup(request.token_ids)
            lookup_total += time.perf_counter() - lookup_start

            store_start = time.perf_counter()
            cache.store(
                request.token_ids,
                request.chunk_values[hit.matched_chunks :],
                start_chunk=hit.matched_chunks,
            )
            store_total += time.perf_counter() - store_start
        best_lookup = min(best_lookup, lookup_total)
        best_store = min(best_store, store_total)
        final_cache = storage

    assert final_cache is not None
    return PolicyWorkloadResult(
        workload=workload,
        policy=policy,
        seed=seed,
        capacity=capacity,
        requests=requests,
        hit_rate=cache.metrics.hit_rate,
        reused_tokens=cache.metrics.hit_tokens,
        eviction_count=final_cache.stats.evictions,
        occupancy=len(final_cache),
        lookup_seconds=best_lookup / requests,
        store_seconds=best_store / requests,
    )


def run_policy_benchmark(
    *,
    requests: int = 1_000,
    seed: int = 2026,
    capacity: int = 48,
    repeats: int = 3,
) -> list[PolicyWorkloadResult]:
    """Compare LRU and LFU on uniform and skewed sequences."""
    return [
        run_policy_workload(
            workload,
            policy=policy,
            requests=requests,
            seed=seed,
            capacity=capacity,
            repeats=repeats,
        )
        for workload in ("uniform", "zipf")
        for policy in ("lru", "lfu")
    ]


def format_policy_table(results: Sequence[PolicyWorkloadResult]) -> str:
    """Render policy metrics, including average lookup/store latency."""
    header = (
        f"{'workload':>9} {'policy':>6} {'hit rate':>9} "
        f"{'reused':>8} {'evictions':>10} {'occupancy':>10} "
        f"{'lookup (us)':>12} {'store (us)':>11}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.workload:>9} {result.policy:>6} "
            f"{result.hit_rate:>9.3%} {result.reused_tokens:>8} "
            f"{result.eviction_count:>10} {result.occupancy:>10} "
            f"{result.lookup_seconds * 1_000_000:>12.2f} "
            f"{result.store_seconds * 1_000_000:>11.2f}"
        )
    return "\n".join(lines)


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
    parser.add_argument(
        "--policy-requests",
        type=int,
        default=1_000,
        help="requests per LRU/LFU policy workload",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=2026,
        help="fixed random seed shared by both policy workloads",
    )
    parser.add_argument(
        "--policy-capacity",
        type=int,
        default=48,
        help="entry capacity for the policy comparison",
    )
    parser.add_argument(
        "--policy-repeats",
        type=int,
        default=3,
        help="timing repeats for each policy workload",
    )
    args = parser.parse_args()
    if args.churn_prompts <= 0 or args.churn_chunks <= 0:
        parser.error("--churn-prompts and --churn-chunks must be positive")
    if (
        args.policy_requests <= 0
        or args.policy_capacity <= 0
        or args.policy_repeats <= 0
    ):
        parser.error("policy benchmark arguments must be positive")

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

    print("\nDeep-chain eviction churn (large capacity):")
    print(format_churn(run_deep_chain_churn(args.repeats)))

    print("\nLRU versus LFU shared-prefix workloads:")
    print(
        format_policy_table(
            run_policy_benchmark(
                requests=args.policy_requests,
                seed=args.policy_seed,
                capacity=args.policy_capacity,
                repeats=args.policy_repeats,
            )
        )
    )


if __name__ == "__main__":
    main()
