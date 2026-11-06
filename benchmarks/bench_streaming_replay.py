"""Measure replay memory growth with fixed cache and detail budgets.

Run: uv run python benchmarks/bench_streaming_replay.py --counts 1000 10000 100000
This measures Python allocations, not RSS or real KV tensor memory.
"""

import argparse
import gc
import time
import tracemalloc
from collections.abc import Iterator

from nano_lmcache.datasets import WorkloadRequest
from nano_lmcache.observability import TraceRecorder
from nano_lmcache.replay import BlockTraceReplayRunner
from nano_lmcache.storage.memory import MemoryKVCache


def workload(count: int) -> Iterator[WorkloadRequest]:
    for index in range(count):
        yield WorkloadRequest(
            request_id=f"r{index:012d}", session_id="streaming-benchmark",
            arrival_time_ms=index, model_id="opaque-demo", prompt_tokens=48,
            prefix_blocks=(1, 2, index + 100), observed_cached_tokens=16,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 10_000])
    parser.add_argument("--max-entries", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument(
        "--batch-baseline", action="store_true",
        help="sort all input and retain all detail, matching the original replay path",
    )
    args = parser.parse_args()
    if any(count < 1 for count in args.counts):
        parser.error("counts must be positive")
    if args.max_entries < 1 or args.max_events < 0 or args.max_requests < 0:
        parser.error("cache capacity must be positive; detail budgets non-negative")
    print("requests\tpeak KiB\tseconds\tretained requests\tretained events\tevictions")
    for count in args.counts:
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        storage = MemoryKVCache(max_entries=args.max_entries)
        report = BlockTraceReplayRunner(
            storage=storage,
            recorder=TraceRecorder(
                max_events=None if args.batch_baseline else args.max_events,
                sample_every=1 if args.batch_baseline else 10,
            ),
        ).run(
            workload(count), sort_by_arrival=args.batch_baseline,
            max_requests=None if args.batch_baseline else args.max_requests,
        )
        peak = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        elapsed = time.perf_counter() - started
        assert report.total_chunks == count * 3
        assert report.matched_chunks == (count - 1) * min(2, args.max_entries)
        assert report.observed_cached_tokens == count * 16
        print(f"{count}\t{peak:.1f}\t{elapsed:.3f}\t{len(report.requests)}"
              f"\t{len(report.events)}\t{storage.stats.evictions}")


if __name__ == "__main__":
    main()
