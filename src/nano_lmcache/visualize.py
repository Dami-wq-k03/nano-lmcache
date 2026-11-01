"""Command-line entry point for offline replay visualizations."""

import argparse
from pathlib import Path

from nano_lmcache.datasets import SYNTHETIC_SCENARIOS, generate_synthetic
from nano_lmcache.datasets.adapters.freeinference import iter_freeinference_jsonl
from nano_lmcache.observability import TraceRecorder, TracingStorage
from nano_lmcache.replay import BlockTraceReplayRunner, WorkloadReplayRunner
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.tiered import TieredStorage
from nano_lmcache.visualization import write_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a workload into an offline HTML trace viewer"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", choices=sorted(SYNTHETIC_SCENARIOS))
    source.add_argument("--freeinference-jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("trace.html"))
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--storage",
        choices=("memory", "disk", "tiered"),
        default="memory",
        help="storage backend for synthetic token replay",
    )
    parser.add_argument("--storage-dir", type=Path, default=Path(".trace-cache"))
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    if args.scenario is not None:
        recorder = TraceRecorder()
        storage = _make_storage(args, recorder)
        report = WorkloadReplayRunner(
            chunk_size=args.chunk_size,
            namespace="synthetic",
            storage=storage,
            recorder=recorder,
        ).run(generate_synthetic(args.scenario))
        title = f"nano-lmcache · {args.scenario}"
    else:
        requests = iter_freeinference_jsonl(args.freeinference_jsonl, limit=args.limit)
        report = BlockTraceReplayRunner().run(requests)
        title = "nano-lmcache · FreeInference opaque block replay"

    output = write_html(args.output, report, title=title)
    print(
        f"wrote {output} ({len(report.events)} events, "
        f"{len(report.requests)} requests)"
    )


def _make_storage(args: argparse.Namespace, recorder: TraceRecorder) -> TracingStorage:
    backend: CacheStorage
    if args.storage == "memory":
        backend = MemoryKVCache(max_entries=args.max_entries)
    elif args.storage == "disk":
        backend = DiskKVCache(args.storage_dir, max_entries=args.max_entries)
    else:
        fast = MemoryKVCache(max_entries=args.max_entries)
        slow = DiskKVCache(args.storage_dir, max_entries=args.max_entries)
        backend = TieredStorage(fast=fast, slow=slow)
    return TracingStorage(backend, recorder, name=args.storage)


if __name__ == "__main__":
    main()
