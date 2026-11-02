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
from nano_lmcache.visualization import load_trace_json, write_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a workload into an offline HTML trace viewer"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", choices=sorted(SYNTHETIC_SCENARIOS))
    source.add_argument("--freeinference-jsonl", type=Path)
    source.add_argument("--trace-json", type=Path, help="recorded engine Trace JSON")
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

    if args.trace_json is not None:
        try:
            payload = load_trace_json(args.trace_json)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        title = "nano-lmcache · Engine trace"
    elif args.scenario is not None:
        recorder = TraceRecorder()
        storage = _make_storage(args, recorder)
        report = WorkloadReplayRunner(
            chunk_size=args.chunk_size,
            namespace="synthetic",
            storage=storage,
            recorder=recorder,
        ).run(generate_synthetic(args.scenario))
        payload = report.to_dict()
        title = f"nano-lmcache · {args.scenario}"
    else:
        requests = iter_freeinference_jsonl(args.freeinference_jsonl, limit=args.limit)
        report = BlockTraceReplayRunner().run(requests)
        payload = report.to_dict()
        title = "nano-lmcache · FreeInference opaque block replay"

    output = write_html(args.output, payload, title=title)
    print(
        f"wrote {output} ({len(payload['events'])} events, "
        f"{len(payload['requests'])} requests)"
    )


def _make_storage(
    args: argparse.Namespace, recorder: TraceRecorder
) -> CacheStorage:
    backend: CacheStorage
    if args.storage == "memory":
        backend = MemoryKVCache(max_entries=args.max_entries)
    elif args.storage == "disk":
        backend = DiskKVCache(args.storage_dir, max_entries=args.max_entries)
    else:
        fast = TracingStorage(
            MemoryKVCache(max_entries=args.max_entries),
            recorder,
            name="memory",
        )
        slow = TracingStorage(
            DiskKVCache(args.storage_dir, max_entries=args.max_entries),
            recorder,
            name="disk",
        )
        return TieredStorage(
            fast=fast,
            slow=slow,
            trace_recorder=recorder,
        )
    return TracingStorage(backend, recorder, name=args.storage)


if __name__ == "__main__":
    main()
