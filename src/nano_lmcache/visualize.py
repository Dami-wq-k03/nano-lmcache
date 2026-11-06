"""Command-line entry point for offline replay visualizations."""

import argparse
import json
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from nano_lmcache.datasets import (
    SYNTHETIC_SCENARIOS,
    WorkloadRequest,
    generate_synthetic,
    iter_workload_jsonl,
)
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
    source.add_argument(
        "--workload-jsonl", type=Path, help="one WorkloadRequest per line",
    )
    source.add_argument("--trace-json", type=Path, help="recorded engine Trace JSON")
    parser.add_argument("--output", type=Path, default=Path("trace.html"))
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--stream", action="store_true",
        help="consume JSONL in (arrival_time_ms, request_id) order without sorting",
    )
    parser.add_argument(
        "--opaque-blocks", action="store_true",
        help="use prefix_blocks rather than token_ids in --workload-jsonl",
    )
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=10_000)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument(
        "--summary-json", type=Path,
        help="write exact aggregate statistics and retention/configuration metadata",
    )
    parser.add_argument(
        "--storage",
        choices=("memory", "disk", "tiered"),
        default="memory",
        help="storage backend for token or opaque-block replay",
    )
    parser.add_argument("--storage-dir", type=Path, default=Path(".trace-cache"))
    parser.add_argument("--max-entries", type=int, default=10_000)
    args = parser.parse_args()

    if args.stream and not (args.workload_jsonl or args.freeinference_jsonl):
        parser.error("--stream requires --workload-jsonl or --freeinference-jsonl")
    if args.opaque_blocks and args.workload_jsonl is None:
        parser.error("--opaque-blocks requires --workload-jsonl")
    if args.trace_json is not None and args.limit is not None:
        parser.error("--limit applies to workload replay, not --trace-json")
    for name in ("max_events", "max_requests", "limit"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("max_entries", "sample_every", "chunk_size", "block_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    inputs = [p.resolve() for p in (
        args.workload_jsonl, args.freeinference_jsonl, args.trace_json,
    ) if p is not None]
    outputs = [p.resolve() for p in (args.output, args.summary_json) if p is not None]
    if len(set(outputs)) != len(outputs) or set(inputs) & set(outputs):
        parser.error("input, HTML output, and summary output paths must be distinct")

    try:
        if args.trace_json is not None:
            payload = load_trace_json(args.trace_json)
            events = payload["events"]
            requests = payload["requests"]
            payload["events"] = events[
                :args.max_events * args.sample_every:args.sample_every
            ]
            payload["requests"] = requests[:args.max_requests]
            payload["retention"] = {
                "max_events": args.max_events, "max_requests": args.max_requests,
                "sample_every": args.sample_every,
                "events_total": len(events),
                "events_retained": len(payload["events"]),
                "events_omitted": len(events) - len(payload["events"]),
                "requests_retained": len(payload["requests"]),
                "requests_omitted": len(requests) - len(payload["requests"]),
            }
            payload["configuration"] = {"ordering": "recorded", "input": "batch-json"}
            title = "nano-lmcache · Engine trace"
        else:
            recorder = TraceRecorder(
                max_events=args.max_events, sample_every=args.sample_every,
            )
            storage = _make_storage(args, recorder)
            opaque = args.freeinference_jsonl is not None or args.opaque_blocks
            workload: Iterable[WorkloadRequest]
            if args.scenario is not None:
                workload = islice(generate_synthetic(args.scenario), args.limit)
                title = f"nano-lmcache · {args.scenario}"
            elif args.workload_jsonl is not None:
                workload = iter_workload_jsonl(args.workload_jsonl, limit=args.limit)
                title = "nano-lmcache · Workload replay"
            else:
                workload = iter_freeinference_jsonl(
                    args.freeinference_jsonl, limit=args.limit,
                )
                title = "nano-lmcache · FreeInference opaque block replay"
            runner = (
                BlockTraceReplayRunner(
                    block_tokens=args.block_tokens, storage=storage, recorder=recorder,
                ) if opaque else WorkloadReplayRunner(
                    chunk_size=args.chunk_size, namespace="replay",
                    storage=storage, recorder=recorder,
                )
            )
            report = runner.run(
                workload, sort_by_arrival=not args.stream,
                max_requests=args.max_requests,
            )
            payload = report.to_dict()
            payload["configuration"].update({
                "storage": args.storage, "max_entries": args.max_entries,
                "chunk_size": args.block_tokens if opaque else args.chunk_size,
            })

        output = write_html(args.output, payload, title=title)
        if args.summary_json is not None:
            args.summary_json.write_text(json.dumps({
                key: payload[key]
                for key in ("source", "summary", "retention", "configuration")
            }, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {output} ({payload['summary']['requests']} requests; "
        f"displaying {len(payload['requests'])} requests and "
        f"{len(payload['events'])}/{payload['retention']['events_total']} events)"
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
