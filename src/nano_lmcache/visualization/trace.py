"""Read metadata-only engine traces without importing torch or nano-vLLM."""

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nano_lmcache.observability.events import TraceEvent


def report_from_events(
    events: Iterable[TraceEvent], *, source: str = "trace"
) -> dict[str, Any]:
    """Normalize recorded events into the offline visualizer's report format.

    Only committed ``l1_hit`` and ``l2_hit`` events count toward reuse.
    ``l2_lookup`` is a repeatable scheduling probe, not a successful restore.
    TTFT is optional, provided by a measured one-token generation in the demo;
    CUDA copy events never stand in for elapsed inference time.
    """
    ordered = list(events)
    grouped: dict[str, list[TraceEvent]] = {}
    for event in ordered:
        grouped.setdefault(event.request_id, []).append(event)

    requests: list[dict[str, Any]] = []
    for request_id, request_events in grouped.items():
        started = next(
            (e.metadata or {} for e in request_events
             if e.event_type == "request_started"),
            {},
        )
        block_size = _nonnegative_int(started.get("block_size", 0), "block_size")
        prompt_tokens = _nonnegative_int(
            started.get("prompt_tokens", 0), "prompt_tokens"
        )
        total = (
            (prompt_tokens + block_size - 1) // block_size if block_size else 0
        )
        reused: dict[int, str] = {}
        for event in request_events:
            if event.event_type in ("l1_hit", "l2_hit"):
                if event.chunk_index is not None:
                    reused[event.chunk_index] = event.event_type
        # Raw storage-only recordings may have no request_started event.
        if not total and reused:
            total = max(reused) + 1
        statuses = ["COMPUTE"] * total
        for index, event_type in reused.items():
            if index < total:
                statuses[index] = "L1_HIT" if event_type == "l1_hit" else "L2_HIT"
        l1_hits = statuses.count("L1_HIT")
        l2_hits = statuses.count("L2_HIT")
        matched = l1_hits + l2_hits
        ttft_ms: float | None = None
        for event in request_events:
            if event.event_type == "request_completed":
                measured = (event.metadata or {}).get("ttft_ms")
                if measured is not None:
                    if (
                        type(measured) not in (int, float)
                        or not math.isfinite(measured)
                        or measured < 0
                    ):
                        raise ValueError("ttft_ms must be finite and non-negative")
                    ttft_ms = float(measured)
        requests.append({
            "request_id": request_id,
            "source": source,
            "model_id": started.get("model_id"),
            "chunk_size": block_size,
            "prompt_tokens": prompt_tokens,
            "total_chunks": total,
            "matched_chunks": matched,
            "matched_tokens": matched * block_size,
            "statuses": statuses,
            "l1_hit_blocks": l1_hits,
            "l2_hit_blocks": l2_hits,
            "ttft_ms": ttft_ms,
            "computed_tokens": sum(
                _nonnegative_int(
                    (e.metadata or {}).get("computed_tokens", 0), "computed_tokens"
                )
                for e in request_events if e.event_type == "prefill_step"
            ),
        })

    total_chunks = sum(r["total_chunks"] for r in requests)
    matched_chunks = sum(r["matched_chunks"] for r in requests)
    return {
        "source": source,
        "summary": {
            "requests": len(requests),
            "total_chunks": total_chunks,
            "matched_chunks": matched_chunks,
            "hit_rate": matched_chunks / total_chunks if total_chunks else 0.0,
            "l1_hit_blocks": sum(r["l1_hit_blocks"] for r in requests),
            "l2_hit_blocks": sum(r["l2_hit_blocks"] for r in requests),
        },
        "requests": requests,
        "events": [event.to_dict() for event in ordered],
    }


def load_trace_json(path: str | Path) -> dict[str, Any]:
    """Load a recorder event array or a version-1 envelope containing events."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    source = "trace"
    if isinstance(data, dict):
        if type(data.get("format_version", 1)) is not int or data.get(
            "format_version", 1
        ) != 1:
            raise ValueError("unsupported trace format_version (expected 1)")
        source = data.get("source", "trace")
        if not isinstance(source, str):
            raise ValueError("trace source must be a string")
        data = data.get("events")
    if not isinstance(data, list):
        raise ValueError("trace must be an event array or an object containing events")

    events = []
    previous_sequence = -1
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"event {index} must be an object")
        try:
            event = TraceEvent(**item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid event {index}: {exc}") from exc
        if event.sequence <= previous_sequence:
            raise ValueError("event sequences must be strictly increasing")
        previous_sequence = event.sequence
        events.append(event)
    return report_from_events(events, source=source)


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
