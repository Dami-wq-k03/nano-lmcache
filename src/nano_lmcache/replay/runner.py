"""Replay engines for token workloads and opaque block traces."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.datasets.schemas.workload import WorkloadRequest
from nano_lmcache.observability.context import request_context
from nano_lmcache.observability.events import TraceEvent
from nano_lmcache.observability.recorder import TraceRecorder
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage


@dataclass(frozen=True, slots=True)
class ReplayRequestResult:
    """Outcome of one replayed request."""

    request_id: str
    session_id: str
    model_id: str
    source: str
    matched_chunks: int
    total_chunks: int
    matched_tokens: int
    chunk_size: int
    statuses: tuple[str, ...]
    observed_cached_tokens: int | None = None

    @property
    def hit_rate(self) -> float:
        return (
            self.matched_chunks / self.total_chunks if self.total_chunks else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["statuses"] = list(self.statuses)
        value["hit_rate"] = self.hit_rate
        return value


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Exact online totals; independent of retained request/event detail."""

    requests: int = 0
    total_chunks: int = 0
    matched_chunks: int = 0
    matched_tokens: int = 0
    prompt_tokens: int = 0
    observed_cached_tokens: int = 0
    observed_requests: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    storage_hits: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Requests, trace events, and aggregate counters from one replay."""

    requests: tuple[ReplayRequestResult, ...]
    events: tuple[TraceEvent, ...]
    source: str
    chunk_size: int
    aggregate: ReplaySummary | None = None
    retention: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        if self.aggregate is not None:
            return self.aggregate.total_chunks
        return sum(item.total_chunks for item in self.requests)

    @property
    def matched_chunks(self) -> int:
        if self.aggregate is not None:
            return self.aggregate.matched_chunks
        return sum(item.matched_chunks for item in self.requests)

    @property
    def hit_rate(self) -> float:
        return self.matched_chunks / self.total_chunks if self.total_chunks else 0.0

    @property
    def observed_cached_tokens(self) -> int:
        if self.aggregate is not None:
            return self.aggregate.observed_cached_tokens
        return sum(
            item.observed_cached_tokens or 0 for item in self.requests
        )

    def to_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = (
            asdict(self.aggregate) if self.aggregate is not None else {
                "requests": len(self.requests),
                "total_chunks": self.total_chunks,
                "matched_chunks": self.matched_chunks,
                "observed_cached_tokens": self.observed_cached_tokens,
            }
        )
        summary["hit_rate"] = self.hit_rate
        summary["miss_chunks"] = self.total_chunks - self.matched_chunks
        return {
            "source": self.source,
            "chunk_size": self.chunk_size,
            "summary": summary,
            "retention": dict(self.retention),
            "configuration": dict(self.configuration),
            "requests": [item.to_dict() for item in self.requests],
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _ordered_requests(
    requests: Iterable[WorkloadRequest], sort_by_arrival: bool,
) -> Iterator[WorkloadRequest]:
    if sort_by_arrival:
        yield from sorted(requests, key=lambda r: (r.arrival_time_ms, r.request_id))
        return
    previous: tuple[int, str] | None = None
    for request in requests:
        current = (request.arrival_time_ms, request.request_id)
        if previous is not None and current < previous:
            raise ValueError(
                "streaming requests must be in nondecreasing order of "
                "(arrival_time_ms, request_id); pre-sort the input or use batch replay"
            )
        previous = current
        yield request


class _ReportBuilder:
    """Keep constant-size counters and an optional prefix of request detail."""

    def __init__(self, max_requests: int | None) -> None:
        if max_requests is not None and (
            type(max_requests) is not int or max_requests < 0
        ):
            raise ValueError("max_requests must be a non-negative integer or None")
        self.max_requests = max_requests
        self.results: list[ReplayRequestResult] = []
        self.count = self.chunks = self.hits = self.tokens = 0
        self.prompt_tokens = self.observed_tokens = self.observed_requests = 0

    @property
    def wants_detail(self) -> bool:
        return self.max_requests is None or len(self.results) < self.max_requests

    def add(self, request: WorkloadRequest, result: ReplayRequestResult) -> None:
        self.count += 1
        self.chunks += result.total_chunks
        self.hits += result.matched_chunks
        self.tokens += result.matched_tokens
        self.prompt_tokens += request.prompt_tokens
        self.observed_tokens += request.observed_cached_tokens or 0
        self.observed_requests += request.observed_cached_tokens is not None
        if self.wants_detail:
            self.results.append(result)

    def finish(
        self, recorder: TraceRecorder, source: str, chunk_size: int,
        *, sort_by_arrival: bool,
    ) -> ReplayReport:
        return ReplayReport(
            requests=tuple(self.results), events=recorder.events,
            source=source, chunk_size=chunk_size,
            aggregate=ReplaySummary(
                requests=self.count, total_chunks=self.chunks,
                matched_chunks=self.hits, matched_tokens=self.tokens,
                prompt_tokens=self.prompt_tokens,
                observed_cached_tokens=self.observed_tokens,
                observed_requests=self.observed_requests,
                event_counts=recorder.event_counts, storage_hits=recorder.storage_hits,
            ),
            retention={
                "max_requests": self.max_requests, "max_events": recorder.max_events,
                "sample_every": recorder.sample_every,
                "requests_retained": len(self.results),
                "requests_omitted": self.count - len(self.results),
                "events_total": recorder.total_events,
                "events_retained": len(recorder),
                "events_omitted": recorder.total_events - len(recorder),
            },
            configuration={"ordering": "sorted" if sort_by_arrival else "stream"},
        )


class WorkloadReplayRunner:
    """Replay token-bearing :class:`WorkloadRequest` objects.

    A request is looked up, then its missing complete chunks are stored as
    opaque placeholders.  This models prefix reuse without requiring a model
    or PyTorch, while preserving the same key and stopping semantics as the
    real cache.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        namespace: str = "replay",
        cache: ChunkedPrefixCache | None = None,
        storage: CacheStorage | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self._owns_cache = cache is None
        self._namespace = namespace
        if cache is None:
            self.recorder = recorder if recorder is not None else TraceRecorder()
            self.cache = ChunkedPrefixCache(
                chunk_size=chunk_size,
                namespace=namespace,
                storage=storage,
                trace_recorder=self.recorder,
            )
        else:
            if cache.chunk_size != chunk_size:
                raise ValueError(
                    f"cache chunk_size {cache.chunk_size} does not match "
                    f"runner chunk_size {chunk_size}"
                )
            if recorder is not None and cache.trace_recorder is not recorder:
                raise ValueError(
                    "a supplied recorder must be attached to the supplied cache"
                )
            self.cache = cache
            self.recorder = (
                recorder if recorder is not None else (
                    cache.trace_recorder if cache.trace_recorder is not None
                    else TraceRecorder(enabled=False)
                )
            )
        self._model_id: str | None = None

    def run(
        self,
        requests: Iterable[WorkloadRequest],
        *,
        sort_by_arrival: bool = True,
        store_missing: bool = True,
        max_requests: int | None = None,
    ) -> ReplayReport:
        """Start a report, clearing the recorder while preserving cache state.

        Disable sorting only for input ordered by (arrival time, request ID).
        ``max_requests`` limits detail, never the number of processed requests.
        """
        builder = _ReportBuilder(max_requests)
        self.recorder.clear()
        for request in _ordered_requests(requests, sort_by_arrival):
            if request.token_ids is None:
                raise ValueError(
                    f"request {request.request_id!r} has no token_ids; use "
                    "BlockTraceReplayRunner for opaque block traces"
                )
            if self._model_id is not None and request.model_id != self._model_id:
                raise ValueError("token replay requires one model per runner")
            if self._model_id is None and self._owns_cache:
                # A persistent backend must not reuse another model's placeholders.
                self.cache = ChunkedPrefixCache(
                    chunk_size=self.cache.chunk_size,
                    namespace=json.dumps([self._namespace, request.model_id]),
                    storage=self.cache.storage, metrics=self.cache.metrics,
                    trace_recorder=self.recorder,
                )
            self._model_id = request.model_id
            lookup = self.cache.lookup(
                request.token_ids,
                request_id=request.request_id,
            )
            total_chunks, remainder = divmod(
                len(request.token_ids), self.cache.chunk_size
            )
            statuses = ["HIT"] * lookup.matched_chunks if builder.wants_detail else []
            if builder.wants_detail and lookup.matched_chunks < total_chunks:
                statuses.append("MISS")
                statuses.extend(["NOT_CHECKED"] * (total_chunks - len(statuses)))
            if builder.wants_detail and remainder:
                statuses.append("INCOMPLETE")

            if store_missing and lookup.matched_chunks < total_chunks:
                values = tuple(
                    {
                        "request_id": request.request_id,
                        "chunk_index": index,
                    }
                    for index in range(lookup.matched_chunks, total_chunks)
                )
                self.cache.store(
                    request.token_ids,
                    values,
                    start_chunk=lookup.matched_chunks,
                    request_id=request.request_id,
                )

            builder.add(request,
                ReplayRequestResult(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    model_id=request.model_id,
                    source="token_workload",
                    matched_chunks=lookup.matched_chunks,
                    total_chunks=total_chunks,
                    matched_tokens=lookup.matched_tokens,
                    chunk_size=self.cache.chunk_size,
                    statuses=tuple(statuses),
                    observed_cached_tokens=request.observed_cached_tokens,
                )
            )

        return builder.finish(
            self.recorder, "token_workload", self.cache.chunk_size,
            sort_by_arrival=sort_by_arrival,
        )


class BlockTraceReplayRunner:
    """Replay FreeInference-style opaque block IDs without token semantics."""

    def __init__(
        self,
        *,
        block_tokens: int = 16,
        namespace: str = "freeinference",
        recorder: TraceRecorder | None = None,
        storage: CacheStorage | None = None,
    ) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        self.block_tokens = block_tokens
        self.namespace = namespace
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self.storage = storage if storage is not None else MemoryKVCache()

    def _key(self, block_id: int, parent_key: str | None, model_id: str) -> str:
        payload = json.dumps(
            {
                "namespace": self.namespace,
                "model_id": model_id,
                "block_tokens": self.block_tokens,
                "parent_key": parent_key,
                "block_id": block_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def run(
        self,
        requests: Iterable[WorkloadRequest],
        *,
        sort_by_arrival: bool = True,
        store_missing: bool = True,
        max_requests: int | None = None,
    ) -> ReplayReport:
        """Replay once with independent report counters and a persistent cache.

        ``sort_by_arrival=False`` validates order without collecting input.
        The attached recorder is cleared before processing the new run.
        """
        builder = _ReportBuilder(max_requests)
        self.recorder.clear()
        for request in _ordered_requests(requests, sort_by_arrival):
            if request.prefix_blocks is None:
                raise ValueError(
                    f"request {request.request_id!r} has no prefix_blocks; use "
                    "WorkloadReplayRunner for token workloads"
                )
            block_ids = request.prefix_blocks
            self.recorder.record(
                "lookup_started",
                request_id=request.request_id,
                metadata={
                    "block_count": len(block_ids),
                    "block_tokens": self.block_tokens,
                    "opaque": True,
                },
            )
            keys: list[str] = []
            parent: str | None = None
            matched = 0
            for index, block_id in enumerate(block_ids):
                key = self._key(block_id, parent, request.model_id)
                keys.append(key)
                with request_context(request.request_id):
                    value = self.storage.get(key)
                if value is None:
                    self.recorder.record(
                        "chunk_miss",
                        request_id=request.request_id,
                        chunk_index=index,
                        key=key,
                        parent_key=parent,
                        result="miss",
                        metadata={"opaque_block": True},
                    )
                    self.recorder.record(
                        "lookup_stopped",
                        request_id=request.request_id,
                        chunk_index=index,
                        key=key,
                        result="first_miss",
                        metadata={"reason": "first_miss", "opaque_block": True},
                    )
                    for unchecked_index in range(index + 1, len(block_ids)):
                        unchecked_key = self._key(
                            block_ids[unchecked_index], key, request.model_id,
                        )
                        self.recorder.record(
                            "chunk_not_checked",
                            request_id=request.request_id,
                            chunk_index=unchecked_index,
                            key=unchecked_key,
                            parent_key=key,
                            result="not_checked",
                            metadata={"reason": "previous_miss", "opaque_block": True},
                        )
                        key = unchecked_key
                    break
                matched += 1
                self.recorder.record(
                    "chunk_hit",
                    request_id=request.request_id,
                    chunk_index=index,
                    key=key,
                    parent_key=parent,
                    result="hit",
                    metadata={"opaque_block": True},
                )
                parent = key

            if store_missing and matched < len(block_ids):
                parent = keys[matched - 1] if matched else None
                for index in range(matched, len(block_ids)):
                    key = self._key(block_ids[index], parent, request.model_id)
                    with request_context(request.request_id):
                        self.storage.put(key, {
                            "request_id": request.request_id,
                            "block_index": index,
                        }, parent_key=parent)
                    parent = key

            self.recorder.record(
                "lookup_completed",
                request_id=request.request_id,
                result="completed",
                metadata={
                    "matched_chunks": matched,
                    "matched_tokens": matched * self.block_tokens,
                    "miss_chunks": len(block_ids) - matched,
                    "opaque_block": True,
                },
            )
            builder.add(request,
                ReplayRequestResult(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    model_id=request.model_id,
                    source="opaque_block_workload",
                    matched_chunks=matched,
                    total_chunks=len(block_ids),
                    matched_tokens=matched * self.block_tokens,
                    chunk_size=self.block_tokens,
                    statuses=tuple(
                        ["HIT"] * matched
                        + (["MISS"] if matched < len(block_ids) else [])
                        + (["NOT_CHECKED"] * (len(block_ids) - matched - 1)
                           if matched < len(block_ids)
                           else [])
                    ) if builder.wants_detail else (),
                    observed_cached_tokens=request.observed_cached_tokens,
                )
            )

        return builder.finish(
            self.recorder, "opaque_block_workload", self.block_tokens,
            sort_by_arrival=sort_by_arrival,
        )
