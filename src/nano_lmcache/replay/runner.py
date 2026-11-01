"""Replay engines for token workloads and opaque block traces."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.datasets.schemas.workload import WorkloadRequest
from nano_lmcache.observability.events import TraceEvent
from nano_lmcache.observability.recorder import TraceRecorder
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
class ReplayReport:
    """Requests, trace events, and aggregate counters from one replay."""

    requests: tuple[ReplayRequestResult, ...]
    events: tuple[TraceEvent, ...]
    source: str
    chunk_size: int

    @property
    def total_chunks(self) -> int:
        return sum(item.total_chunks for item in self.requests)

    @property
    def matched_chunks(self) -> int:
        return sum(item.matched_chunks for item in self.requests)

    @property
    def hit_rate(self) -> float:
        return self.matched_chunks / self.total_chunks if self.total_chunks else 0.0

    @property
    def observed_cached_tokens(self) -> int:
        return sum(
            item.observed_cached_tokens or 0 for item in self.requests
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chunk_size": self.chunk_size,
            "summary": {
                "requests": len(self.requests),
                "total_chunks": self.total_chunks,
                "matched_chunks": self.matched_chunks,
                "hit_rate": self.hit_rate,
                "observed_cached_tokens": self.observed_cached_tokens,
            },
            "requests": [item.to_dict() for item in self.requests],
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


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
                recorder
                or cache.trace_recorder
                or TraceRecorder(enabled=False)
            )

    def run(
        self,
        requests: Iterable[WorkloadRequest],
        *,
        sort_by_arrival: bool = True,
        store_missing: bool = True,
    ) -> ReplayReport:
        ordered = list(requests)
        if sort_by_arrival:
            ordered.sort(key=lambda item: (item.arrival_time_ms, item.request_id))

        results: list[ReplayRequestResult] = []
        for request in ordered:
            if request.token_ids is None:
                raise ValueError(
                    f"request {request.request_id!r} has no token_ids; use "
                    "BlockTraceReplayRunner for opaque block traces"
                )
            lookup = self.cache.lookup(
                request.token_ids,
                request_id=request.request_id,
            )
            total_chunks, remainder = divmod(
                len(request.token_ids), self.cache.chunk_size
            )
            statuses = ["HIT"] * lookup.matched_chunks
            if lookup.matched_chunks < total_chunks:
                statuses.append("MISS")
                statuses.extend(["NOT_CHECKED"] * (total_chunks - len(statuses)))
            if remainder:
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
                )

            results.append(
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

        return ReplayReport(
            requests=tuple(results),
            events=self.recorder.events,
            source="token_workload",
            chunk_size=self.cache.chunk_size,
        )


class BlockTraceReplayRunner:
    """Replay FreeInference-style opaque block IDs without token semantics."""

    def __init__(
        self,
        *,
        block_tokens: int = 16,
        namespace: str = "freeinference",
        recorder: TraceRecorder | None = None,
    ) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        self.block_tokens = block_tokens
        self.namespace = namespace
        self.recorder = recorder if recorder is not None else TraceRecorder()
        self._values: dict[str, object] = {}

    def _key(self, block_id: int, parent_key: str | None) -> str:
        payload = json.dumps(
            {
                "namespace": self.namespace,
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
    ) -> ReplayReport:
        ordered = list(requests)
        if sort_by_arrival:
            ordered.sort(key=lambda item: (item.arrival_time_ms, item.request_id))
        results: list[ReplayRequestResult] = []

        for request in ordered:
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
                key = self._key(block_id, parent)
                keys.append(key)
                if key not in self._values:
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
                            block_ids[unchecked_index], key
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
                    key = self._key(block_ids[index], parent)
                    self._values[key] = {
                        "request_id": request.request_id,
                        "block_index": index,
                    }
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
            results.append(
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
                    ),
                    observed_cached_tokens=request.observed_cached_tokens,
                )
            )

        return ReplayReport(
            requests=tuple(results),
            events=self.recorder.events,
            source="opaque_block_workload",
            chunk_size=self.block_tokens,
        )
