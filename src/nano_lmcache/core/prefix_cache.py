"""Chunked prefix cache built on an in-memory KV store."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.core.metrics import CacheMetrics
from nano_lmcache.observability.context import request_context
from nano_lmcache.observability.recorder import TraceRecorder
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage


@dataclass(frozen=True)
class PrefixLookupResult:
    """Result of a prefix-cache lookup."""

    matched_chunks: int
    matched_tokens: int
    values: tuple[object, ...]
    keys: tuple[str, ...]


class ChunkedPrefixCache:
    """Reuse cached payloads for matching token prefixes.

    Only complete chunks (length == chunk_size) participate in storage and
    matching; a trailing partial chunk is excluded from this version's reuse
    scope, since its KV state depends on a yet-incomplete token span.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        namespace: str,
        storage: CacheStorage | None = None,
        metrics: CacheMetrics | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._chunk_size = chunk_size
        self._namespace = namespace
        self._storage = storage if storage is not None else MemoryKVCache()
        self._metrics = metrics if metrics is not None else CacheMetrics()
        self._trace_recorder = trace_recorder
        self._trace_request_sequence = 0

    @property
    def storage(self) -> CacheStorage:
        """The backend holding this cache's payloads."""
        return self._storage

    @property
    def chunk_size(self) -> int:
        """Number of tokens in one complete chunk."""
        return self._chunk_size

    @property
    def metrics(self) -> CacheMetrics:
        """Hit/miss totals accumulated by this cache's lookups."""
        return self._metrics

    @property
    def trace_recorder(self) -> TraceRecorder | None:
        """The optional recorder receiving lookup events."""
        return self._trace_recorder

    def _split_chunks(
        self, token_ids: Sequence[int]
    ) -> tuple[list[tuple[int, ...]], tuple[int, ...] | None]:
        """Return reusable chunks and the optional incomplete tail."""
        chunks = chunk_tokens(token_ids, self._chunk_size)
        if chunks and len(chunks[-1]) < self._chunk_size:
            return chunks[:-1], chunks[-1]
        return chunks, None

    def store(
        self, token_ids: Sequence[int],
        chunk_values: Sequence[object],
        start_chunk: int = 0,
        *,
        request_id: str | None = None,
    ) -> None:
        complete_chunks, _ = self._split_chunks(token_ids)
        num_chunks = len(complete_chunks)

        if (
            type(start_chunk) is not int
            or not 0 <= start_chunk <= num_chunks
        ):
            raise ValueError(
                f"start_chunk must be an integer between 0 and {num_chunks}, "
                f"got {start_chunk!r}"
            )

        expected_values = num_chunks - start_chunk
        if len(chunk_values) != expected_values:
            raise ValueError(
                f"expected {expected_values} chunk_values "
                f"for {num_chunks} complete chunks "
                f"starting at chunk {start_chunk}, "
                f"got {len(chunk_values)}"
            )

        if any(value is None for value in chunk_values):
            raise ValueError("chunk_values must not contain None")

        prefix_key: str | None = None

        for index, chunk in enumerate(complete_chunks):
            key = make_chunk_key(
                chunk,
                namespace=self._namespace,
                prefix_key=prefix_key
            )

            if index >= start_chunk:
                with request_context(request_id):
                    self._storage.put(
                        key,
                        chunk_values[index - start_chunk],
                        parent_key=prefix_key,
                    )
            prefix_key = key

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        request_id: str | None = None,
    ) -> PrefixLookupResult:
        complete_chunks, incomplete_chunk = self._split_chunks(token_ids)
        all_chunk_count = len(complete_chunks) + (incomplete_chunk is not None)

        trace_enabled = (
            self._trace_recorder is not None and self._trace_recorder.enabled
        )
        trace_request_id = request_id
        if (
            trace_enabled
            and self._trace_recorder is not None
            and trace_request_id is None
            and self._trace_recorder.request_id is None
        ):
            trace_request_id = f"lookup-{self._trace_request_sequence}"
        if trace_enabled:
            self._trace_request_sequence += 1

        self._record_trace(
            "lookup_started",
            request_id=trace_request_id,
            metadata={
                "token_count": len(token_ids),
                "chunk_size": self._chunk_size,
                "complete_chunks": len(complete_chunks),
                "has_incomplete_chunk": incomplete_chunk is not None,
            },
        )

        matched_chunks = 0
        matched_tokens = 0
        values: list[object] = []
        keys: list[str] = []

        prefix_key: str | None = None
        miss_index: int | None = None
        for index, chunk in enumerate(complete_chunks):
            key = make_chunk_key(
                chunk, namespace=self._namespace, prefix_key=prefix_key
            )
            started_ns = (
                time.perf_counter_ns() if trace_enabled else 0
            )
            with request_context(trace_request_id):
                value = self._storage.get(key)
            duration_ns = (
                time.perf_counter_ns() - started_ns
                if trace_enabled
                else None
            )
            if value is None:
                miss_index = index
                self._record_trace(
                    "chunk_miss",
                    request_id=trace_request_id,
                    chunk_index=index,
                    key=key,
                    parent_key=prefix_key,
                    result="miss",
                    duration_ns=duration_ns,
                )
                self._record_trace(
                    "lookup_stopped",
                    request_id=trace_request_id,
                    chunk_index=index,
                    key=key,
                    parent_key=prefix_key,
                    result="first_miss",
                    metadata={"reason": "first_miss"},
                )
                break

            matched_chunks += 1
            matched_tokens += len(chunk)
            values.append(value)
            keys.append(key)
            self._record_trace(
                "chunk_hit",
                request_id=trace_request_id,
                chunk_index=index,
                key=key,
                parent_key=prefix_key,
                result="hit",
                duration_ns=duration_ns,
            )
            prefix_key = key

        if miss_index is not None:
            candidate_parent_key = make_chunk_key(
                complete_chunks[miss_index],
                namespace=self._namespace,
                prefix_key=prefix_key,
            )
            for index in range(miss_index + 1, len(complete_chunks)):
                key = make_chunk_key(
                    complete_chunks[index],
                    namespace=self._namespace,
                    prefix_key=candidate_parent_key,
                )
                self._record_trace(
                    "chunk_not_checked",
                    request_id=trace_request_id,
                    chunk_index=index,
                    key=key,
                    parent_key=candidate_parent_key,
                    result="not_checked",
                    metadata={"reason": "previous_miss"},
                )
                candidate_parent_key = key
            prefix_key = candidate_parent_key

        if incomplete_chunk is not None:
            self._record_trace(
                "chunk_incomplete",
                request_id=trace_request_id,
                chunk_index=all_chunk_count - 1,
                parent_key=prefix_key,
                result="incomplete",
                metadata={
                    "token_count": len(incomplete_chunk),
                    "chunk_size": self._chunk_size,
                },
            )

        self._metrics.record_lookup(
            hit_chunks=matched_chunks,
            miss_chunks=len(complete_chunks) - matched_chunks,
            hit_tokens=matched_tokens,
        )

        self._record_trace(
            "lookup_completed",
            request_id=trace_request_id,
            result="completed",
            metadata={
                "matched_chunks": matched_chunks,
                "matched_tokens": matched_tokens,
                "miss_chunks": len(complete_chunks) - matched_chunks,
                "checked_chunks": matched_chunks + (1 if miss_index is not None else 0),
                "unchecked_chunks": (
                    len(complete_chunks) - miss_index - 1
                    if miss_index is not None
                    else 0
                ),
                "incomplete_chunks": 1 if incomplete_chunk is not None else 0,
            },
        )

        return PrefixLookupResult(
            matched_chunks=matched_chunks,
            matched_tokens=matched_tokens,
            values=tuple(values),
            keys=tuple(keys),
        )

    def _record_trace(
        self,
        event_type: str,
        *,
        request_id: str | None,
        chunk_index: int | None = None,
        key: str | None = None,
        parent_key: str | None = None,
        result: str | None = None,
        duration_ns: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self._trace_recorder is None or not self._trace_recorder.enabled:
            return
        self._trace_recorder.record(
            event_type,
            request_id=request_id,
            chunk_index=chunk_index,
            key=key,
            parent_key=parent_key,
            result=result,
            duration_ns=duration_ns,
            metadata=metadata,
        )
