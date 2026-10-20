"""Chunked prefix cache built on an in-memory KV store."""

from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.core.metrics import CacheMetrics
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
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._chunk_size = chunk_size
        self._namespace = namespace
        self._storage = storage if storage is not None else MemoryKVCache()
        self._metrics = metrics if metrics is not None else CacheMetrics()

    @property
    def metrics(self) -> CacheMetrics:
        """Hit/miss totals accumulated by this cache's lookups."""
        return self._metrics

    def _complete_chunks(
        self, token_ids: Sequence[int]
    ) -> list[tuple[int, ...]]:
        chunks = chunk_tokens(token_ids, self._chunk_size)
        return [chunk for chunk in chunks if len(chunk) == self._chunk_size]

    def store(
        self, token_ids: Sequence[int], chunk_values: Sequence[object]
    ) -> None:
        complete_chunks = self._complete_chunks(token_ids)
        if len(chunk_values) != len(complete_chunks):
            raise ValueError(
                f"expected {len(complete_chunks)} chunk_values "
                f"for {len(complete_chunks)} complete chunks, "
                f"got {len(chunk_values)}"
            )

        # None 是存储层的 miss 标记，先整体校验再写入，避免写入一半失败。
        if any(value is None for value in chunk_values):
            raise ValueError("chunk_values must not contain None")

        prefix_key: str | None = None
        for chunk, value in zip(complete_chunks, chunk_values, strict=True):
            key = make_chunk_key(
                chunk, namespace=self._namespace, prefix_key=prefix_key
            )
            self._storage.put(key, value, parent_key=prefix_key)
            prefix_key = key

    def lookup(self, token_ids: Sequence[int]) -> PrefixLookupResult:
        complete_chunks = self._complete_chunks(token_ids)

        matched_chunks = 0
        matched_tokens = 0
        values: list[object] = []
        keys: list[str] = []

        prefix_key: str | None = None
        for chunk in complete_chunks:
            key = make_chunk_key(
                chunk, namespace=self._namespace, prefix_key=prefix_key
            )
            value = self._storage.get(key)
            if value is None:
                break

            matched_chunks += 1
            matched_tokens += len(chunk)
            values.append(value)
            keys.append(key)
            prefix_key = key

        self._metrics.record_lookup(
            hit_chunks=matched_chunks,
            miss_chunks=len(complete_chunks) - matched_chunks,
            hit_tokens=matched_tokens,
        )

        return PrefixLookupResult(
            matched_chunks=matched_chunks,
            matched_tokens=matched_tokens,
            values=tuple(values),
            keys=tuple(keys),
        )
