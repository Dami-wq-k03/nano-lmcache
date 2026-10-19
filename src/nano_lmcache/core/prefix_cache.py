"""Chunked prefix cache built on an in-memory KV store."""

from collections.abc import Sequence
from dataclasses import dataclass

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.storage.memory import MemoryKVCache


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
        storage: MemoryKVCache | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._chunk_size = chunk_size
        self._namespace = namespace
        self._storage = storage if storage is not None else MemoryKVCache()

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

        prefix_key: str | None = None
        for chunk, value in zip(complete_chunks, chunk_values, strict=True):
            key = make_chunk_key(
                chunk, namespace=self._namespace, prefix_key=prefix_key
            )
            self._storage.put(key, value)
            prefix_key = key

    def lookup(self, token_ids: Sequence[int]) -> PrefixLookupResult:
        matched_chunks = 0
        matched_tokens = 0
        values: list[object] = []
        keys: list[str] = []

        prefix_key: str | None = None
        for chunk in self._complete_chunks(token_ids):
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

        return PrefixLookupResult(
            matched_chunks=matched_chunks,
            matched_tokens=matched_tokens,
            values=tuple(values),
            keys=tuple(keys),
        )
