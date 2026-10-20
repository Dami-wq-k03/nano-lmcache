"""In-memory key-value cache, standing in for a real KV-tensor store."""

import sys
from collections.abc import Callable

from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.stats import StorageStats


class MemoryKVCache:
    """Optionally bounded cache keyed by chunk key, storing opaque payloads.

    Payloads are opaque `object` values here on purpose: this stage models
    the storage/lookup contract (put/get by key) without binding to any
    particular KV-tensor representation (e.g. PyTorch tensors).

    Capacity can be bounded by entries, payload bytes, or both. When full, the
    least recently used *leaf* is evicted, following the prefix-safe rules
    in `PrefixLRUIndex`: evicting an entry that still has descendants would
    strand them, since a prefix lookup could never reach them again.

    `size_fn` measures each payload on `put`, defaulting to the shallow
    `sys.getsizeof` (not recursive or tensor storage size). Keys and cache
    bookkeeping are excluded. Do not resize cached payloads in place;
    sizes are snapshots, refreshed only when the payload is put again.
    """

    def __init__(
        self,
        max_entries: int | None = None,
        *,
        max_bytes: int | None = None,
        size_fn: Callable[[object], int] = sys.getsizeof,
    ) -> None:
        self._index = PrefixLRUIndex(max_entries, max_bytes=max_bytes)
        self._size_fn = size_fn
        self._values: dict[str, object] = {}

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when the cache is unbounded."""
        return self._index.max_entries

    @property
    def max_bytes(self) -> int | None:
        """Payload byte limit, or None when unbounded."""
        return self._index.max_bytes

    @property
    def nbytes(self) -> int:
        """Sum of payload sizes measured at their last successful put."""
        return self._index.nbytes

    @property
    def stats(self) -> StorageStats:
        """Eviction and rejection totals for this cache."""
        return self._index.stats

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        """Whether `key` is stored; unlike `get`, does not count as a use."""
        return key in self._values

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for a stored `key`; KeyError if absent."""
        if key not in self._values:
            raise KeyError(key)
        return self._index.parent_of(key)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        # Measure and validate before changing recency, values, or capacity.
        decision = self._index.admit(key, parent_key, size_bytes=self._size_fn(value))
        for victim in decision.evicted:
            del self._values[victim]

        if decision.admitted:
            self._values[key] = value

    def get(self, key: str) -> object | None:
        if key not in self._values:
            return None

        self._index.touch(key)
        return self._values[key]
