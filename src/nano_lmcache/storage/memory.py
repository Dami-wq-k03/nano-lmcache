"""In-memory key-value cache, standing in for a real KV-tensor store."""

from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.stats import StorageStats


class MemoryKVCache:
    """Optionally bounded cache keyed by chunk key, storing opaque payloads.

    Payloads are opaque `object` values here on purpose: this stage models
    the storage/lookup contract (put/get by key) without binding to any
    particular KV-tensor representation (e.g. PyTorch tensors).

    Capacity is counted in entries (one entry per chunk). When full, the
    least recently used *leaf* is evicted, following the prefix-safe rules
    in `PrefixLRUIndex`: evicting an entry that still has descendants would
    strand them, since a prefix lookup could never reach them again.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._index = PrefixLRUIndex(max_entries)
        self._values: dict[str, object] = {}

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when the cache is unbounded."""
        return self._index.max_entries

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
        decision = self._index.admit(key, parent_key)
        for victim in decision.evicted:
            del self._values[victim]

        if decision.admitted:
            self._values[key] = value

    def get(self, key: str) -> object | None:
        if key not in self._values:
            return None

        self._index.touch(key)
        return self._values[key]
