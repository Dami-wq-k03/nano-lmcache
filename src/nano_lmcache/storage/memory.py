"""In-memory key-value cache, standing in for a real KV-tensor store."""

from collections import OrderedDict

from nano_lmcache.storage.stats import StorageStats


class MemoryKVCache:
    """Optionally bounded cache keyed by chunk key, storing opaque payloads.

    Payloads are opaque `object` values here on purpose: this stage models
    the storage/lookup contract (put/get by key) without binding to any
    particular KV-tensor representation (e.g. PyTorch tensors).

    Capacity is counted in entries (one entry per chunk). When full, the
    least recently used *leaf* is evicted. Entries form prefix chains via
    `parent_key`, and a prefix lookup walks a chain from its root and stops
    at the first miss, so evicting an entry that still has descendants
    would strand them: they would occupy memory yet never be reachable.
    Evicting leaves only keeps every stored entry's whole ancestor chain
    present, so no orphan suffix can ever exist.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self._max_entries = max_entries
        # Order is recency: the first entry is the least recently used.
        self._entries: OrderedDict[str, object] = OrderedDict()
        self._parents: dict[str, str | None] = {}
        self._child_counts: dict[str, int] = {}
        self._stats = StorageStats()

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when the cache is unbounded."""
        return self._max_entries

    @property
    def stats(self) -> StorageStats:
        """Eviction and rejection totals for this cache."""
        return self._stats

    def __len__(self) -> int:
        return len(self._entries)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        # Overwrite: capacity and the entry's place in its chain are unchanged.
        if key in self._entries:
            self._entries[key] = value
            self._entries.move_to_end(key)
            return

        # Without its parent this entry could never be reached, so drop it.
        if parent_key is not None and parent_key not in self._entries:
            self._stats.rejected_missing_parent += 1
            return

        if (
            self._max_entries is not None
            and len(self._entries) >= self._max_entries
        ):
            # Never evict the new entry's parent, or it would be orphaned on arrival.
            victim = self._find_victim(protected=parent_key)
            if victim is None:
                # No room: keep the existing prefix and drop this longer suffix.
                self._stats.rejected_no_room += 1
                return
            self._evict(victim)

        self._entries[key] = value
        self._parents[key] = parent_key
        if parent_key is not None:
            self._child_counts[parent_key] = (
                self._child_counts.get(parent_key, 0) + 1
            )

    def get(self, key: str) -> object | None:
        if key not in self._entries:
            return None

        self._entries.move_to_end(key)
        return self._entries[key]

    def _find_victim(self, protected: str | None) -> str | None:
        """Oldest entry with no descendants, excluding `protected`."""
        for key in self._entries:
            if key != protected and self._child_counts.get(key, 0) == 0:
                return key

        return None

    def _evict(self, key: str) -> None:
        del self._entries[key]
        self._stats.evictions += 1
        parent_key = self._parents.pop(key)
        self._child_counts.pop(key, None)

        if parent_key is not None:
            self._child_counts[parent_key] -= 1
