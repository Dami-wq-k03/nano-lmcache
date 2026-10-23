"""Counters describing what a storage backend did to its entries."""

from dataclasses import dataclass


@dataclass
class StorageStats:
    """Eviction, rejection, and corruption totals kept by a storage backend.

    These live next to the storage, not in `CacheMetrics`, because only the
    storage knows when it evicts or refuses a write. `CacheMetrics` counts
    lookup outcomes, which the prefix cache observes. Keeping both lets you
    tell "hit rate fell because the working set is too big" (evictions
    climbing) from "hit rate fell because prompts stopped sharing prefixes".

    A write is rejected, rather than stored, for one of two reasons:

    - `rejected_no_room`: the payload and its required ancestors cannot fit
      within the entry or byte limits, so the write leaves the cache unchanged.
    - `rejected_missing_parent`: the new entry's parent is not stored (for
      example it was never stored, or an earlier chunk of the same prompt
      was rejected), so the entry would be unreachable.

    `corrupted_entries` counts entries whose stored payload failed an
    integrity check. Entries removed only because their ancestor was corrupt
    are not themselves counted as corrupted.
    """

    evictions: int = 0
    rejected_no_room: int = 0
    rejected_missing_parent: int = 0
    corrupted_entries: int = 0

    @property
    def rejected(self) -> int:
        """Total writes that were dropped instead of stored."""
        return self.rejected_no_room + self.rejected_missing_parent
