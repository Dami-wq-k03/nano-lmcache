"""Prefix-safe LRU eviction, kept separate from where payloads live.

Every bounded backend (memory, disk, CPU offload) needs the same rules, so
the bookkeeping lives here and each backend only moves bytes:

1. Only an entry with no children (a leaf) may be evicted.
2. Among leaves, the least recently used goes first.
3. The parent of the entry being admitted is never the victim.
4. If no victim exists, the new entry is rejected, so a chain longer than
   the capacity keeps its longest prefix that fits.
5. An entry whose parent is not present is rejected.

Together these keep one invariant: every entry's whole ancestor chain is
present. A prefix lookup walks a chain from its root and stops at the
first miss, so an entry without its ancestors could never be reached.

A backend calls `admit` before storing a payload, deletes the payloads of
the keys it names as evicted, and stores the new payload only if admitted.
"""

import heapq
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

from nano_lmcache.storage.stats import StorageStats


@dataclass(frozen=True)
class Admission:
    """What a backend must do to make room for one new entry."""

    admitted: bool
    # Keys whose payloads the backend must delete, oldest first.
    evicted: tuple[str, ...]


class PrefixLRUIndex:
    """Tracks keys, their parents, and recency; decides what to evict."""

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self._max_entries = max_entries
        # Order is recency: the first key is the least recently used.
        # The value is the key's parent (None for a chain root).
        self._parents: OrderedDict[str, str | None] = OrderedDict()
        self._child_counts: dict[str, int] = {}
        self._clock = 0
        self._last_access: dict[str, int] = {}
        # Bounded indexes keep leaf candidates ordered by access time. Hits
        # and becoming an internal node invalidate old heap records lazily.
        # Unbounded indexes need no candidates, since they never evict.
        self._leaf_heap: list[tuple[int, str]] = []
        self._stats = StorageStats()

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when unbounded."""
        return self._max_entries

    @property
    def stats(self) -> StorageStats:
        """Eviction and rejection totals."""
        return self._stats

    def __len__(self) -> int:
        return len(self._parents)

    def __contains__(self, key: object) -> bool:
        return key in self._parents

    def keys_by_recency(self) -> Iterator[str]:
        """Keys from least to most recently used."""
        return iter(self._parents)

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for `key`, which must be present."""
        return self._parents[key]

    def touch(self, key: str) -> None:
        """Mark `key` as most recently used, for example after a hit."""
        self._parents.move_to_end(key)
        self._record_access(key)

    def admit(self, key: str, parent_key: str | None) -> Admission:
        """Decide whether `key` may be stored, and what must go to fit it."""
        # Re-storing a present key: its place in its chain cannot change.
        if key in self._parents:
            self.touch(key)
            return Admission(admitted=True, evicted=())

        if parent_key is not None and parent_key not in self._parents:
            self._stats.rejected_missing_parent += 1
            return Admission(admitted=False, evicted=())

        evicted: tuple[str, ...] = ()
        if self._max_entries is not None and len(self) >= self._max_entries:
            victim = self._find_victim(protected=parent_key)
            if victim is None:
                self._stats.rejected_no_room += 1
                return Admission(admitted=False, evicted=())
            self._remove(victim)
            self._stats.evictions += 1
            evicted = (victim,)

        self._add(key, parent_key)
        return Admission(admitted=True, evicted=evicted)

    def restore(self, key: str, parent_key: str | None) -> None:
        """Re-add an entry that already exists, for example after a restart.

        Unlike `admit`, this never evicts and never counts stats; a backend
        uses it to rebuild its index from storage it already holds. Restore
        keys from least to most recently used, parents before children.
        """
        if parent_key is not None and parent_key not in self._parents:
            raise ValueError(f"parent {parent_key!r} must be restored first")
        if self._max_entries is not None and len(self) >= self._max_entries:
            raise ValueError("cannot restore more entries than max_entries")

        self._add(key, parent_key)

    def _add(self, key: str, parent_key: str | None) -> None:
        self._parents[key] = parent_key
        if parent_key is not None:
            self._child_counts[parent_key] = (
                self._child_counts.get(parent_key, 0) + 1
            )
        self._record_access(key)

    def _remove(self, key: str) -> None:
        parent_key = self._parents.pop(key)
        self._last_access.pop(key, None)
        self._child_counts.pop(key, None)
        if parent_key is not None:
            self._child_counts[parent_key] -= 1
            if self._max_entries is not None and self._child_counts[parent_key] == 0:
                # Exposure is not an access. Reinsert with the parent's last
                # actual access time, preserving the original LRU policy.
                heapq.heappush(
                    self._leaf_heap, (self._last_access[parent_key], parent_key)
                )

    def _record_access(self, key: str) -> None:
        if self._max_entries is None:
            return
        self._clock += 1
        self._last_access[key] = self._clock
        if self._child_counts.get(key, 0):
            return
        heapq.heappush(self._leaf_heap, (self._clock, key))

        # Lazy deletion must not retain every hit forever. Rebuild after
        # enough stale records accumulate to pay for this O(n) pass, keeping
        # memory O(n) and candidate maintenance O(log n) amortized.
        if len(self._leaf_heap) > 64 and len(self._leaf_heap) > 2 * len(self._parents):
            self._leaf_heap = [
                (tick, candidate)
                for candidate, tick in self._last_access.items()
                if self._child_counts.get(candidate, 0) == 0
            ]
            heapq.heapify(self._leaf_heap)

    def _find_victim(self, protected: str | None) -> str | None:
        """Least recently used leaf, excluding `protected`.

        A stale record is popped at most once. A protected leaf is only
        skipped for this admission; keep it for future victim selection.
        """
        protected_leaf: tuple[int, str] | None = None
        victim = None
        while self._leaf_heap:
            tick, key = heapq.heappop(self._leaf_heap)
            if self._last_access.get(key) != tick or self._child_counts.get(key, 0):
                continue
            if key == protected:
                protected_leaf = (tick, key)
                continue
            victim = key
            break
        if protected_leaf is not None:
            heapq.heappush(self._leaf_heap, protected_leaf)
        return victim
