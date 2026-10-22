"""Prefix-safe LRU eviction, kept separate from where payloads live.

Every bounded backend (memory, disk, CPU offload) needs the same rules, so
the bookkeeping lives here and each backend only moves bytes:

1. Only an entry with no children (a leaf) may be evicted.
2. Among leaves, the least recently used goes first.
3. A new entry's parent, or the key being overwritten, is protected.
4. If the entry and its ancestors cannot fit, the write is rejected
   without evicting anything, keeping the longest prefix that fits.
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
    """Tracks keys, their parents, recency, and backend-reported sizes.

    Byte accounting requires the backend to supply `size_bytes`; it
    defaults to zero for entry-only backends. Currently only MemoryKVCache
    exposes a byte limit. Disk and CPU-offload caches use entry limits.
    """

    def __init__(
        self, max_entries: int | None = None, *, max_bytes: int | None = None
    ) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")

        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._sizes: dict[str, int] = {}
        self._nbytes = 0
        # Order is recency: the first key is the least recently used.
        # The value is the key's parent (None for a chain root).
        self._parents: OrderedDict[str, str | None] = OrderedDict()
        self._child_counts: dict[str, int] = {}
        self._clock = 0
        # In bounded indexes (either limit set), _last_access and _parents
        # have the same keys at public-operation boundaries. Removing a key
        # deletes its tick; stale heap records are identified by a missing
        # or different tick.
        # Unbounded indexes leave _last_access empty.
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
    def max_bytes(self) -> int | None:
        """Payload byte limit, or None when unbounded."""
        return self._max_bytes

    @property
    def nbytes(self) -> int:
        """Sum of the recorded payload sizes."""
        return self._nbytes

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

    def admit(
        self, key: str, parent_key: str | None, *, size_bytes: int = 0
    ) -> Admission:
        """Decide whether `key` may be stored, and what must go to fit it."""
        _validate_size(size_bytes)
        present = key in self._parents
        # Re-storing a present key: its place in its chain cannot change.
        if present:
            parent_key = self._parents[key]

        if parent_key is not None and parent_key not in self._parents:
            self._stats.rejected_missing_parent += 1
            return Admission(admitted=False, evicted=())

        added_entries = 0 if present else 1
        added_bytes = size_bytes - self._sizes.get(key, 0)
        if not self._fits(len(self) + added_entries, self._nbytes + added_bytes):
            # Check the smallest possible surviving chain before evicting.
            # Otherwise an impossible write could discard unrelated data
            # before discovering that its protected ancestors take too much room.
            chain_entries, chain_bytes = 1, size_bytes
            ancestor = parent_key
            while ancestor is not None:
                chain_entries += 1
                chain_bytes += self._sizes[ancestor]
                ancestor = self._parents[ancestor]
            if not self._fits(chain_entries, chain_bytes):
                self._stats.rejected_no_room += 1
                return Admission(admitted=False, evicted=())

        evicted: list[str] = []
        protected = key if present else parent_key
        while not self._fits(len(self) + added_entries, self._nbytes + added_bytes):
            victim = self._find_victim(protected=protected)
            # The preflight proved that evicting down to the protected
            # chain is enough, so an unprotected leaf must exist here.
            assert victim is not None
            self._remove(victim)
            self._stats.evictions += 1
            evicted.append(victim)

        if present:
            self._sizes[key] = size_bytes
            self._nbytes += added_bytes
            self.touch(key)
        else:
            self._add(key, parent_key, size_bytes)
        return Admission(admitted=True, evicted=tuple(evicted))

    def invalidate(self, key: str) -> tuple[str, ...]:
        """Remove a subtree, counting each removed entry as an eviction."""
        if key not in self._parents:
            return ()
        children: dict[str, list[str]] = {}
        for child, parent in self._parents.items():
            if parent is not None:
                children.setdefault(parent, []).append(child)
        subtree = [key]
        for parent in subtree:
            subtree.extend(children.get(parent, ()))
        # Recency need not put parents before children. Traverse the tree,
        # then remove leaves first so _remove keeps child counts consistent.
        victims = tuple(reversed(subtree))
        for victim in victims:
            self._remove(victim)
            self._stats.evictions += 1
        return victims

    def restore(self, key: str, parent_key: str | None, *, size_bytes: int = 0) -> None:
        """Re-add an entry that already exists, for example after a restart.

        Unlike `admit`, this never evicts and never counts stats; a backend
        uses it to rebuild its index from storage it already holds. Restore
        keys from least to most recently used, parents before children.
        """
        _validate_size(size_bytes)
        if parent_key is not None and parent_key not in self._parents:
            raise ValueError(f"parent {parent_key!r} must be restored first")
        if self._max_entries is not None and len(self) >= self._max_entries:
            raise ValueError("cannot restore more entries than max_entries")
        if self._max_bytes is not None and self._nbytes + size_bytes > self._max_bytes:
            raise ValueError("cannot restore more bytes than max_bytes")

        self._add(key, parent_key, size_bytes)

    @property
    def _bounded(self) -> bool:
        # Either limit can force evictions, so either needs leaf candidates.
        return self._max_entries is not None or self._max_bytes is not None

    def _fits(self, entries: int, nbytes: int) -> bool:
        return (self._max_entries is None or entries <= self._max_entries) and (
            self._max_bytes is None or nbytes <= self._max_bytes
        )

    def _add(self, key: str, parent_key: str | None, size_bytes: int) -> None:
        self._parents[key] = parent_key
        self._sizes[key] = size_bytes
        self._nbytes += size_bytes
        if parent_key is not None:
            self._child_counts[parent_key] = (
                self._child_counts.get(parent_key, 0) + 1
            )
        self._record_access(key)

    def _remove(self, key: str) -> None:
        parent_key = self._parents.pop(key)
        self._nbytes -= self._sizes.pop(key)
        self._last_access.pop(key, None)
        self._child_counts.pop(key, None)
        if parent_key is not None:
            self._child_counts[parent_key] -= 1
            if self._bounded and self._child_counts[parent_key] == 0:
                # Exposure is not an access. Reinsert with the parent's last
                # actual access time, preserving the original LRU policy.
                heapq.heappush(
                    self._leaf_heap, (self._last_access[parent_key], parent_key)
                )

    def _record_access(self, key: str) -> None:
        if not self._bounded:
            return
        self._clock += 1
        self._last_access[key] = self._clock
        if self._child_counts.get(key, 0):
            return
        heapq.heappush(self._leaf_heap, (self._clock, key))

        # 64 is a small-cache slack floor: avoid rebuilding a tiny heap on
        # frequent hits. Above 2*n records, at most n records are needed
        # for current leaves; stale/duplicate records justify an O(n)
        # rebuild. Charging that rebuild
        # to the preceding additions costs O(1) amortized per addition;
        # heap operations remain O(log n). A rebuild can still cause a
        # latency spike, while retained records stay O(n) overall.
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


def _validate_size(size_bytes: int) -> None:
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise ValueError("size_bytes must be a non-negative integer")
