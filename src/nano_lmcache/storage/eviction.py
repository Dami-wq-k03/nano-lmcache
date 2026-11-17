"""Prefix-safe admission shared by memory, disk, and CPU-offload storage.

Only leaves may be evicted; the incoming entry's ancestor chain is protected.
Victim selection is a plain O(n) scan, deliberately trading heap optimization
for a small, readable implementation. Policies only score eligible keys.
"""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

from nano_lmcache.storage.eviction_policies import PolicyLike, make_eviction_policy
from nano_lmcache.storage.stats import StorageStats


@dataclass(frozen=True)
class Admission:
    """Whether a backend may store the value, and which payloads to delete."""

    admitted: bool
    evicted: tuple[str, ...]


@dataclass
class _Entry:
    parent: str | None
    size: int
    frequency: int = 1
    last_access: int = 0


class PrefixIndex:
    """Track metadata, not payloads. Every stored key has all its ancestors."""

    def __init__(
        self,
        max_entries: int | None = None,
        *,
        max_bytes: int | None = None,
        eviction_policy: PolicyLike = "lru",
    ) -> None:
        for name, limit in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if limit is not None and (type(limit) is not int or limit <= 0):
                raise ValueError(f"{name} must be a positive integer")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._policy = make_eviction_policy(eviction_policy)
        # Oldest to newest; each key has exactly one metadata record.
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._clock = 0
        self._nbytes = 0
        self._stats = StorageStats()

    @property
    def max_entries(self) -> int | None:
        return self._max_entries

    @property
    def max_bytes(self) -> int | None:
        return self._max_bytes

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def stats(self) -> StorageStats:
        return self._stats

    @property
    def eviction_policy(self) -> str:
        return self._policy.name

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def keys_by_recency(self) -> Iterator[str]:
        return iter(self._entries)

    def parent_of(self, key: str) -> str | None:
        return self._entries[key].parent

    def frequency_of(self, key: str) -> int:
        return self._entries[key].frequency

    def touch(self, key: str) -> None:
        """A real hit increases frequency; an overwrite only refreshes recency."""
        self._entries[key].frequency += 1
        self._mark_used(key)

    def admit(
        self, key: str, parent_key: str | None, *, size_bytes: int = 0
    ) -> Admission:
        """Reject impossible writes before changing any existing entry."""
        _validate_size(size_bytes)
        present = key in self._entries
        if present:
            parent_key = self.parent_of(key)  # Overwrites cannot reparent a key.
        if parent_key is not None and parent_key not in self._entries:
            self._stats.rejected_missing_parent += 1
            return Admission(False, ())

        added_entries = 0 if present else 1
        added_bytes = size_bytes - (self._entries[key].size if present else 0)
        if not self._fits(len(self) + added_entries, self._nbytes + added_bytes):
            # The new value and its ancestors must survive every eviction.
            chain_entries, chain_bytes = 1, size_bytes
            ancestor = parent_key
            while ancestor is not None:
                entry = self._entries[ancestor]
                chain_entries += 1
                chain_bytes += entry.size
                ancestor = entry.parent
            if not self._fits(chain_entries, chain_bytes):
                self._stats.rejected_no_room += 1
                return Admission(False, ())

        evicted: list[str] = []
        protected = key if present else parent_key
        while not self._fits(len(self) + added_entries, self._nbytes + added_bytes):
            victim = self._find_victim(protected)
            assert victim is not None  # Preflight proved an eligible leaf exists.
            self._remove(victim)
            self._stats.evictions += 1
            evicted.append(victim)

        if present:
            self._entries[key].size = size_bytes
        else:
            self._entries[key] = _Entry(parent_key, size_bytes)
        self._nbytes += added_bytes
        self._mark_used(key)
        return Admission(True, tuple(evicted))

    def discard_subtree(self, key: str) -> tuple[str, ...]:
        """Remove an unusable key and descendants, children first, without stats."""
        if key not in self._entries:
            return ()
        children: dict[str | None, list[str]] = {}
        for child, entry in self._entries.items():
            children.setdefault(entry.parent, []).append(child)
        subtree = [key]
        for parent in subtree:
            subtree.extend(children.get(parent, ()))
        victims = tuple(reversed(subtree))
        for victim in victims:
            self._remove(victim)
        return victims

    def invalidate(self, key: str) -> tuple[str, ...]:
        """Remove a stale subtree, counting its entries as evictions."""
        victims = self.discard_subtree(key)
        self._stats.evictions += len(victims)
        return victims

    def restore(
        self, key: str, parent_key: str | None, *, size_bytes: int = 0
    ) -> None:
        """Rebuild parent-first after restart, without eviction or fake hits."""
        _validate_size(size_bytes)
        if key in self._entries:
            raise ValueError(f"key {key!r} already restored")
        if parent_key is not None and parent_key not in self._entries:
            raise ValueError(f"parent {parent_key!r} must be restored first")
        if not self._fits(len(self) + 1, self._nbytes + size_bytes):
            raise ValueError("cannot restore beyond max_entries or max_bytes")
        self._entries[key] = _Entry(parent_key, size_bytes)
        self._nbytes += size_bytes
        self._mark_used(key)

    def restore_access(self, key: str, access_tick: int) -> None:
        """Restore persisted recency; frequency intentionally starts again at 1."""
        if type(access_tick) is not int:
            raise ValueError("access_tick must be an integer")
        self._entries[key].last_access = access_tick
        self._clock = max(self._clock, access_tick)
        self._entries.move_to_end(key)

    def _fits(self, entries: int, nbytes: int) -> bool:
        return (self._max_entries is None or entries <= self._max_entries) and (
            self._max_bytes is None or nbytes <= self._max_bytes
        )

    def _mark_used(self, key: str) -> None:
        self._clock += 1
        self._entries[key].last_access = self._clock
        self._entries.move_to_end(key)

    def _remove(self, key: str) -> None:
        self._nbytes -= self._entries.pop(key).size

    def _find_victim(self, protected: str | None) -> str | None:
        # Any key named as a parent is internal, so it cannot be evicted.
        parents = {entry.parent for entry in self._entries.values()}
        leaves = (
            key for key in self._entries if key != protected and key not in parents
        )

        def rank(key: str) -> tuple[tuple[float, ...], str]:
            entry = self._entries[key]
            return self._policy(key, entry.frequency, entry.last_access), key

        return min(leaves, key=rank, default=None)


# Existing backends use this name; both policies share the same prefix rules.
PrefixLRUIndex = PrefixIndex


def _validate_size(size_bytes: int) -> None:
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("size_bytes must be a non-negative integer")
