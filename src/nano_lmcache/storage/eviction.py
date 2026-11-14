"""Prefix-safe admission mechanics with pluggable leaf eviction policies.

The index owns the prefix tree, capacity checks, byte accounting, and
protected ancestor chain.  An :class:`EvictionPolicy` only ranks keys that
this index reports as leaves.  Memory, disk, and CPU-offload backends therefore
share the same prefix-safety mechanism while choosing an eviction order.

The historical ``PrefixLRUIndex`` name remains as a compatibility alias.  Its
default policy is still LRU; pass ``eviction_policy="lfu"`` for LFU.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from nano_lmcache.storage.eviction_policies import (
    EvictionPolicy,
    PolicyLike,
    make_eviction_policy,
)
from nano_lmcache.storage.stats import StorageStats


@dataclass(frozen=True)
class Admission:
    """What a backend must do to make room for one new entry."""

    admitted: bool
    # Keys whose payloads the backend must delete, oldest/best first.
    evicted: tuple[str, ...]


class PrefixIndex:
    """Track prefix relationships and delegate victim ordering to a policy.

    Backends report ``size_bytes``, which defaults to zero.  Both entry and
    byte limits are enforced when supplied.  Only leaves can be removed, and
    an incoming key's parent (or the key itself for an overwrite) is protected
    for the duration of admission.
    """

    def __init__(
        self,
        max_entries: int | None = None,
        *,
        max_bytes: int | None = None,
        eviction_policy: PolicyLike = "lru",
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
        self._policy = make_eviction_policy(
            eviction_policy, bounded=self._bounded
        )
        self._sizes: dict[str, int] = {}
        self._nbytes = 0
        # Order is recency for the public compatibility API: first is oldest.
        # The value is the key's parent (None for a chain root).
        self._parents: OrderedDict[str, str | None] = OrderedDict()
        self._child_counts: dict[str, int] = {}
        self._stats = StorageStats()

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when unbounded."""
        return self._max_entries

    @property
    def max_bytes(self) -> int | None:
        """Backend-reported byte limit, or None when unbounded."""
        return self._max_bytes

    @property
    def nbytes(self) -> int:
        """Sum of the recorded entry sizes."""
        return self._nbytes

    @property
    def stats(self) -> StorageStats:
        """Eviction and rejection totals."""
        return self._stats

    @property
    def eviction_policy(self) -> str:
        """Name of the policy used by this index."""
        return self._policy.name

    @property
    def policy(self) -> EvictionPolicy:
        """The independent policy instance owned by this index."""
        return self._policy

    @property
    def _leaf_heap(self) -> list[tuple[tuple[int, ...], str, int]]:
        """Compatibility view used by heap-bound tests and diagnostics."""
        heap = getattr(self._policy, "_heap", None)
        if heap is None:
            return []
        return cast(list[tuple[tuple[int, ...], str, int]], heap)

    @property
    def _last_access(self) -> dict[str, int]:
        """Compatibility view of policy access ticks."""
        last_access = getattr(self._policy, "last_access", None)
        if last_access is None:
            return {}
        return cast(dict[str, int], last_access)

    def __len__(self) -> int:
        return len(self._parents)

    def __contains__(self, key: object) -> bool:
        return key in self._parents

    def keys_by_recency(self) -> Iterator[str]:
        """Keys from least to most recently used/written."""
        return iter(self._parents)

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for ``key``, which must be present."""
        return self._parents[key]

    def frequency_of(self, key: str) -> int:
        """Return the policy frequency for a stored key."""
        return self._policy.frequency_of(key)

    def touch(self, key: str) -> None:
        """Mark ``key`` as most recently used after a real cache hit."""
        self._parents.move_to_end(key)
        self._policy.on_touch(key, is_leaf=self._is_leaf(key))

    def discard_subtree(self, key: str) -> tuple[str, ...]:
        """Remove ``key`` and descendants, returning children first.

        This is for a backend that discovers an entry is unusable, rather than
        capacity-driven eviction.  It deliberately does not count as an
        eviction.
        """
        if key not in self._parents:
            return ()

        children: dict[str, list[str]] = {}
        for child, parent in self._parents.items():
            if parent is not None:
                children.setdefault(parent, []).append(child)

        discarded: list[str] = []
        pending = [(key, False)]
        while pending:
            current, visited = pending.pop()
            if visited:
                discarded.append(current)
                continue
            pending.append((current, True))
            pending.extend((child, False) for child in children.get(current, ()))

        for victim in discarded:
            self._remove(victim)
        return tuple(discarded)

    def admit(
        self, key: str, parent_key: str | None, *, size_bytes: int = 0
    ) -> Admission:
        """Decide whether ``key`` may be stored and what must be removed."""
        _validate_size(size_bytes)
        present = key in self._parents
        # Re-storing a present key cannot change its place in the chain.
        if present:
            parent_key = self._parents[key]

        if parent_key is not None and parent_key not in self._parents:
            self._stats.rejected_missing_parent += 1
            return Admission(admitted=False, evicted=())

        added_entries = 0 if present else 1
        added_bytes = size_bytes - self._sizes.get(key, 0)
        if not self._fits(len(self) + added_entries, self._nbytes + added_bytes):
            # Preflight the smallest required surviving chain before evicting.
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
            # The preflight proved that an unprotected leaf must exist here.
            assert victim is not None
            self._remove(victim)
            self._stats.evictions += 1
            evicted.append(victim)

        if present:
            self._sizes[key] = size_bytes
            self._nbytes += added_bytes
            self._move_after_write(key)
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
        # Remove leaves first so child counts stay valid.
        victims = tuple(reversed(subtree))
        for victim in victims:
            self._remove(victim)
            self._stats.evictions += 1
        return victims

    def restore(
        self,
        key: str,
        parent_key: str | None,
        *,
        size_bytes: int = 0,
    ) -> None:
        """Re-add an existing entry without evicting or changing stats.

        A restored entry starts with policy frequency 1.  A backend may then
        call :meth:`restore_access` with persisted recency metadata; that
        updates ordering without replaying fake cache hits.
        """
        _validate_size(size_bytes)
        if parent_key is not None and parent_key not in self._parents:
            raise ValueError(f"parent {parent_key!r} must be restored first")
        if self._max_entries is not None and len(self) >= self._max_entries:
            raise ValueError("cannot restore more entries than max_entries")
        if self._max_bytes is not None and self._nbytes + size_bytes > self._max_bytes:
            raise ValueError("cannot restore more bytes than max_bytes")

        self._add(key, parent_key, size_bytes, restored=True)

    def restore_access(self, key: str, access_tick: int) -> None:
        """Restore recency without increasing LFU frequency."""
        self._parents.move_to_end(key)
        self._policy.on_restore_access(key, access_tick)

    @property
    def _bounded(self) -> bool:
        return self._max_entries is not None or self._max_bytes is not None

    def _fits(self, entries: int, nbytes: int) -> bool:
        return (self._max_entries is None or entries <= self._max_entries) and (
            self._max_bytes is None or nbytes <= self._max_bytes
        )

    def _add(
        self,
        key: str,
        parent_key: str | None,
        size_bytes: int,
        *,
        restored: bool = False,
    ) -> None:
        if parent_key is not None and self._child_counts.get(parent_key, 0) == 0:
            self._policy.on_leaf_status_change(parent_key, is_leaf=False)
        self._parents[key] = parent_key
        self._sizes[key] = size_bytes
        self._nbytes += size_bytes
        if parent_key is not None:
            self._child_counts[parent_key] = (
                self._child_counts.get(parent_key, 0) + 1
            )
        if restored:
            self._policy.on_restore(key, is_leaf=True)
        else:
            self._policy.on_insert(key, is_leaf=True)

    def _remove(self, key: str) -> None:
        parent_key = self._parents.pop(key)
        self._nbytes -= self._sizes.pop(key)
        self._policy.on_remove(key)
        self._child_counts.pop(key, None)
        if parent_key is not None:
            self._child_counts[parent_key] -= 1
            if self._child_counts[parent_key] == 0:
                # Exposure is not an access.  The policy retains the parent's
                # frequency and last-access state while making it eligible.
                self._policy.on_leaf_status_change(parent_key, is_leaf=True)

    def _move_after_write(self, key: str) -> None:
        self._parents.move_to_end(key)
        self._policy.on_write(key, is_leaf=self._is_leaf(key))

    def _is_leaf(self, key: str) -> bool:
        return self._child_counts.get(key, 0) == 0

    def _find_victim(self, protected: str | None) -> str | None:
        """Choose the best eligible leaf, excluding ``protected``."""
        return self._policy.choose_victim(protected)


# Backward-compatible public name used by existing callers and tests.  The
# implementation is now generic even though the old name says LRU.
PrefixLRUIndex = PrefixIndex


def _validate_size(size_bytes: int) -> None:
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise ValueError("size_bytes must be a non-negative integer")
