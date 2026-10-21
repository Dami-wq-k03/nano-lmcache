"""Two storage tiers: a small fast one in front of a large slow one.

This mirrors one of LMCache's core ideas: keep hot chunks in fast memory
and the long tail in a bigger, slower store such as local disk.

Write policy: write-through. Every `put` goes to the slow tier and then to
the fast tier, so the slow tier always has its own copy. The alternative,
write-back (write to the slow tier only when the fast tier evicts), saves
writes but breaks prefix safety in a subtle way: the fast tier evicts
leaves, and when a leaf is demoted its parent may still exist only in the
fast tier, so a prefix-safe slow tier must reject it and the data is lost.
Write-through also means a crash never loses anything that was stored.

Read policy: promote on hit. A slow-tier hit is copied into the fast tier
with its recorded parent, so the fast tier keeps prefix safety too. A
prefix lookup walks a chain root first, so a parent is normally promoted
just before its child; if the fast tier has no room for the parent, it
also refuses the child, and the slow tier keeps serving both.

Each tier enforces the prefix-safety invariant on its own (every entry's
whole ancestor chain is in that tier), so their union does as well.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nano_lmcache.storage.protocol import CacheStorage


@runtime_checkable
class SlowTier(CacheStorage, Protocol):
    """A tier that can report each entry's parent, so hits can be promoted."""

    def __contains__(self, key: object) -> bool: ...

    def parent_of(self, key: str) -> str | None: ...


@runtime_checkable
class FastTier(CacheStorage, Protocol):
    """A tier that can be checked for membership without counting a use."""

    def __contains__(self, key: object) -> bool: ...


@dataclass
class TierStats:
    """Where each `get` was served from."""

    fast_hits: int = 0
    slow_hits: int = 0
    misses: int = 0
    # Slow-tier hits copied into the fast tier. Lower than `slow_hits`
    # when the fast tier refused an entry (no room, or parent missing).
    promotions: int = 0

    @property
    def lookups(self) -> int:
        return self.fast_hits + self.slow_hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Share of gets served by either tier, 0.0 before any get."""
        if self.lookups == 0:
            return 0.0
        return (self.fast_hits + self.slow_hits) / self.lookups

    @property
    def fast_hit_rate(self) -> float:
        """Share of gets served by the fast tier alone."""
        if self.lookups == 0:
            return 0.0
        return self.fast_hits / self.lookups


class TieredStorage:
    """Write-through, promote-on-hit combination of two backends."""

    def __init__(self, *, fast: FastTier, slow: SlowTier) -> None:
        self._fast = fast
        self._slow = slow
        self._tier_stats = TierStats()

    @property
    def fast(self) -> FastTier:
        return self._fast

    @property
    def slow(self) -> SlowTier:
        return self._slow

    @property
    def tier_stats(self) -> TierStats:
        return self._tier_stats

    def __contains__(self, key: object) -> bool:
        """Whether either tier has `key`; does not promote or count a use."""
        return key in self._fast or key in self._slow

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        # Slow tier first: if it raises (for example, a payload it cannot
        # serialize), the fast tier is left untouched.
        self._slow.put(key, value, parent_key=parent_key)
        self._fast.put(key, value, parent_key=parent_key)

    def get(self, key: str) -> object | None:
        value = self._fast.get(key)
        if value is not None:
            self._tier_stats.fast_hits += 1
            return value

        value = self._slow.get(key)
        if value is None:
            self._tier_stats.misses += 1
            return None

        self._tier_stats.slow_hits += 1
        self._fast.put(key, value, parent_key=self._slow.parent_of(key))
        if key in self._fast:
            self._tier_stats.promotions += 1
        return value
