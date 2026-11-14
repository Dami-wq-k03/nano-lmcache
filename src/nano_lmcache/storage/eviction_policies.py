"""Eviction-order policies for the prefix-safe storage index.

The prefix index owns the tree and admission rules.  Policies only remember
access state and rank keys that the index tells them are leaves.  This keeps a
new policy from needing a second (and potentially inconsistent) prefix tree.
"""

from __future__ import annotations

import copy
import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast


class EvictionPolicy(Protocol):
    """The small interface between prefix mechanics and victim ordering."""

    @property
    def name(self) -> str:
        """Stable configuration name for this policy."""
        ...

    def on_insert(self, key: str, *, is_leaf: bool) -> None:
        """Record a newly stored key with its initial access state."""
        ...

    def on_touch(self, key: str, *, is_leaf: bool) -> None:
        """Record a real cache hit."""
        ...

    def on_write(self, key: str, *, is_leaf: bool) -> None:
        """Record a successful overwrite without counting a cache hit."""
        ...

    def on_remove(self, key: str) -> None:
        """Forget a key that is no longer in the prefix index."""
        ...

    def on_leaf_status_change(self, key: str, *, is_leaf: bool) -> None:
        """Update candidate eligibility without changing access state."""
        ...

    def on_restore(self, key: str, *, is_leaf: bool) -> None:
        """Restore a key after restart, starting its policy state afresh."""
        ...

    def on_restore_access(self, key: str, access_tick: int) -> None:
        """Restore recency metadata without counting an access."""
        ...

    def choose_victim(self, protected: str | None) -> str | None:
        """Return the best eligible key other than ``protected``."""
        ...

    def frequency_of(self, key: str) -> int:
        """Return a key's frequency, for diagnostics and tests."""
        ...

    @property
    def heap_size(self) -> int:
        """Number of retained candidate records."""
        ...


PolicyFactory = Callable[[], EvictionPolicy]
PolicyLike = str | EvictionPolicy | PolicyFactory


@dataclass
class _AccessState:
    frequency: int
    last_access: int
    generation: int = 0


class _HeapEvictionPolicy(ABC):
    """Shared lazy-heap implementation used by LRU and LFU.

    A candidate contains the policy priority, the key, and a generation.  A
    changed access state makes old records stale; compaction rebuilds the heap
    from current leaves after enough stale records accumulate.  The policy
    receives leaf status from the index but never traverses parent/child links.
    """

    name: str
    _track_unbounded_state = False

    def __init__(self, *, bounded: bool = True) -> None:
        self._bounded = bounded
        self._track_state = bounded or self._track_unbounded_state
        self._states: dict[str, _AccessState] = {}
        self._leaf_keys: set[str] = set()
        self._heap: list[tuple[tuple[int, ...], str, int]] = []
        self._clock = 0

    def on_insert(self, key: str, *, is_leaf: bool) -> None:
        if not self._track_state:
            return
        self._states[key] = _AccessState(frequency=1, last_access=self._tick())
        self._set_leaf(key, is_leaf)
        self._push_candidate(key)

    def on_touch(self, key: str, *, is_leaf: bool) -> None:
        state = self._states.get(key)
        if state is None:
            # This is only possible for an unbounded LRU, whose state is not
            # needed because it can never select a victim.
            return
        state.frequency = self._next_frequency(state.frequency)
        state.last_access = self._tick()
        state.generation += 1
        self._set_leaf(key, is_leaf)
        self._push_candidate(key)

    def on_write(self, key: str, *, is_leaf: bool) -> None:
        """Refresh recency for an overwrite, but do not add LFU frequency."""
        state = self._states.get(key)
        if state is None:
            return
        state.last_access = self._tick()
        state.generation += 1
        self._set_leaf(key, is_leaf)
        self._push_candidate(key)

    def on_remove(self, key: str) -> None:
        self._states.pop(key, None)
        self._leaf_keys.discard(key)

    def on_leaf_status_change(self, key: str, *, is_leaf: bool) -> None:
        if key not in self._states:
            return
        self._set_leaf(key, is_leaf)
        self._push_candidate(key)

    def on_restore(self, key: str, *, is_leaf: bool) -> None:
        if not self._track_state:
            return
        self._states[key] = _AccessState(frequency=1, last_access=self._tick())
        self._set_leaf(key, is_leaf)
        self._push_candidate(key)

    def on_restore_access(self, key: str, access_tick: int) -> None:
        state = self._states.get(key)
        if state is None:
            return
        if not isinstance(access_tick, int) or isinstance(access_tick, bool):
            raise ValueError("access_tick must be an integer")
        self._clock = max(self._clock, access_tick)
        state.last_access = access_tick
        state.generation += 1
        self._push_candidate(key)

    def choose_victim(self, protected: str | None) -> str | None:
        protected_record: tuple[tuple[int, ...], str, int] | None = None
        while self._heap:
            priority, key, generation = heapq.heappop(self._heap)
            state = self._states.get(key)
            if (
                state is None
                or key not in self._leaf_keys
                or state.generation != generation
                or self._priority(state) != priority
            ):
                continue
            record = (priority, key, generation)
            if key == protected:
                protected_record = record
                continue
            if protected_record is not None:
                heapq.heappush(self._heap, protected_record)
            return key
        if protected_record is not None:
            heapq.heappush(self._heap, protected_record)
        return None

    def frequency_of(self, key: str) -> int:
        state = self._states.get(key)
        if state is None:
            raise KeyError(key)
        return state.frequency

    @property
    def heap_size(self) -> int:
        return len(self._heap)

    @property
    def last_access(self) -> dict[str, int]:
        """Access ticks, exposed read-only by convention for diagnostics."""
        return {key: state.last_access for key, state in self._states.items()}

    def _set_leaf(self, key: str, is_leaf: bool) -> None:
        if not self._bounded:
            return
        if is_leaf:
            self._leaf_keys.add(key)
        else:
            self._leaf_keys.discard(key)

    def _push_candidate(self, key: str) -> None:
        if not self._bounded or key not in self._leaf_keys:
            return
        state = self._states[key]
        heapq.heappush(
            self._heap, (self._priority(state), key, state.generation)
        )
        # Keep retained lazy records O(n).  The floor avoids rebuilding tiny
        # heaps; above 2*n, at least n records are stale or duplicates.
        if len(self._heap) > 64 and len(self._heap) > 2 * len(self._states):
            self._heap = [
                (self._priority(state), key, state.generation)
                for key, state in self._states.items()
                if key in self._leaf_keys
            ]
            heapq.heapify(self._heap)

    def _next_frequency(self, frequency: int) -> int:
        return frequency

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _priority(self, state: _AccessState) -> tuple[int, ...]:
        return self._priority_for(state)

    @abstractmethod
    def _priority_for(self, state: _AccessState) -> tuple[int, ...]:
        """Return the numeric portion of a candidate priority."""
        ...


class LRUEvictionPolicy(_HeapEvictionPolicy):
    """Least-recently-used leaf policy with the original heap behavior."""

    name = "lru"

    def _priority_for(self, state: _AccessState) -> tuple[int, ...]:
        return (state.last_access,)


class LFUEvictionPolicy(_HeapEvictionPolicy):
    """Least-frequently-used leaf policy with recency and key tie-breakers."""

    name = "lfu"
    _track_unbounded_state = True

    def _next_frequency(self, frequency: int) -> int:
        return frequency + 1

    def _priority_for(self, state: _AccessState) -> tuple[int, ...]:
        # The heap compares the key after this tuple, giving deterministic key
        # order when both frequency and last-access tick are equal.
        return (state.frequency, state.last_access)


def make_eviction_policy(policy: PolicyLike, *, bounded: bool) -> EvictionPolicy:
    """Create an independent policy instance for one prefix index.

    String configuration is the normal API.  A factory is useful for custom
    policies.  Passing a policy instance is also supported, but it is copied so
    two cache instances cannot accidentally share mutable access state.
    """
    if isinstance(policy, str):
        name = policy.lower()
        if name == "lru":
            return LRUEvictionPolicy(bounded=bounded)
        if name == "lfu":
            return LFUEvictionPolicy(bounded=bounded)
        raise ValueError(
            f"unknown eviction_policy {policy!r}; expected 'lru' or 'lfu'"
        )

    if isinstance(policy, _HeapEvictionPolicy):
        return type(policy)(bounded=bounded)

    candidate: object
    if callable(policy) and not hasattr(policy, "on_insert"):
        candidate = policy()
    else:
        candidate = copy.deepcopy(policy)
    _validate_policy(candidate)
    return cast(EvictionPolicy, candidate)


def _validate_policy(policy: object) -> None:
    required = (
        "name",
        "on_insert",
        "on_touch",
        "on_write",
        "on_remove",
        "on_leaf_status_change",
        "on_restore",
        "on_restore_access",
        "choose_victim",
        "frequency_of",
        "heap_size",
    )
    if any(not hasattr(policy, attribute) for attribute in required):
        raise TypeError("eviction policy does not implement EvictionPolicy")
