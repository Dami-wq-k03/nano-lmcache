"""Small policy research baselines, not workload-independent optimizers.

Victim ranking reuses LFU access/leaf state; PrefixIndex still owns the tree.
Reuse distance is an exact reference analyzer, not a scalable trace processor.
Prefetch feedback describes completed windows, never still-in-flight bytes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

from nano_lmcache.storage.eviction_policies import LFUEvictionPolicy


def _nonnegative(*values: float) -> None:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("costs must be finite and non-negative")


@dataclass(frozen=True)
class CacheValue:
    """Known serialized size and comparable per-access latency estimates."""

    nbytes: int
    recompute_ms: float
    load_ms: float

    def __post_init__(self) -> None:
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("nbytes must be a positive integer")
        _nonnegative(self.recompute_ms, self.load_ms)


def admission_score(
    value: CacheValue, *, reuse_probability: float = 1.0, write_ms: float = 0.0
) -> float:
    """Expected saved ms per byte: (p * (recompute - load) - write) / bytes.

    Positive suggests admission, not a guarantee; callers supply their own p.
    This helper does not bypass prefix-safe capacity or missing-parent checks.
    """
    _nonnegative(reuse_probability, write_ms)
    if reuse_probability > 1:
        raise ValueError("reuse_probability must be in [0, 1]")
    return (
        reuse_probability * (value.recompute_ms - value.load_ms) - write_ms
    ) / value.nbytes


class CostAwareEvictionPolicy:
    """Evict the lowest LFU-frequency * saved-ms/byte eligible leaf.

    Metadata is queried at selection, so known costs can change without stale
    priorities. Unknown/non-beneficial values have score zero, with LRU/key
    ties. Selection scans leaves; there is no extra heap or prefix tree.
    Use a factory (lambda: CostAwareEvictionPolicy(values.get)) when injecting
    into storage, to keep a live callback rather than a deep-copied template.
    """

    name = "cost-aware"

    def __init__(self, value_of: Callable[[str], CacheValue | None]) -> None:
        self._value_of = value_of
        self._access = LFUEvictionPolicy()

    def on_insert(self, key: str, *, is_leaf: bool) -> None:
        self._access.on_insert(key, is_leaf=is_leaf)

    def on_touch(self, key: str, *, is_leaf: bool) -> None:
        self._access.on_touch(key, is_leaf=is_leaf)

    def on_write(self, key: str, *, is_leaf: bool) -> None:
        self._access.on_write(key, is_leaf=is_leaf)

    def on_remove(self, key: str) -> None:
        self._access.on_remove(key)

    def on_leaf_status_change(self, key: str, *, is_leaf: bool) -> None:
        self._access.on_leaf_status_change(key, is_leaf=is_leaf)

    def on_restore(self, key: str, *, is_leaf: bool) -> None:
        self._access.on_restore(key, is_leaf=is_leaf)

    def on_restore_access(self, key: str, access_tick: int) -> None:
        self._access.on_restore_access(key, access_tick)

    def frequency_of(self, key: str) -> int:
        return self._access.frequency_of(key)

    @property
    def heap_size(self) -> int:
        return self._access.heap_size

    def choose_victim(self, protected: str | None) -> str | None:
        recency = self._access.last_access

        def rank(key: str) -> tuple[float, int, str]:
            value = self._value_of(key)
            score = max(0.0, admission_score(value)) if value is not None else 0.0
            return (score * self.frequency_of(key), recency[key], key)

        return min(
            (key for key in self._access._leaf_keys if key != protected),
            key=rank,
            default=None,
        )


def reuse_distances(keys: Iterable[Hashable]) -> list[int | None]:
    """Distinct keys since last access; None means cold, immediate reuse is 0.

    MRU stack reference: O(accesses * unique keys) time, O(unique) stack space.
    Distance >= capacity predicts a miss ONLY for fully associative entry-LRU
    on independent keys, not variable-byte or prefix-constrained eviction.
    """
    stack: list[Hashable] = []
    distances: list[int | None] = []
    for key in keys:
        try:
            distance = stack.index(key)
        except ValueError:
            distances.append(None)
        else:
            distances.append(distance)
            stack.pop(distance)
        stack.insert(0, key)
    return distances


class PrefetchDepthTuner:
    """Bounded halve-on-waste/add-one-on-stall feedback heuristic.

    Each observe call resolves a disjoint window: used_bytes is the subset of
    prefetched_bytes actually consumed. Waste is counted in bytes, not pages.
    Depth zero can probe again on a stall with no prefetch; no latency or waste
    guarantee is claimed. The tuner schedules no I/O and owns no page state.
    """

    def __init__(
        self,
        *,
        initial_depth: int = 1,
        max_depth: int = 8,
        waste_tolerance: float = 0.25,
    ) -> None:
        if (
            type(initial_depth) is not int
            or type(max_depth) is not int
            or not 0 <= initial_depth <= max_depth
            or max_depth <= 0
        ):
            raise ValueError("depths must be integers in 0..positive max_depth")
        _nonnegative(waste_tolerance)
        if waste_tolerance > 1:
            raise ValueError("waste_tolerance must be in [0, 1]")
        self.depth = initial_depth
        self.max_depth = max_depth
        self.waste_tolerance = waste_tolerance
        self.prefetched_bytes = 0
        self.used_bytes = 0

    @property
    def wasted_bytes(self) -> int:
        return self.prefetched_bytes - self.used_bytes

    @property
    def waste_ratio(self) -> float:
        return (
            self.wasted_bytes / self.prefetched_bytes if self.prefetched_bytes else 0.0
        )

    def observe(
        self, *, prefetched_bytes: int, used_bytes: int, stall_ms: float = 0.0
    ) -> int:
        if (
            type(prefetched_bytes) is not int
            or type(used_bytes) is not int
            or not 0 <= used_bytes <= prefetched_bytes
        ):
            raise ValueError("bytes must be integers with 0 <= used <= prefetched")
        _nonnegative(stall_ms)
        self.prefetched_bytes += prefetched_bytes
        self.used_bytes += used_bytes
        waste = (
            (prefetched_bytes - used_bytes) / prefetched_bytes
            if prefetched_bytes
            else 0.0
        )
        if prefetched_bytes and waste > self.waste_tolerance:
            self.depth //= 2
        elif stall_ms > 0:
            self.depth = min(self.max_depth, self.depth + 1)
        return self.depth
