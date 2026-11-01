"""Opt-in storage tracing for memory, disk, tiered, and tensor backends."""

import time
from collections.abc import Iterator
from typing import Any, cast

from nano_lmcache.observability.recorder import TraceRecorder
from nano_lmcache.storage.protocol import CacheStorage


class TracingStorage:
    """Wrap a storage backend and emit storage/tier/eviction events.

    The wrapper preserves the ``CacheStorage`` interface and does not require
    a backend change.  Backends that expose ``keys()`` provide exact eviction
    victims; otherwise the wrapper still records operation and counter events.
    """

    def __init__(
        self,
        backend: CacheStorage,
        recorder: TraceRecorder,
        *,
        name: str = "storage",
        request_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._recorder = recorder
        self._name = name
        self._request_id = request_id or recorder.request_id or "storage"
        self._known_keys: set[str] = set()

    @property
    def backend(self) -> CacheStorage:
        return self._backend

    def __contains__(self, key: object) -> bool:
        return key in cast(Any, self._backend)

    def __len__(self) -> int:
        return len(self._backend)  # type: ignore[arg-type]

    def parent_of(self, key: str) -> str | None:
        backend = cast(Any, self._backend)
        return cast(str | None, backend.parent_of(key))

    def keys(self) -> Iterator[str]:
        return iter(self._snapshot_keys())

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        self.try_put(key, value, parent_key=parent_key)

    def try_put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> bool:
        before = self._snapshot_keys()
        started = time.perf_counter_ns()
        try:
            method = getattr(self._backend, "try_put", None)
            if method is None:
                self._backend.put(key, value, parent_key=parent_key)
                admitted = True
            else:
                admitted = bool(
                    method(key, value, parent_key=parent_key)
                )
        except Exception as exc:
            self._record(
                "storage_put",
                key=key,
                result="error",
                duration_ns=time.perf_counter_ns() - started,
                metadata={"backend": self._name, "error": type(exc).__name__},
            )
            raise

        after = self._snapshot_keys()
        self._record(
            "storage_put",
            key=key,
            result="admitted" if admitted else "rejected",
            duration_ns=time.perf_counter_ns() - started,
            metadata={"backend": self._name, "parent_key": parent_key},
        )
        for victim in sorted(before - after):
            self._record(
                "eviction",
                key=victim,
                result="capacity",
                metadata={"backend": self._name},
            )
        return admitted

    def get(self, key: str) -> object | None:
        before_tier = self._tier_counters()
        started = time.perf_counter_ns()
        value = self._backend.get(key)
        self._record(
            "storage_get",
            key=key,
            result="hit" if value is not None else "miss",
            duration_ns=time.perf_counter_ns() - started,
            metadata={"backend": self._name},
        )
        self._record_tier_changes(before_tier)
        return value

    def invalidate(self, key: str) -> None:
        before = self._snapshot_keys()
        cast(Any, self._backend).invalidate(key)
        for victim in sorted(before - self._snapshot_keys()):
            self._record(
                "eviction",
                key=victim,
                result="invalidate",
                metadata={"backend": self._name},
            )

    def _snapshot_keys(self) -> set[str]:
        method = getattr(self._backend, "keys", None)
        if method is not None:
            keys = set(method())
            self._known_keys = keys
            return keys
        backend = cast(Any, self._backend)
        current = {key for key in self._known_keys if key in backend}
        self._known_keys = current
        return current

    def _tier_counters(self) -> tuple[int, int, int, int] | None:
        stats = getattr(self._backend, "tier_stats", None)
        if stats is None:
            return None
        return (
            stats.fast_hits,
            stats.slow_hits,
            stats.misses,
            stats.promotions,
        )

    def _record_tier_changes(
        self, before: tuple[int, int, int, int] | None
    ) -> None:
        if before is None:
            return
        after = self._tier_counters()
        if after is None:
            return
        if after[0] > before[0]:
            self._record("tier_hit", result="fast", metadata={"backend": self._name})
        if after[1] > before[1]:
            self._record("tier_hit", result="slow", metadata={"backend": self._name})
        if after[2] > before[2]:
            self._record("tier_miss", result="miss", metadata={"backend": self._name})
        if after[3] > before[3]:
            self._record(
                "promotion",
                result="slow_to_fast",
                metadata={"backend": self._name},
            )

    def _record(
        self,
        event_type: str,
        *,
        key: str | None = None,
        result: str | None = None,
        duration_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._recorder.enabled:
            return
        self._recorder.record(
            event_type,
            request_id=self._request_id,
            key=key,
            result=result,
            duration_ns=duration_ns,
            metadata=metadata,
        )
