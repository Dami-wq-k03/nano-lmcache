"""Storage interfaces used by the cache and optional fast paths."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheStorage(Protocol):
    """Minimal interface required by the prefix cache.

    Any object with these two methods can serve as a backend; it does not
    need to inherit from anything. This keeps the cache's matching logic
    independent of where payloads actually live (memory, disk, remote...).

    `get` returns None to signal a miss, so None is reserved and cannot be
    stored as a payload.

    `put` also receives `parent_key`, the key of the entry that precedes
    this one in its prefix chain. Backends that evict must use it to avoid
    removing an entry while its descendants remain, since descendants are
    unreachable without their ancestors. Backends that never evict may
    ignore it.
    """

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        """Store `value` under `key`, replacing any existing payload."""
        ...

    def get(self, key: str) -> object | None:
        """Return the payload for `key`, or None on a miss."""
        ...


@runtime_checkable
class BatchReadableStorage(Protocol):
    """Optional capability for reading several ordered keys in one operation.

    Implementations return exactly one result per requested key and preserve
    request order. ``None`` retains the scalar ``get`` meaning of a cache miss.
    Keeping this separate from :class:`CacheStorage` lets simple teaching
    backends implement only the two operations required by the prefix cache.
    """

    def get_many(self, keys: Sequence[str]) -> list[object | None]:
        """Return ordered payloads and misses for all requested keys."""
        ...
