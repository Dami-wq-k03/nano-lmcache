"""Storage interface the prefix cache depends on."""

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
