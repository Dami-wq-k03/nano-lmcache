"""Logical KV pages and their physical replicas."""

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from .layout import KVLayout


class PageLocation(StrEnum):
    """A physical tier that can hold a KV page replica."""

    GPU = "gpu"
    CPU = "cpu"
    PINNED_CPU = "pinned_cpu"
    PINNED = "pinned_cpu"
    NVME = "nvme"
    DISK = "nvme"
    REMOTE = "remote"


class PageState(StrEnum):
    """Lifecycle state of one physical page replica."""

    RESERVED = "reserved"
    TRANSFERRING = "transferring"
    READY = "ready"
    EVICTING = "evicting"
    EVICTED = "evicted"
    FAILED = "failed"


PhysicalPageId = tuple[PageLocation, str | int]

_ALLOWED_TRANSITIONS: dict[PageState, frozenset[PageState]] = {
    PageState.RESERVED: frozenset(
        {PageState.TRANSFERRING, PageState.READY, PageState.FAILED,
         PageState.EVICTED}
    ),
    PageState.TRANSFERRING: frozenset(
        {PageState.READY, PageState.FAILED, PageState.EVICTING}
    ),
    PageState.READY: frozenset(
        {PageState.TRANSFERRING, PageState.EVICTING, PageState.FAILED}
    ),
    PageState.EVICTING: frozenset({PageState.EVICTED, PageState.FAILED}),
    PageState.FAILED: frozenset({PageState.RESERVED, PageState.EVICTED}),
    PageState.EVICTED: frozenset(),
}


@dataclass(slots=True)
class KVPageRef:
    """Metadata for one physical replica of a logical KV page."""

    logical_key: str
    page_id: str | int
    token_range: tuple[int, int]
    location: PageLocation
    layout: KVLayout
    compression: str = "none"
    generation: int = 0
    state: PageState = PageState.RESERVED
    ready_event: object | None = None

    def __post_init__(self) -> None:
        if not self.logical_key:
            raise ValueError("logical_key must not be empty")
        if isinstance(self.page_id, bool) or not isinstance(
            self.page_id, (str, int)
        ):
            raise ValueError("page_id must be a string or integer")
        if isinstance(self.page_id, str) and not self.page_id:
            raise ValueError("page_id must not be empty")
        if len(self.token_range) != 2:
            raise ValueError("token_range must contain start and end")
        start, end = self.token_range
        if start < 0 or end <= start or end - start > self.layout.page_tokens:
            raise ValueError(
                "token_range must be non-empty and no larger than one page"
            )
        if self.generation < 0:
            raise ValueError("generation must not be negative")
        if not self.compression:
            raise ValueError("compression must not be empty")
        self.location = PageLocation(self.location)
        self.state = PageState(self.state)

    @property
    def identity(self) -> PhysicalPageId:
        """Stable identity of the physical allocation within a tier."""
        return self.location, self.page_id

    @property
    def is_ready(self) -> bool:
        return self.state is PageState.READY

    def transition(self, state: PageState, *, ready_event: object = None) -> None:
        """Move to a legal lifecycle state.

        A page reference is never reused after eviction. A caller that needs a
        new physical allocation creates a new reference, which prevents stale
        transfers from silently publishing an old page.
        """
        target = PageState(state)
        if target is self.state:
            if ready_event is not None:
                self.ready_event = ready_event
            return
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"cannot transition page from {self.state.value} "
                f"to {target.value}"
            )
        if ready_event is not None:
            self.ready_event = ready_event
        self.state = target
        if target in (PageState.FAILED, PageState.EVICTED):
            self.ready_event = None

    def begin_transfer(self, ready_event: object | None = None) -> None:
        self.transition(PageState.TRANSFERRING, ready_event=ready_event)

    def mark_ready(self, ready_event: object | None = None) -> None:
        self.transition(PageState.READY, ready_event=ready_event)

    def mark_failed(self) -> None:
        self.transition(PageState.FAILED)

    def begin_eviction(self) -> None:
        self.transition(PageState.EVICTING)

    def mark_evicted(self) -> None:
        self.transition(PageState.EVICTED)


@dataclass(slots=True)
class KVPageDirectory:
    """Map one logical page to independently managed physical replicas."""

    _pages: dict[str, dict[PhysicalPageId, KVPageRef]] = field(default_factory=dict)
    _authoritative: dict[str, PhysicalPageId] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def register(self, ref: KVPageRef, *, authoritative: bool = False) -> None:
        """Register a replica and optionally make it the source of truth."""
        with self._lock:
            if ref.state is PageState.EVICTED:
                raise ValueError("cannot register an evicted page")
            if authoritative and not ref.is_ready:
                raise ValueError("authoritative replica must be ready")

            bucket = self._pages.setdefault(ref.logical_key, {})
            existing = bucket.get(ref.identity)
            if existing is not None:
                if existing is not ref:
                    raise ValueError("physical page identity is already registered")
                if authoritative:
                    self._authoritative[ref.logical_key] = ref.identity
                return

            self._check_compatible(ref, bucket.values())
            bucket[ref.identity] = ref
            if authoritative or (
                ref.is_ready and ref.logical_key not in self._authoritative
            ):
                self._authoritative[ref.logical_key] = ref.identity

    def get(
        self,
        logical_key: str,
        *,
        location: PageLocation | None = None,
        page_id: str | int | None = None,
    ) -> KVPageRef | None:
        """Return a ready replica, preferring the authoritative one."""
        with self._lock:
            bucket = self._pages.get(logical_key)
            if bucket is None:
                return None

            if location is not None:
                location = PageLocation(location)
            preferred = self._authoritative.get(logical_key)
            if location is None and page_id is None and preferred is not None:
                ref = bucket.get(preferred)
                if ref is not None and ref.is_ready:
                    return ref

            for ref in bucket.values():
                if not ref.is_ready:
                    continue
                if location is not None and ref.location is not location:
                    continue
                if page_id is not None and ref.page_id != page_id:
                    continue
                if location is None and page_id is None:
                    self._authoritative[logical_key] = ref.identity
                return ref

            if location is None and page_id is None:
                self._authoritative.pop(logical_key, None)
            return None

    def replicas(self, logical_key: str) -> tuple[KVPageRef, ...]:
        """Return all non-evicted physical replicas for a logical page."""
        with self._lock:
            bucket = self._pages.get(logical_key, {})
            return tuple(
                ref for ref in bucket.values() if ref.state is not PageState.EVICTED
            )

    def authoritative(self, logical_key: str) -> KVPageRef | None:
        """Return the current source-of-truth replica, if one is ready."""
        return self.get(logical_key)

    def set_authoritative(self, ref: KVPageRef) -> None:
        """Select a registered ready replica as the source of truth."""
        with self._lock:
            bucket = self._pages.get(ref.logical_key)
            if bucket is None or bucket.get(ref.identity) is not ref:
                raise ValueError("authoritative replica must be registered")
            if not ref.is_ready:
                raise ValueError("authoritative replica must be ready")
            self._authoritative[ref.logical_key] = ref.identity

    def remove(self, ref: KVPageRef) -> None:
        """Forget one physical replica and promote a ready copy if needed."""
        with self._lock:
            bucket = self._pages.get(ref.logical_key)
            if bucket is None or bucket.get(ref.identity) is not ref:
                raise KeyError(ref.identity)
            del bucket[ref.identity]
            if not bucket:
                self._pages.pop(ref.logical_key)
                self._authoritative.pop(ref.logical_key, None)
                return
            if self._authoritative.get(ref.logical_key) == ref.identity:
                self._authoritative.pop(ref.logical_key, None)
                self.get(ref.logical_key)

    @property
    def logical_page_count(self) -> int:
        with self._lock:
            return len(self._pages)

    @property
    def replica_count(self) -> int:
        with self._lock:
            return sum(len(bucket) for bucket in self._pages.values())

    @staticmethod
    def _check_compatible(
        ref: KVPageRef,
        existing: Iterable[KVPageRef],
    ) -> None:
        for other in existing:
            if other.layout != ref.layout:
                raise ValueError("replicas for one logical page must share a layout")
            if other.token_range != ref.token_range:
                raise ValueError(
                    "replicas for one logical page must share a token_range"
                )
            if other.generation != ref.generation:
                raise ValueError(
                    "replicas for one logical page must share a generation"
                )


# Names that read naturally at call sites while retaining the KV-prefixed API.
KVPageState = PageState
KVPageLocation = PageLocation
