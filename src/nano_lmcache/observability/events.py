"""Structured events emitted while a cache request is being processed."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TraceEventType(StrEnum):
    """Event types used by the cache trace format.

    ``event_type`` remains a string in :class:`TraceEvent` so that callers can
    add application-specific events without having to change this package.
    This enum provides stable names for the events understood by the built-in
    visualizers.
    """

    LOOKUP_STARTED = "lookup_started"
    CHUNK_HIT = "chunk_hit"
    CHUNK_MISS = "chunk_miss"
    CHUNK_NOT_CHECKED = "chunk_not_checked"
    CHUNK_INCOMPLETE = "chunk_incomplete"
    LOOKUP_STOPPED = "lookup_stopped"
    LOOKUP_COMPLETED = "lookup_completed"
    CHUNK_STORED = "chunk_stored"
    EVICTION = "eviction"
    STORAGE_GET = "storage_get"
    STORAGE_PUT = "storage_put"
    TIER_HIT = "tier_hit"
    TIER_MISS = "tier_miss"
    PROMOTION = "promotion"
    GPU_EXPORT = "gpu_export"
    CPU_OFFLOAD = "cpu_offload"
    L2_LOOKUP = "l2_lookup"
    L2_RESTORE = "l2_restore"
    GPU_IMPORT = "gpu_import"
    L1_HIT = "l1_hit"
    L2_HIT = "l2_hit"
    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    PREFILL_STEP = "prefill_step"


# Short alias for callers that prefer ``EventType.CHUNK_HIT``.
EventType = TraceEventType


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One ordered, metadata-only observation from a cache operation.

    ``sequence`` is assigned by :class:`~.TraceRecorder` and starts at zero
    for each recorder.  Keys and metadata describe a cache operation; payloads
    such as KV tensors are deliberately outside this schema.
    """

    sequence: int
    request_id: str
    event_type: str
    chunk_index: int | None = None
    key: str | None = None
    parent_key: str | None = None
    result: str | None = None
    duration_ns: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")

        if isinstance(self.event_type, TraceEventType):
            object.__setattr__(self, "event_type", self.event_type.value)
        elif not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("event_type must be a non-empty string")

        if self.chunk_index is not None and (
            type(self.chunk_index) is not int or self.chunk_index < 0
        ):
            raise ValueError("chunk_index must be a non-negative integer or None")
        if self.duration_ns is not None and (
            type(self.duration_ns) is not int or self.duration_ns < 0
        ):
            raise ValueError("duration_ns must be a non-negative integer or None")

        for name in ("key", "parent_key", "result"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError("metadata must be a dictionary or None")
            # A shallow copy prevents changes to the caller's dictionary from
            # silently changing an event that has already been recorded.
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of this event."""
        return {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "event_type": self.event_type,
            "chunk_index": self.chunk_index,
            "key": self.key,
            "parent_key": self.parent_key,
            "result": self.result,
            "duration_ns": self.duration_ns,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }
