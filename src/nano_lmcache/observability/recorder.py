"""Optional bounded recording, online counters, and JSON trace export."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nano_lmcache.observability.context import current_request_id
from nano_lmcache.observability.events import TraceEvent, TraceEventType


class TraceRecorder:
    """Collect ordered :class:`TraceEvent` objects.

    A recorder can be disabled for production or benchmark runs.  Disabled
    recorders accept calls and return ``None`` without allocating events, so a
    cache can keep one optional recorder hook without branching at every call
    site.  A ``request_id`` may be bound at construction time, or supplied to
    each call to :meth:`record` when a recorder is shared by requests.

    ``sample_every=k`` retains sequence 0, k, 2k, ... until ``max_events``
    is reached. This is a prefix sample, not a random sample of the full run.
    Counts include every enabled event, even after the detail cap is reached.
    Defaults preserve full recording. ``max_events=0`` keeps only counters.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        request_id: str | None = None,
        max_events: int | None = None,
        sample_every: int = 1,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise ValueError("request_id must be a non-empty string or None")

        self._enabled = enabled
        self._request_id = request_id
        if max_events is not None and (
            type(max_events) is not int or max_events < 0
        ):
            raise ValueError("max_events must be a non-negative integer or None")
        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("sample_every must be a positive integer")
        self.max_events = max_events
        self.sample_every = sample_every
        self._events: list[TraceEvent] = []
        self._total_events = 0
        self._event_counts: Counter[str] = Counter()
        self._storage_hits: Counter[str] = Counter()

    @property
    def total_events(self) -> int:
        """Number of enabled observations, including omitted detail."""
        return self._total_events

    @property
    def event_counts(self) -> dict[str, int]:
        """Exact counts by event type, before sampling and truncation."""
        return dict(self._event_counts)

    @property
    def storage_hits(self) -> dict[str, int]:
        """Exact successful reads per backend, before detail retention."""
        return dict(self._storage_hits)

    @property
    def enabled(self) -> bool:
        """Whether observations contribute to counters and retained detail."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("enabled must be a boolean")
        self._enabled = value

    @property
    def request_id(self) -> str | None:
        """The default request ID, if one was bound at construction time."""
        return self._request_id

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """A read-only snapshot of recorded events in sequence order."""
        return tuple(self._events)

    def enable(self) -> None:
        """Enable event recording."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event recording without clearing existing events."""
        self._enabled = False

    def record(
        self,
        event_type: str | TraceEventType,
        *,
        request_id: str | None = None,
        chunk_index: int | None = None,
        key: str | None = None,
        parent_key: str | None = None,
        result: str | None = None,
        duration_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent | None:
        """Record one event and return it, or return ``None`` when disabled.

        ``request_id`` overrides the recorder's bound default for this event.
        A request ID is required while recording, which prevents traces from
        silently mixing anonymous requests.
        Enabled calls return the event even if sampling omits its retention.
        """
        if not self._enabled:
            return None

        event_request_id = (
            request_id
            if request_id is not None
            else self._request_id or current_request_id()
        )
        if event_request_id is None:
            raise ValueError(
                "request_id is required; bind one on TraceRecorder or pass it "
                "to record()"
            )

        event = TraceEvent(
            sequence=self._total_events,
            request_id=event_request_id,
            event_type=event_type,
            chunk_index=chunk_index,
            key=key,
            parent_key=parent_key,
            result=result,
            duration_ns=duration_ns,
            metadata=metadata,
        )
        self._total_events += 1
        self._event_counts[event.event_type] += 1
        if event.event_type == "storage_get" and event.result == "hit":
            backend = (event.metadata or {}).get("backend")
            if isinstance(backend, str):
                self._storage_hits[backend] += 1
        if event.sequence % self.sample_every == 0 and (
            self.max_events is None or len(self._events) < self.max_events
        ):
            self._events.append(event)
        return event

    def clear(self) -> None:
        """Discard all events and restart sequence numbering at zero."""
        self._events.clear()
        self._total_events = 0
        self._event_counts.clear()
        self._storage_hits.clear()

    def to_list(self) -> list[dict[str, Any]]:
        """Return a detached list of event dictionaries."""
        return [event.to_dict() for event in self._events]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the trace as a JSON array ordered by sequence."""
        return json.dumps(self.to_list(), indent=indent)

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        encoding: str = "utf-8",
    ) -> None:
        """Write retained events to ``path``; omitted detail is not reconstructed."""
        Path(path).write_text(self.to_json(indent=indent), encoding=encoding)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)
