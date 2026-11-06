"""Adapter for the public FreeInference Agentic Trace format.

The adapter intentionally uses only the standard library.  It accepts raw
JSONL session records, which makes small samples usable without installing
Hugging Face, pandas, or PyArrow.  The much larger Parquet release can be
converted to the same mapping shape by an optional application-level reader.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from nano_lmcache.datasets.schemas.workload import WorkloadRequest


class FreeInferenceAdapter:
    """Convert FreeInference session records into workload requests."""

    block_tokens = 16

    def __init__(self, *, include_nested_sessions: bool = True) -> None:
        self.include_nested_sessions = include_nested_sessions

    def adapt_session(
        self, session: Mapping[str, Any]
    ) -> tuple[WorkloadRequest, ...]:
        """Adapt one top-level session and any nested subagent sessions."""
        requests: list[WorkloadRequest] = []
        self._collect_session(session, requests)
        requests.sort(key=lambda item: (item.arrival_time_ms, item.request_id))
        return tuple(requests)

    def iter_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        limit: int | None = None,
    ) -> Iterator[WorkloadRequest]:
        """Adapt session mappings while preserving input record order."""
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        if limit == 0:
            return

        emitted = 0
        for session in records:
            for request in self.adapt_session(session):
                if limit is not None and emitted >= limit:
                    return
                yield request
                emitted += 1
                if emitted == limit:
                    return

    def _collect_session(
        self,
        session: Mapping[str, Any],
        output: list[WorkloadRequest],
    ) -> None:
        session_id = _required_string(session, "session_id")
        requests = session.get("requests")
        if not isinstance(requests, list):
            raise ValueError(f"session {session_id!r} has no requests list")

        for raw_request in requests:
            if not isinstance(raw_request, Mapping):
                raise ValueError(
                    f"session {session_id!r} contains a non-object request"
                )
            output.append(self._adapt_request(session, raw_request))

            if not self.include_nested_sessions:
                continue
            for tool_call in raw_request.get("tool_calls", []):
                if not isinstance(tool_call, Mapping):
                    continue
                nested = tool_call.get("subagent")
                if isinstance(nested, Mapping):
                    self._collect_session(nested, output)

    def _adapt_request(
        self,
        session: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> WorkloadRequest:
        session_id = _required_string(session, "session_id")
        step = _required_int(request, "step", minimum=0)
        request_id = f"{session_id}:{step}"
        provider_tokens = request.get("provider_tokens")
        if not isinstance(provider_tokens, Mapping):
            raise ValueError(f"request {request_id!r} has no provider_tokens object")

        prompt_tokens = _required_int(provider_tokens, "prompt", minimum=0)
        completion_value = _optional_int(
            provider_tokens, "completion", default=0
        )
        completion_tokens = 0 if completion_value is None else completion_value
        cached_tokens = _optional_int(provider_tokens, "cached", default=None)
        blocks = request.get("block_ids")
        prefix_blocks = None
        if blocks is not None:
            if not isinstance(blocks, (list, tuple)):
                raise ValueError(f"request {request_id!r} has invalid block_ids")
            prefix_blocks = tuple(
                _non_negative_int(block, "block_ids") for block in blocks
            )

        metadata: dict[str, Any] = {
            "source": "freeinference_agentic_trace",
            "block_tokens": self.block_tokens,
            "provider": request.get("provider"),
            "step": step,
            "turn_index": request.get("turn_index"),
            "step_trigger": request.get("step_trigger"),
            "mutation": request.get("mutation"),
        }
        model_id = _required_string(request, "model")
        return WorkloadRequest(
            request_id=request_id,
            session_id=session_id,
            arrival_time_ms=_required_int(request, "start_ms", minimum=0),
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            prefix_blocks=prefix_blocks,
            observed_cached_tokens=cached_tokens,
            metadata=metadata,
        )


def iter_freeinference_jsonl(
    path: str | Path,
    *,
    include_nested_sessions: bool = True,
    limit: int | None = None,
) -> Iterator[WorkloadRequest]:
    """Stream requests from raw FreeInference session JSONL."""
    adapter = FreeInferenceAdapter(
        include_nested_sessions=include_nested_sessions,
    )

    def records() -> Iterator[Mapping[str, Any]]:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"line {line_number} is not a JSON object")
                yield value

    yield from adapter.iter_records(records(), limit=limit)


def _required_string(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"missing non-empty string field {name!r}")
    return item


def _required_int(
    value: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:
    item = value.get(name)
    if type(item) is not int or item < minimum:
        raise ValueError(f"field {name!r} must be an integer >= {minimum}")
    return item


def _optional_int(
    value: Mapping[str, Any],
    name: str,
    *,
    default: int | None,
) -> int | None:
    item = value.get(name, default)
    if item is None:
        return None
    if type(item) is not int or item < 0:
        raise ValueError(f"field {name!r} must be a non-negative integer or null")
    return item


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return value
