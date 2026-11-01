"""Common request schema for replaying LLM serving workloads."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    """One request in a replayable serving workload.

    ``prompt_tokens`` is the reported prompt length.  ``token_ids`` is only
    populated when the source exposes the actual token sequence.  In contrast,
    ``prefix_blocks`` contains source-specific block identifiers (for example,
    FreeInference's chained 16-token hashes), and must not be treated as token
    IDs.  ``observed_cached_tokens`` is the source's reported cache read count;
    a replay computes its own hit count using its configured cache policy.
    """

    request_id: str
    session_id: str
    arrival_time_ms: int
    model_id: str
    prompt_tokens: int
    output_tokens: int = 0
    token_ids: tuple[int, ...] | None = None
    prefix_blocks: tuple[int, ...] | None = None
    observed_cached_tokens: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "session_id", "model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

        for name in ("arrival_time_ms", "prompt_tokens", "output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if self.token_ids is not None:
            if any(type(token) is not int for token in self.token_ids):
                raise TypeError("token_ids must contain integers")
            if len(self.token_ids) != self.prompt_tokens:
                raise ValueError(
                    "prompt_tokens must equal len(token_ids) when token_ids "
                    "are provided"
                )

        if self.prefix_blocks is not None:
            if any(type(block) is not int or block < 0 for block in self.prefix_blocks):
                raise ValueError("prefix_blocks must contain non-negative integers")

        if self.observed_cached_tokens is not None and (
            type(self.observed_cached_tokens) is not int
            or not 0 <= self.observed_cached_tokens <= self.prompt_tokens
        ):
            raise ValueError(
                "observed_cached_tokens must be between zero and prompt_tokens"
            )

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError("metadata must be a dictionary or None")
            object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_token_ids(
        cls,
        *,
        request_id: str,
        session_id: str,
        arrival_time_ms: int,
        model_id: str,
        token_ids: tuple[int, ...] | list[int],
        output_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkloadRequest":
        """Construct a request whose prompt length comes from its token IDs."""
        tokens = tuple(token_ids)
        return cls(
            request_id=request_id,
            session_id=session_id,
            arrival_time_ms=arrival_time_ms,
            model_id=model_id,
            prompt_tokens=len(tokens),
            output_tokens=output_tokens,
            token_ids=tokens,
            metadata=metadata,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkloadRequest":
        """Build a request from the serialized schema representation."""
        token_ids = value.get("token_ids")
        prefix_blocks = value.get("prefix_blocks")
        return cls(
            request_id=value["request_id"],
            session_id=value["session_id"],
            arrival_time_ms=value["arrival_time_ms"],
            model_id=value["model_id"],
            prompt_tokens=value["prompt_tokens"],
            output_tokens=value.get("output_tokens", 0),
            token_ids=None if token_ids is None else tuple(token_ids),
            prefix_blocks=(
                None if prefix_blocks is None else tuple(prefix_blocks)
            ),
            observed_cached_tokens=value.get("observed_cached_tokens"),
            metadata=value.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this request."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "arrival_time_ms": self.arrival_time_ms,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "token_ids": None if self.token_ids is None else list(self.token_ids),
            "prefix_blocks": (
                None if self.prefix_blocks is None else list(self.prefix_blocks)
            ),
            "observed_cached_tokens": self.observed_cached_tokens,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }
