"""Small policy-plane primitives for load/recompute and prefetch decisions."""

from dataclasses import dataclass
from enum import StrEnum


class LoadDecision(StrEnum):
    """Action selected by the load-versus-recompute estimator."""

    LOAD = "load"
    RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Comparable wall-clock estimates for one cached page."""

    tier: str
    token_count: int
    nbytes: int
    load_ms: float
    recompute_ms: float

    @property
    def decision(self) -> LoadDecision:
        return (
            LoadDecision.LOAD
            if self.load_ms <= self.recompute_ms
            else LoadDecision.RECOMPUTE
        )

    @property
    def savings_ms(self) -> float:
        """Positive value means loading is faster than recomputing."""
        return self.recompute_ms - self.load_ms


class RecomputeLoadCostModel:
    """Estimate whether moving a page is cheaper than recomputing it.

    The model is intentionally transparent: a fixed per-tier latency plus
    bytes divided by effective bandwidth is compared with a linear prefill
    cost. Callers can replace the measurements without changing cache logic.
    """

    def __init__(
        self,
        *,
        recompute_ms_per_token: float,
        bandwidth_gbps: dict[str, float],
        fixed_latency_ms: dict[str, float] | None = None,
    ) -> None:
        if recompute_ms_per_token <= 0:
            raise ValueError("recompute_ms_per_token must be positive")
        if not bandwidth_gbps:
            raise ValueError("bandwidth_gbps must not be empty")
        if any(value <= 0 for value in bandwidth_gbps.values()):
            raise ValueError("bandwidth values must be positive")
        fixed = fixed_latency_ms or {}
        if any(value < 0 for value in fixed.values()):
            raise ValueError("fixed latencies must not be negative")
        unknown = set(fixed) - set(bandwidth_gbps)
        if unknown:
            raise ValueError(f"fixed latency has unknown tiers: {sorted(unknown)}")
        self.recompute_ms_per_token = recompute_ms_per_token
        self.bandwidth_gbps = dict(bandwidth_gbps)
        self.fixed_latency_ms = {tier: fixed.get(tier, 0.0) for tier in bandwidth_gbps}

    def estimate(self, *, tier: str, token_count: int, nbytes: int) -> CostEstimate:
        """Estimate costs for one page from ``tier``."""
        if tier not in self.bandwidth_gbps:
            raise ValueError(f"unknown tier {tier!r}")
        if token_count <= 0 or nbytes < 0:
            raise ValueError("token_count must be positive and nbytes non-negative")
        load_ms = self.fixed_latency_ms[tier] + (
            nbytes / (self.bandwidth_gbps[tier] * 1_000_000_000) * 1_000
        )
        return CostEstimate(
            tier=tier,
            token_count=token_count,
            nbytes=nbytes,
            load_ms=load_ms,
            recompute_ms=token_count * self.recompute_ms_per_token,
        )


class PrefetchState(StrEnum):
    """State of one logical-page prefetch request."""

    PLANNED = "planned"
    IN_FLIGHT = "in_flight"
    READY = "ready"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PrefetchRequest:
    """A prefetch request and its current state."""

    logical_key: str
    tier: str
    state: PrefetchState = PrefetchState.PLANNED
    error: str | None = None


class PrefetchStateMachine:
    """Track prefetch ownership without allocating duplicate transfers."""

    def __init__(self) -> None:
        self._requests: dict[str, PrefetchRequest] = {}

    def plan(self, logical_key: str, *, tier: str) -> PrefetchRequest:
        """Create or return an existing non-terminal request."""
        if not logical_key or not tier:
            raise ValueError("logical_key and tier must not be empty")
        current = self._requests.get(logical_key)
        if current is not None and current.state in (
            PrefetchState.PLANNED,
            PrefetchState.IN_FLIGHT,
            PrefetchState.READY,
        ):
            return current
        request = PrefetchRequest(logical_key, tier)
        self._requests[logical_key] = request
        return request

    def start(self, logical_key: str) -> PrefetchRequest:
        return self._transition(logical_key, PrefetchState.IN_FLIGHT)

    def ready(self, logical_key: str) -> PrefetchRequest:
        return self._transition(logical_key, PrefetchState.READY)

    def cancel(self, logical_key: str) -> PrefetchRequest:
        return self._transition(logical_key, PrefetchState.CANCELLED)

    def fail(self, logical_key: str, error: str) -> PrefetchRequest:
        if not error:
            raise ValueError("error must not be empty")
        request = self._transition(logical_key, PrefetchState.FAILED)
        failed = PrefetchRequest(
            request.logical_key, request.tier, request.state, error
        )
        self._requests[logical_key] = failed
        return failed

    def get(self, logical_key: str) -> PrefetchRequest | None:
        return self._requests.get(logical_key)

    @property
    def active(self) -> tuple[PrefetchRequest, ...]:
        return tuple(
            request
            for request in self._requests.values()
            if request.state in (PrefetchState.PLANNED, PrefetchState.IN_FLIGHT)
        )

    def _transition(
        self, logical_key: str, target: PrefetchState
    ) -> PrefetchRequest:
        request = self._requests.get(logical_key)
        if request is None:
            raise KeyError(logical_key)
        allowed = {
            PrefetchState.PLANNED: {PrefetchState.IN_FLIGHT, PrefetchState.CANCELLED},
            PrefetchState.IN_FLIGHT: {
                PrefetchState.READY,
                PrefetchState.CANCELLED,
                PrefetchState.FAILED,
            },
            PrefetchState.READY: set(),
            PrefetchState.CANCELLED: set(),
            PrefetchState.FAILED: set(),
        }
        if target not in allowed[request.state]:
            raise ValueError(
                f"cannot transition prefetch from {request.state.value} "
                f"to {target.value}"
            )
        updated = PrefetchRequest(request.logical_key, request.tier, target)
        self._requests[logical_key] = updated
        return updated
