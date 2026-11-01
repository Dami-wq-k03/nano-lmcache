"""Small deterministic workloads for teaching and regression tests."""

from collections.abc import Callable

from nano_lmcache.datasets.schemas.workload import WorkloadRequest


def _request(
    scenario: str,
    index: int,
    tokens: tuple[int, ...],
) -> WorkloadRequest:
    return WorkloadRequest.from_token_ids(
        request_id=f"{scenario}-{index}",
        session_id=scenario,
        arrival_time_ms=index,
        model_id="synthetic-model",
        token_ids=tokens,
        metadata={"scenario": scenario},
    )


def cold_cache() -> tuple[WorkloadRequest, ...]:
    """One request that has no reusable predecessor."""
    return (_request("cold_cache", 0, (101, 205, 307, 409)),)


def full_prefix_hit() -> tuple[WorkloadRequest, ...]:
    """Two identical requests, so the second can fully reuse the first."""
    tokens = (101, 205, 307, 409)
    return (
        _request("full_prefix_hit", 0, tokens),
        _request("full_prefix_hit", 1, tokens),
    )


def partial_prefix_hit() -> tuple[WorkloadRequest, ...]:
    """Two shared chunks followed by a divergent chunk."""
    return (
        _request("partial_prefix_hit", 0, (101, 205, 307, 409, 511, 613)),
        _request("partial_prefix_hit", 1, (101, 205, 307, 409, 777, 888)),
    )


def prefix_branching() -> tuple[WorkloadRequest, ...]:
    """Two requests sharing a prefix and branching into separate children."""
    return (
        _request("prefix_branching", 0, (101, 205, 307, 409, 511, 613)),
        _request("prefix_branching", 1, (101, 205, 307, 409, 777, 888)),
        _request("prefix_branching", 2, (101, 205, 307, 409, 999, 1001)),
    )


def lru_eviction() -> tuple[WorkloadRequest, ...]:
    """Distinct prefixes followed by a revisit, suitable for bounded caches."""
    return (
        _request("lru_eviction", 0, (101, 205)),
        _request("lru_eviction", 1, (307, 409)),
        _request("lru_eviction", 2, (511, 613)),
        _request("lru_eviction", 3, (101, 205)),
    )


def incomplete_chunk() -> tuple[WorkloadRequest, ...]:
    """A request with a trailing token span that cannot form a full chunk."""
    return (_request("incomplete_chunk", 0, (101, 205, 307, 409, 511)),)


SYNTHETIC_SCENARIOS: dict[
    str, Callable[[], tuple[WorkloadRequest, ...]]
] = {
    "cold_cache": cold_cache,
    "full_prefix_hit": full_prefix_hit,
    "partial_prefix_hit": partial_prefix_hit,
    "prefix_branching": prefix_branching,
    "lru_eviction": lru_eviction,
    "incomplete_chunk": incomplete_chunk,
}


def generate_synthetic(name: str) -> tuple[WorkloadRequest, ...]:
    """Return a fresh deterministic scenario by name."""
    try:
        factory = SYNTHETIC_SCENARIOS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SYNTHETIC_SCENARIOS))
        raise ValueError(
            f"unknown synthetic scenario {name!r}; choose {choices}"
        ) from exc
    return factory()
