"""Deterministic synthetic policy traces, with LRU as the control.

Reports hit bytes and MODELLED latency, not wall-clock or hardware results.
The admission mode uses a deliberately simple online p=1-1/access_count;
this is a heuristic, not future trace knowledge or a calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from nano_lmcache.experimental.policy import (
    CacheValue,
    CostAwareEvictionPolicy,
    admission_score,
    reuse_distances,
)
from nano_lmcache.storage.eviction_policies import PolicyLike
from nano_lmcache.storage.memory import MemoryKVCache


@dataclass(frozen=True)
class Access:
    key: str
    value: CacheValue
    parent: str | None = None


def value_for(index: int) -> CacheValue:
    return CacheValue(1024 * (1 + index % 4), 1.0 + index % 8, 0.1)


def make_traces(requests: int, seed: int) -> dict[str, list[Access]]:
    rng = random.Random(seed)
    uniform = []
    shared = []
    low_reuse = []
    system = CacheValue(4096, 12.0, 0.1)
    for index in range(requests):
        key_id = rng.randrange(64)
        uniform.append(Access(f"uniform-{key_id}", value_for(key_id)))
        shared.append(Access("system", system))
        suffix = rng.randrange(8) if rng.random() < 0.8 else 8 + index
        shared.append(Access(f"suffix-{suffix}", value_for(suffix), "system"))
        rare_id = rng.randrange(4) if rng.random() < 0.05 else 4 + index
        low_reuse.append(Access(f"low-{rare_id}", value_for(rare_id)))
    return {
        "uniform": uniform,
        "shared_system_prompt": shared,
        "low_reuse": low_reuse,
    }


def simulate(
    trace: Sequence[Access], *, mode: str, capacity_bytes: int, write_ms: float
) -> dict[str, int | float]:
    values = {access.key: access.value for access in trace}
    policy: PolicyLike = "lru"
    if mode != "lru":
        if mode not in ("cost-aware", "cost-aware+admission"):
            raise ValueError("unknown policy mode")

        def cost_policy() -> CostAwareEvictionPolicy:
            return CostAwareEvictionPolicy(values.get)

        policy = cost_policy
    storage = MemoryKVCache(
        max_bytes=capacity_bytes,
        size_fn=lambda value: cast(CacheValue, value).nbytes,
        eviction_policy=policy,
    )
    counts: Counter[str] = Counter()
    hits = hit_bytes = requested_bytes = write_bytes = skipped = rejected = 0
    latency = 0.0
    for access in trace:
        counts[access.key] += 1
        requested_bytes += access.value.nbytes
        if storage.get(access.key) is not None:
            hits += 1
            hit_bytes += access.value.nbytes
            latency += min(access.value.load_ms, access.value.recompute_ms)
            continue
        latency += access.value.recompute_ms
        probability = 1.0 - 1.0 / counts[access.key]
        if (
            mode == "cost-aware+admission"
            and admission_score(
                access.value, reuse_probability=probability, write_ms=write_ms
            )
            <= 0
        ):
            skipped += 1
            continue
        if storage.try_put(access.key, access.value, parent_key=access.parent):
            write_bytes += access.value.nbytes
            latency += write_ms
        else:
            rejected += 1
    return {
        "accesses": len(trace),
        "hits": hits,
        "hit_bytes": hit_bytes,
        "requested_bytes": requested_bytes,
        "byte_hit_ratio": hit_bytes / requested_bytes if requested_bytes else 0.0,
        "estimated_latency_ms": latency,
        "write_bytes": write_bytes,
        "skipped_admissions": skipped,
        "capacity_or_parent_rejections": rejected,
        "evictions": storage.stats.evictions,
    }


def benchmark(
    *,
    requests: int = 2000,
    seed: int = 7,
    capacity_bytes: int = 16 << 10,
    write_ms: float = 0.15,
) -> dict[str, object]:
    # Validate even when the trace is empty or no admission score is evaluated.
    admission_score(value_for(0), write_ms=write_ms)
    if requests <= 0 or capacity_bytes <= 0:
        raise ValueError("requests and capacity_bytes must be positive")
    results = {}
    for name, trace in make_traces(requests, seed).items():
        distances = reuse_distances(access.key for access in trace)
        histogram = Counter(distance for distance in distances if distance is not None)
        modes = {
            mode: simulate(
                trace, mode=mode, capacity_bytes=capacity_bytes, write_ms=write_ms
            )
            for mode in ("lru", "cost-aware", "cost-aware+admission")
        }
        control = modes["lru"]["estimated_latency_ms"]
        for result in modes.values():
            result["estimated_latency_vs_lru"] = (
                result["estimated_latency_ms"] / control if control else 1.0
            )
        results[name] = {
            "reuse_distance": {
                "cold_accesses": distances.count(None),
                "histogram": {str(k): v for k, v in sorted(histogram.items())},
                "scope": "key stack distance; not a byte/prefix miss prediction",
            },
            "policies": modes,
        }
    return {
        "scope": "synthetic estimates; no hardware measurements or universal guarantee",
        "requests": requests,
        "seed": seed,
        "capacity_bytes": capacity_bytes,
        "model": {
            "load_ms": 0.1,
            "recompute_ms": "1..8; shared root 12",
            "write_ms": write_ms,
            "admission_probability": "online heuristic 1 - 1/access_count",
            "metadata": "fixed size/cost descriptors, not observed future reuse",
        },
        "traces": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity-bytes", type=int, default=16 << 10)
    parser.add_argument("--write-ms", type=float, default=0.15)
    args = parser.parse_args()
    try:
        result = benchmark(
            requests=args.requests,
            seed=args.seed,
            capacity_bytes=args.capacity_bytes,
            write_ms=args.write_ms,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
