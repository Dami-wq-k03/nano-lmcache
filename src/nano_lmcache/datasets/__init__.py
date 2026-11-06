"""Workload schemas, adapters, and deterministic datasets for cache replay."""

from nano_lmcache.datasets.jsonl import iter_workload_jsonl
from nano_lmcache.datasets.schemas.workload import WorkloadRequest
from nano_lmcache.datasets.synthetic import (
    SYNTHETIC_SCENARIOS,
    generate_synthetic,
)

__all__ = [
    "SYNTHETIC_SCENARIOS",
    "WorkloadRequest",
    "generate_synthetic",
    "iter_workload_jsonl",
]
