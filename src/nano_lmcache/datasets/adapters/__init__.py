"""Adapters from public serving datasets to the common workload schema."""

from nano_lmcache.datasets.adapters.freeinference import (
    FreeInferenceAdapter,
    iter_freeinference_jsonl,
)

__all__ = ["FreeInferenceAdapter", "iter_freeinference_jsonl"]
