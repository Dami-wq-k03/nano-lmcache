"""Incremental input for the common workload schema, one request per line."""

import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from nano_lmcache.datasets.schemas.workload import WorkloadRequest


def iter_workload_jsonl(
    path: str | Path, *, limit: int | None = None,
) -> Iterator[WorkloadRequest]:
    """Read at most ``limit`` requests without parsing later records."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if limit == 0:
        return

    def records() -> Iterator[WorkloadRequest]:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("expected a request object")
                    yield WorkloadRequest.from_mapping(value)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc

    yield from islice(records(), limit)
