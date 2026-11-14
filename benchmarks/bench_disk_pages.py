"""Warm host-page-cache comparison: entry files, packed reads, async prefetch.

Run with ``uv run python benchmarks/bench_disk_pages.py``. No cache dropping,
GPU transfer, io_uring or GDS is measured. Optional SHA256 work between submit
and wait demonstrates caller/read overlap, not engine integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from nano_lmcache.storage.coalesced import BytesSerializer, CoalescedDiskKVCache
from nano_lmcache.storage.disk import DiskKVCache, EntryHeader
from nano_lmcache.storage.prefetch import AsyncPrefetchStorage


class CountingDiskKVCache(DiskKVCache):
    """Count entry-file loads (not kernel/block-device operations)."""

    read_ops = 0

    def _read(self, path: Path) -> tuple[EntryHeader, bytes]:
        self.read_ops += 1
        return super()._read(path)


def measure(
    operation: Callable[[], list[object | None]],
    reader: CountingDiskKVCache | CoalescedDiskKVCache,
    expected: list[object | None],
    repeats: int,
) -> dict[str, float]:
    assert operation() == expected  # explicit warmup, outside measurement
    reader.read_ops = 0
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - start) * 1000)
        if result != expected:
            raise RuntimeError("disk benchmark returned incorrect pages")
    return {
        "median_ms": statistics.median(samples),
        "payload_read_operations_per_lookup": reader.read_ops / repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--page-bytes", type=int, default=64 << 10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--work-rounds", type=int, default=0)
    args = parser.parse_args()
    if min(args.pages, args.page_bytes, args.repeats) <= 0 or args.work_rounds < 0:
        raise SystemExit(
            "pages, page-bytes, repeats must be positive; work-rounds >= 0"
        )
    payload = b"x" * args.page_bytes
    keys = [f"page-{index}" for index in range(args.pages)]
    expected: list[object | None] = [payload] * args.pages

    def caller_work() -> None:
        for _ in range(args.work_rounds):
            hashlib.sha256(payload).digest()

    with tempfile.TemporaryDirectory(prefix="nano-lmcache-disk-") as directory:
        root = Path(directory)
        files = CountingDiskKVCache(root / "files", serializer=BytesSerializer())
        with CoalescedDiskKVCache(root / "packed") as packed:
            for index, key in enumerate(keys):
                files.put(key, payload, parent_key=keys[index - 1] if index else None)
            packed.put_many(
                [
                    (key, payload, keys[index - 1] if index else None)
                    for index, key in enumerate(keys)
                ]
            )

            def per_file() -> list[object | None]:
                values = [files.get(key) for key in keys]
                caller_work()
                return values

            def per_page() -> list[object | None]:
                values = [packed.get(key) for key in keys]
                caller_work()
                return values

            def batch() -> list[object | None]:
                values = packed.get_many(keys)
                caller_work()
                return values

            with AsyncPrefetchStorage(packed, max_batch_size=args.pages) as prefetch:

                def asynchronous() -> list[object | None]:
                    future = prefetch.prefetch(keys)
                    caller_work()
                    return future.result()

                modes = {
                    "entry_files": measure(per_file, files, expected, args.repeats),
                    "packed_per_page": measure(
                        per_page, packed, expected, args.repeats
                    ),
                    "packed_get_many": measure(batch, packed, expected, args.repeats),
                    "threaded_prefetch": measure(
                        asynchronous, packed, expected, args.repeats
                    ),
                }
            result = {
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "scope": "warm buffered host reads; no GPU, cold-disk, io_uring or GDS",
                "read_counter": "entry-file loads for baseline; pread calls for packed",
                "pages": args.pages,
                "page_bytes": args.page_bytes,
                "repeats": args.repeats,
                "caller_sha256_rounds": args.work_rounds,
                "packed_live_bytes": packed.nbytes,
                "packed_file_bytes": (packed.directory / "pages.data").stat().st_size,
                "modes": modes,
                "batch_vs_entry_files_speedup": (
                    modes["entry_files"]["median_ms"]
                    / modes["packed_get_many"]["median_ms"]
                ),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
