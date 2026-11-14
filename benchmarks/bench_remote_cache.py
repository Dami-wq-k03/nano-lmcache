"""Compare one-request-per-key lookup with the shared cache batch API."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import Any

from nano_lmcache.storage.remote import RemoteCacheClient, RemoteCacheServer


def measure(operation: Any, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.entries <= 0 or args.repeats <= 0:
        raise SystemExit("entries and repeats must be positive")

    server = RemoteCacheServer()
    host, port = server.start()
    try:
        client = RemoteCacheClient(host, port)
        for index in range(args.entries):
            client.put(
                f"key-{index}",
                {"index": index},
                parent_key=f"key-{index - 1}" if index else None,
            )
        keys = [f"key-{index}" for index in range(args.entries)]
        per_key_ms = measure(
            lambda: [client.get(key) for key in keys], args.repeats
        )
        batch_ms = measure(lambda: client.get_many(keys), args.repeats)
        result: dict[str, Any] = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "entries": args.entries,
            "repeats": args.repeats,
            "per_key_ms": per_key_ms,
            "batch_ms": batch_ms,
            "speedup": per_key_ms / batch_ms,
        }
    finally:
        server.close()

    encoded = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output:
            output.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
