"""Show that a disk-backed cache survives a process restart.

Run it twice:

    uv run python examples/demo_disk_cache.py   # cold: computes, stores
    uv run python examples/demo_disk_cache.py   # new process: hits on disk

Each run is a fresh Python process, so the second run's hits can only come
from files the first run left behind. vLLM's built-in prefix caching lives
in GPU memory and is lost on restart; this is the gap an external KV cache
fills. Pass --clear to start over.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from nano_lmcache.core.chunking import chunk_tokens
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache

CHUNK_SIZE = 2
DEFAULT_DIR = Path(tempfile.gettempdir()) / "nano-lmcache-demo"


def compute_kv_payload(chunk: tuple[int, ...]) -> str:
    """Stand-in for a real prefill pass over one chunk."""
    return "kv:" + "-".join(str(token) for token in chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear:
        shutil.rmtree(args.dir, ignore_errors=True)
        print(f"Cleared {args.dir}")
        return

    storage = DiskKVCache(args.dir)
    cache = ChunkedPrefixCache(
        chunk_size=CHUNK_SIZE, namespace="demo-model", storage=storage
    )
    tokens = [101, 205, 307, 409, 511, 613]

    print(f"Cache directory: {args.dir}")
    print(f"Entries found on disk at startup: {len(storage)}\n")

    result = cache.lookup(tokens)
    chunks = [c for c in chunk_tokens(tokens, CHUNK_SIZE) if len(c) == CHUNK_SIZE]
    for index, chunk in enumerate(chunks):
        hit = index < result.matched_chunks
        status = "HIT  (from disk)" if hit else "MISS (computed)"
        print(f"  chunk {index} {chunk} {status}")

    recomputed = [compute_kv_payload(c) for c in chunks[result.matched_chunks :]]
    cache.store(tokens, list(result.values) + recomputed)

    print(f"\nReused {result.matched_chunks}/{len(chunks)} chunks.")
    if result.matched_chunks == 0:
        print("Run the script again: the next process will find these on disk.")


if __name__ == "__main__":
    main()
