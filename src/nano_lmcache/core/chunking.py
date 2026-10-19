"""Token chunking and stable chunk-key derivation."""

import hashlib
import json
from collections.abc import Sequence


def chunk_tokens(
    token_ids: Sequence[int],
    chunk_size: int,
) -> list[tuple[int, ...]]:
    """Split token_ids into fixed-size chunks; the last chunk may be shorter."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    return [
        tuple(token_ids[i : i + chunk_size])
        for i in range(0, len(token_ids), chunk_size)
    ]


def make_chunk_key(
    token_ids: Sequence[int],
    *,
    namespace: str,
    prefix_key: str | None = None,
) -> str:
    """Derive a stable key for a chunk within its prefix context.

    In a causal model, a token's KV state depends on everything before it,
    so two chunks with identical tokens must not collide unless their
    preceding context (prefix_key, typically the previous chunk's key)
    also matches. JSON serialization keeps the key's input structure
    unambiguous, instead of hand-concatenating strings.
    """
    payload = json.dumps(
        {
            "namespace": namespace,
            "prefix_key": prefix_key,
            "token_ids": list(token_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
