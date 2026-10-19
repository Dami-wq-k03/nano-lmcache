"""Token chunking and stable chunk-key derivation."""

import hashlib
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
) -> str:
    """Derive a stable, content-addressed key for a token chunk.

    namespace is mixed into the payload so the same tokens produce different
    keys under different models/configs, avoiding false cache hits.
    """
    payload = f"{namespace}:{','.join(str(t) for t in token_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
