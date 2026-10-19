"""Token chunking and stable chunk-key derivation."""

import hashlib
from collections.abc import Sequence


def chunk_tokens(
    token_ids: Sequence[int],
    chunk_size: int,
) -> list[tuple[int, ...]]:
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
    payload = f"{namespace}:{','.join(str(t) for t in token_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
