"""Move single KV blocks between nano-vLLM's paged cache and nano-lmcache.

nano-vLLM (commit `bb823b3`) allocates one tensor for its whole KV cache
in `ModelRunner.allocate_kv_cache`:

    kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
               K/V                 physical block   tokens

Attention writes new keys and values into it through `slot_mapping`, and
`BlockManager` hands out physical block IDs. One physical block is the
strided view `kv_cache[:, :, block_id]`, shaped
`[2, num_layers, block_size, num_kv_heads, head_dim]`.

nano-lmcache already has a chunk format, used by `tensor/kv_chunks.py`
and every storage backend that holds tensors:

    chunk:    [num_layers, 2, num_kv_heads, chunk_tokens, head_dim]

With `chunk_tokens == block_size`, a block and a chunk hold the same
numbers in a different order, so moving one is a `permute` plus a copy:

    block [kv, layer, token, head, dim]  --permute(1, 0, 3, 2, 4)-->
    chunk [layer, kv, head, token, dim]

Keeping one chunk format means everything built for chunks (serializers,
CPU offload, disk and tiered storage) works for nano-vLLM blocks
unchanged, and a reader learns one layout instead of one per engine.
"""

from collections.abc import Sequence

import torch


def chunk_shape(kv_cache: torch.Tensor) -> tuple[int, int, int, int, int]:
    """Shape of one exported block: [layers, 2, kv_heads, block_size, head_dim]."""
    _check_kv_cache(kv_cache)
    _, num_layers, _, block_size, num_kv_heads, head_dim = kv_cache.shape
    return (num_layers, 2, num_kv_heads, block_size, head_dim)


def export_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    """Copy physical block `block_id` out of nano-vLLM's KV cache as a chunk.

    The result is a new contiguous tensor on the KV cache's device. It must
    be a copy, not a view: nano-vLLM reuses a physical block for other
    tokens once the sequence that owned it finishes.
    """
    return export_blocks(kv_cache, [block_id])[0]


def export_blocks(
    kv_cache: torch.Tensor, block_ids: Sequence[int]
) -> list[torch.Tensor]:
    """Gather blocks once; return independent, contiguous [L, 2, H, T, D] chunks.

    The chunks are disjoint views of one owned batch, not views of the engine's
    reusable cache. Block order (including repeated source IDs) is preserved.
    """
    _check_kv_cache(kv_cache)
    for block_id in block_ids:
        _check_block_id(kv_cache, block_id)
    if not block_ids:
        return []
    indices = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    batch = kv_cache.index_select(2, indices).permute(2, 1, 0, 4, 3, 5).contiguous()
    return list(batch.unbind(0))


def import_block(
    kv_cache: torch.Tensor, block_id: int, payload: torch.Tensor
) -> None:
    """Write a chunk into physical block `block_id` of nano-vLLM's KV cache.

    `payload` may live on another device (a disk tier returns CPU tensors);
    the copy moves it. Everything is checked before anything is written,
    so a rejected payload leaves the cache untouched.

    The copy is issued on the current CUDA stream, the same stream
    nano-vLLM's forward pass runs on, so attention cannot read the block
    before the copy has finished. Code that runs the model on another
    stream must synchronize first.
    """
    import_blocks(kv_cache, [block_id], [payload])


def import_blocks(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    payloads: Sequence[torch.Tensor],
) -> None:
    """Validate a batch, then scatter on the current stream before forward.

    Stack on each source device so mixed CPU/GPU storage tiers still work,
    with one transfer and indexed scatter per source device, not per block.
    Destination IDs must be unique; duplicate indexed writes are ambiguous.
    """
    _check_kv_cache(kv_cache)
    if len(block_ids) != len(payloads):
        raise ValueError("block_ids and payloads must have the same length")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("destination block_ids must be unique")
    groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for block_id, payload in zip(block_ids, payloads, strict=True):
        _check_block_id(kv_cache, block_id)
        _check_payload(kv_cache, payload)
        groups.setdefault(payload.device, []).append((block_id, payload))
    for group in groups.values():
        indices = torch.tensor(
            [block_id for block_id, _ in group],
            dtype=torch.long,
            device=kv_cache.device,
        )
        batch = torch.stack([payload for _, payload in group])
        source = batch.permute(2, 1, 0, 4, 3, 5).to(kv_cache.device)
        kv_cache.index_copy_(2, indices, source)


def _check_payload(kv_cache: torch.Tensor, payload: torch.Tensor) -> None:
    expected = chunk_shape(kv_cache)

    if payload.dim() != len(expected):
        raise ValueError(
            f"payload shape {tuple(payload.shape)} is not a chunk; expected "
            f"{expected} = [layers, 2, kv_heads, block_size, head_dim]"
        )
    # Report a block size mismatch on its own: it is the likeliest mistake
    # (nano-vLLM's kvcache_block_size and nano-lmcache's chunk_size differ).
    if payload.shape[3] != expected[3] and (
        tuple(payload.shape[:3]) + (payload.shape[4],)
        == expected[:3] + (expected[4],)
    ):
        raise ValueError(
            f"payload holds {payload.shape[3]} tokens, but the KV cache "
            f"block size is {expected[3]}"
        )
    if tuple(payload.shape) != expected:
        raise ValueError(
            f"payload shape {tuple(payload.shape)} does not match this KV "
            f"cache's chunk shape {expected} = "
            "[layers, 2, kv_heads, block_size, head_dim]"
        )
    if payload.dtype != kv_cache.dtype:
        raise ValueError(
            f"payload dtype {payload.dtype} does not match the KV cache "
            f"dtype {kv_cache.dtype}"
        )

def _check_kv_cache(kv_cache: torch.Tensor) -> None:
    if kv_cache.dim() != 6 or kv_cache.shape[0] != 2:
        raise ValueError(
            "expected a nano-vLLM KV cache shaped [2, num_layers, num_blocks, "
            f"block_size, num_kv_heads, head_dim], got {tuple(kv_cache.shape)}"
        )


def _check_block_id(kv_cache: torch.Tensor, block_id: int) -> None:
    _check_kv_cache(kv_cache)
    num_blocks = kv_cache.shape[2]
    # Negative IDs must be rejected explicitly: nano-vLLM pads block tables
    # with -1, and `kv_cache[:, :, -1]` would quietly mean the last block.
    if type(block_id) is not int or not 0 <= block_id < num_blocks:
        raise ValueError(f"block_id must be in 0..{num_blocks - 1}, got {block_id}")
