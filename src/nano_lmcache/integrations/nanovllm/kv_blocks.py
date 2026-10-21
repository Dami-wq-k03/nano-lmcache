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

import torch

# Axis order that turns a block view [kv, layer, token, head, dim] into a
# chunk [layer, kv, head, token, dim]. The same permutation turns a chunk
# back into block order, because it only swaps two pairs of axes.
_BLOCK_TO_CHUNK = (1, 0, 3, 2, 4)


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
    _check_block_id(kv_cache, block_id)
    block = kv_cache[:, :, block_id]
    # `.contiguous()` already copies a permuted view; `.clone()` guarantees
    # a copy even in the corner case where the permuted view is contiguous.
    return block.permute(_BLOCK_TO_CHUNK).contiguous().clone()


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
    _check_block_id(kv_cache, block_id)
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

    kv_cache[:, :, block_id].copy_(payload.permute(_BLOCK_TO_CHUNK))


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
    if not 0 <= block_id < num_blocks:
        raise ValueError(f"block_id must be in 0..{num_blocks - 1}, got {block_id}")
