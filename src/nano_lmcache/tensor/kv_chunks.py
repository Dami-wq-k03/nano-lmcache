"""Split per-layer KV tensors into chunks and join them back.

A cached chunk is one tensor holding the K and V of every layer for
`chunk_size` consecutive tokens:

    shape = [num_layers, 2, num_kv_heads, chunk_tokens, head_dim]
              layer      K/V  heads        tokens        per-head size

Index 0 of the K/V axis is the key, index 1 the value. The batch dimension
is dropped because a cached prefix belongs to a single sequence.

Stacking layers into one tensor needs every layer to have the same number
of KV heads and head size, which holds for plain-attention models such as
Qwen3 or GPT-2. Models with mixed layer shapes (sliding-window or hybrid
layers) would need a richer chunk type.
"""

from collections.abc import Sequence

import torch

# One layer's cached attention state: (keys, values), each shaped
# [num_kv_heads, num_tokens, head_dim].
LayerKV = tuple[torch.Tensor, torch.Tensor]


def split_kv(layers: Sequence[LayerKV], chunk_size: int) -> list[torch.Tensor]:
    """Cut per-layer KV into one tensor per complete chunk.

    Only complete chunks are returned, matching `ChunkedPrefixCache`, which
    never stores or reuses an incomplete tail. Chunks are copies, so they
    stay valid however the source tensors are modified or freed later.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    expected = layers[0][0].shape
    for key, value in layers:
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                "all layers must share one KV shape, "
                f"got {tuple(key.shape)} vs {tuple(expected)}"
            )

    # [layers, 2, kv_heads, tokens, head_dim]
    stacked = torch.stack([torch.stack((key, value)) for key, value in layers])
    num_tokens = stacked.shape[3]

    return [
        stacked[:, :, :, start : start + chunk_size].clone()
        for start in range(0, num_tokens - chunk_size + 1, chunk_size)
    ]


def join_kv(chunks: Sequence[torch.Tensor]) -> list[LayerKV]:
    """Concatenate consecutive chunks back into per-layer (K, V) tensors."""
    if not chunks:
        raise ValueError("need at least one chunk to join")

    joined = torch.cat(list(chunks), dim=3)  # along the token axis

    return [(layer[0], layer[1]) for layer in joined]
