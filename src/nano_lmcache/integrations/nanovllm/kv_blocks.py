from collections.abc import Sequence

import torch


def export_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    return export_blocks(kv_cache, [block_id])[0]


def export_blocks(
    kv_cache: torch.Tensor, block_ids: Sequence[int]
) -> list[torch.Tensor]:
    if not block_ids:
        return []
    indices = torch.tensor(block_ids, dtype=torch.long, device=kv_cache.device)
    batch = kv_cache.index_select(2, indices).permute(2, 1, 0, 4, 3, 5).contiguous()
    return list(batch.unbind(0))


def import_block(
    kv_cache: torch.Tensor, block_id: int, payload: torch.Tensor
) -> None:
    import_blocks(kv_cache, [block_id], [payload])


def import_blocks(
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    payloads: Sequence[torch.Tensor],
) -> None:
    groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for block_id, payload in zip(block_ids, payloads, strict=True):
        groups.setdefault(payload.device, []).append((block_id, payload))

    for group in groups.values():
        ids, tensors = zip(*group, strict=True)
        indices = torch.tensor(ids, dtype=torch.long, device=kv_cache.device)
        source = torch.stack(tensors).permute(2, 1, 0, 4, 3, 5)
        kv_cache.index_copy_(2, indices, source.to(kv_cache.device))