from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.integrations.nanovllm.layout import KVBlockLayout
from nano_lmcache.storage.protocol import CacheStorage


@runtime_checkable
class MembershipStorage(CacheStorage, Protocol):
    def __contains__(self, key: object) -> bool: ...
    def get(self, key: str) -> torch.Tensor | None: ...


def max_reusable_blocks(num_tokens: int, block_size: int) -> int:
    num_blocks = (num_tokens + block_size - 1) // block_size
    return max(0, num_blocks - 1)


@dataclass
class ExternalCacheStats:
    hit_blocks: int = 0
    fetched_blocks: int = 0
    stored_blocks: int = 0


class NanoVLLMExternalCache:
    def __init__(
        self,
        layout: KVBlockLayout,
        storage: MembershipStorage,
    ) -> None:
        self._layout = layout
        self._storage = storage
        self._namespace = layout.namespace()
        self._stats = ExternalCacheStats()

    def block_keys(
        self, token_ids: Sequence[int], num_blocks: int | None = None
    ) -> tuple[str, ...]:
        complete = len(token_ids) // self._layout.block_size
        if num_blocks is None:
            num_blocks = complete
        prefix = token_ids[: num_blocks * self._layout.block_size]
        keys: list[str] = []
        prefix_key: str | None = None
        for block in chunk_tokens(prefix, self._layout.block_size):
            prefix_key = make_chunk_key(
                block, namespace=self._namespace, prefix_key=prefix_key
            )
            keys.append(prefix_key)
        return tuple(keys)

    def count_hits(self, token_ids: Sequence[int], num_gpu_cached_blocks: int) -> int:
        limit = max_reusable_blocks(len(token_ids), self._layout.block_size)
        keys = self.block_keys(token_ids, limit)
        hits = 0
        for key in keys[num_gpu_cached_blocks:]:
            present = key in self._storage
            if not present:
                break
            hits += 1

        self._stats.hit_blocks += hits
        return hits

    def fetch(
        self, token_ids: Sequence[int], start_block: int, num_blocks: int
    ) -> list[torch.Tensor]:
        if num_blocks == 0:
            return []
        keys = self.block_keys(token_ids, start_block + num_blocks)
        payloads: list[torch.Tensor] = []
        for key in keys[start_block:start_block + num_blocks]:
            payload = self._storage.get(key)
            if payload is None:
                break
            payloads.append(payload)
        self._stats.fetched_blocks += len(payloads)
        return payloads

    def missing_blocks(
        self, token_ids: Sequence[int], num_completed_blocks: int
    ) -> range:
        keys = self.block_keys(token_ids, num_completed_blocks)
        first_missing = next(
            (i for i, key in enumerate(keys) if key not in self._storage),
            len(keys),
        )
        return range(first_missing, num_completed_blocks)

    def store(
        self,
        token_ids: Sequence[int],
        start_block: int,
        payloads: Sequence[torch.Tensor],
    ) -> int:
        keys = self.block_keys(token_ids, start_block + len(payloads))
        block_keys = keys[start_block:]
        for index, (key, payload) in enumerate(
            zip(block_keys, payloads, strict=True), start=start_block
        ):
            parent_key = keys[index - 1] if index else None
            self._storage.put(key, payload, parent_key=parent_key)

        stored = sum(key in self._storage for key in block_keys)
        self._stats.stored_blocks += stored
        return stored

    @property
    def layout(self) -> KVBlockLayout:
        return self._layout