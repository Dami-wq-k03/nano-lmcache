"""Find, fetch, and store nano-vLLM KV blocks in nano-lmcache (the L2 cache).

nano-vLLM's own prefix cache (L1) answers first. This module answers for
the blocks after the L1 hits, and works purely on token IDs, keys, and
payloads: copying blocks in and out of the GPU KV cache is N0's
`import_block` / `export_block`, called from `ModelRunner`.

Keys
----

Block `i` of a prompt gets the nano-lmcache chunk key of its tokens,
chained to the key of block `i - 1` and hashed with the layout's
namespace, exactly as `ChunkedPrefixCache` does with
`chunk_size == block_size`. So a key names the whole prefix up to that
block, for one model configuration. Keys are computed from token IDs
alone, which is cheap; nothing is read from storage to build them.

Calls, by nano-vLLM hook
------------------------

- `count_hits`, from `BlockManager.can_allocate`, which runs on every
  scheduling step while a request waits: membership checks only.
- `fetch`, from `BlockManager.allocate`: read the payloads once.
- `missing_blocks` and `store`, after each step: store the blocks that
  `hash_blocks` just completed and L2 does not have yet.

See `docs/nanovllm-integration.md` for the rules behind each call.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.integrations.nanovllm.layout import KVBlockLayout
from nano_lmcache.storage.protocol import CacheStorage


@runtime_checkable
class MembershipStorage(CacheStorage, Protocol):
    """Storage that can say whether it holds a key without reading it.

    Every nano-lmcache backend (`MemoryKVCache`, `DiskKVCache`,
    `TieredStorage`, `CPUOffloadKVCache`) supports this.
    """

    def __contains__(self, key: object) -> bool: ...


def max_reusable_blocks(num_tokens: int, block_size: int) -> int:
    """How many leading blocks of a prompt may come from a cache at all.

    Mirrors nano-vLLM's `BlockManager.can_allocate`, which only looks at
    `range(num_blocks - 1)`: the last block is always computed, even when
    it is full, so the forward pass has tokens to produce logits from.
    """
    num_blocks = (num_tokens + block_size - 1) // block_size
    return max(0, num_blocks - 1)


@dataclass
class ExternalCacheStats:
    """What the L2 cache did, counted in blocks."""

    # Blocks reported as available after the L1 hits.
    hit_blocks: int = 0
    # Blocks whose payload was read from storage.
    fetched_blocks: int = 0
    # Blocks written to storage (and kept by it).
    stored_blocks: int = 0


class NanoVLLMExternalCache:
    """nano-lmcache storage addressed by nano-vLLM token blocks."""

    def __init__(self, layout: KVBlockLayout, storage: MembershipStorage) -> None:
        if not isinstance(storage, MembershipStorage):
            raise TypeError(
                "storage must support `key in storage` (__contains__) so hits "
                "can be counted without reading payloads"
            )
        self._layout = layout
        self._storage = storage
        self._namespace = layout.namespace()
        self._stats = ExternalCacheStats()

    @property
    def layout(self) -> KVBlockLayout:
        return self._layout

    @property
    def storage(self) -> MembershipStorage:
        return self._storage

    @property
    def stats(self) -> ExternalCacheStats:
        return self._stats

    def block_keys(
        self, token_ids: Sequence[int], num_blocks: int | None = None
    ) -> tuple[str, ...]:
        """Keys of the first `num_blocks` complete blocks (default: all)."""
        complete = len(token_ids) // self._layout.block_size
        if num_blocks is None:
            num_blocks = complete
        if not 0 <= num_blocks <= complete:
            raise ValueError(
                f"num_blocks must be in 0..{complete} complete blocks, "
                f"got {num_blocks}"
            )

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
        """How many blocks right after the L1 hits L2 holds, without reading them.

        Never counts the prompt's last block (see `max_reusable_blocks`).
        """
        limit = max_reusable_blocks(len(token_ids), self._layout.block_size)
        if not 0 <= num_gpu_cached_blocks <= limit:
            raise ValueError(
                f"num_gpu_cached_blocks must be in 0..{limit}, "
                f"got {num_gpu_cached_blocks}"
            )

        hits = 0
        for key in self.block_keys(token_ids, limit)[num_gpu_cached_blocks:]:
            if key not in self._storage:
                break
            hits += 1

        self._stats.hit_blocks += hits
        return hits

    def fetch(
        self, token_ids: Sequence[int], start_block: int, num_blocks: int
    ) -> list[torch.Tensor]:
        """Read the payloads of blocks `start_block .. start_block + num_blocks - 1`.

        Stops at the first block that is no longer stored, so it can return
        fewer payloads than asked for if storage evicted some since
        `count_hits`; the caller must then treat only the returned blocks as
        cached.
        """
        limit = max_reusable_blocks(len(token_ids), self._layout.block_size)
        if start_block < 0 or num_blocks < 0:
            raise ValueError("start_block and num_blocks must not be negative")
        if start_block + num_blocks > limit:
            raise ValueError(
                f"blocks {start_block}..{start_block + num_blocks - 1} go past "
                f"the {limit} reusable blocks of this prompt (its last block "
                "is always computed)"
            )

        keys = self.block_keys(token_ids, start_block + num_blocks)[start_block:]
        payloads: list[torch.Tensor] = []
        for key in keys:
            payload = self._storage.get(key)
            if payload is None:
                break
            if not isinstance(payload, torch.Tensor):
                raise TypeError(
                    f"stored payload for {key} is {type(payload).__name__}"
                )
            payloads.append(payload)

        self._stats.fetched_blocks += len(payloads)
        return payloads

    def missing_blocks(
        self, token_ids: Sequence[int], num_completed_blocks: int
    ) -> range:
        """Completed blocks L2 does not hold yet, as a range of block indices.

        Prefix-safe storage keeps chains whole, so the blocks L2 holds are
        always a prefix of the prompt: everything from the first missing
        block on is missing.
        """
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
        """Store payloads for blocks `start_block`, `start_block + 1`, ...

        Each block is linked to the key of the block before it, so storage
        drops a block whose prefix it does not hold (it could never be
        reached). Returns how many blocks storage actually kept.
        """
        if start_block < 0:
            raise ValueError("start_block must not be negative")
        keys = self.block_keys(token_ids, start_block + len(payloads))

        # Validate everything before writing anything, so a bad call never
        # leaves a partial chain behind.
        for payload in payloads:
            if not isinstance(payload, torch.Tensor):
                raise ValueError(
                    f"payloads must be tensors, got {type(payload).__name__}"
                )
            if tuple(payload.shape) != self._layout.chunk_shape:
                raise ValueError(
                    f"payload shape {tuple(payload.shape)} is not this layout's "
                    f"chunk shape {self._layout.chunk_shape}"
                )

        for index, payload in enumerate(payloads, start=start_block):
            parent_key = keys[index - 1] if index > 0 else None
            self._storage.put(keys[index], payload, parent_key=parent_key)

        stored = sum(key in self._storage for key in keys[start_block:])
        self._stats.stored_blocks += stored
        return stored
