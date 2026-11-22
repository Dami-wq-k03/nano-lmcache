from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from nanovllm.engine.block_manager import BlockManager

from nano_lmcache.integrations.nanovllm.external_cache import NanoVLLMExternalCache
from nano_lmcache.integrations.nanovllm.kv_blocks import (
    export_blocks,
    import_blocks,
)
from nano_lmcache.integrations.nanovllm.layout import KVBlockLayout


@dataclass
class EngineCacheStats:
    l1_hit_blocks: int = 0
    l2_hit_blocks: int = 0
    prefill_tokens: int = 0
    stored_blocks: int = 0


class ExternalBlockManager(BlockManager):  # type: ignore[misc]
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        external: NanoVLLMExternalCache,
        kv_cache: torch.Tensor | None = None,
    ) -> None:
        super().__init__(num_blocks, block_size)
        self.external = external
        self.kv_cache = kv_cache
        self.stats = EngineCacheStats()
        self._l1_hits: dict[int, int] = {}
        self._pending_loads: list[tuple[int, torch.Tensor]] = []


    def can_allocate(self, seq: Any) -> int:
        l1_hits: int = super().can_allocate(seq)
        if l1_hits == -1:
            return -1

        self._l1_hits[seq.seq_id] = l1_hits
        l2_hits = self.external.count_hits(seq.token_ids, l1_hits)
        return l1_hits + l2_hits

    def allocate(self, seq: Any, num_cached_blocks: int) -> None:
        l1_hits = self._l1_hits.pop(seq.seq_id)
        super().allocate(seq, l1_hits)
        self.stats.l1_hit_blocks += l1_hits
        wanted = num_cached_blocks - l1_hits
        payloads = (
            self.external.fetch(seq.token_ids, l1_hits, wanted) 
            if wanted
            else []
        )

        restored_end = l1_hits + len(payloads)
        seq.num_cached_tokens = restored_end * self.block_size
        self.stats.l2_hit_blocks += len(payloads)

        for block_id, payload in zip(
            seq.block_table[l1_hits:restored_end], payloads, strict=True
        ):
            self._pending_loads.append((block_id, payload))
        
        self._register_l1_hashes(seq, l1_hits, restored_end)

    def hash_blocks(self, seq: Any) -> None:
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        super().hash_blocks(seq)
        kv_cache = self.kv_cache
        if start == end or kv_cache is None:
            return
        
        missing = self.external.missing_blocks(seq.token_ids, end)
        if not missing:
            return
        payloads = export_blocks(
            kv_cache, [seq.block_table[index] for index in missing]
        )
        self.stats.stored_blocks += self.external.store(
            seq.token_ids, missing.start, payloads
        )


    def _register_l1_hashes(self, seq: Any, start: int, end: int) -> None:
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for index in range(start, end):
            token_ids = seq.block(index)
            h = self.compute_hash(token_ids, h)
            block = self.blocks[seq.block_table[index]]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def take_pending_loads(self) -> list[tuple[int, torch.Tensor]]:
        pending, self._pending_loads = self._pending_loads, []
        return pending


def attach(llm: Any, external: NanoVLLMExternalCache) -> ExternalBlockManager:
    if not llm.scheduler.is_finished():
        raise RuntimeError("attach before adding requests")

    runner = llm.model_runner
    expected = external.layout
    actual = KVBlockLayout.from_kv_cache(
        runner.kv_cache,
        model=expected.model,
        revision=expected.revision,
        tensor_parallel_size=expected.tensor_parallel_size,
        tensor_parallel_rank=expected.tensor_parallel_rank,
        config_fingerprint=expected.config_fingerprint,
    )

    if actual != expected:
        raise ValueError(
            f"engine layout {actual} does not match cache layout {expected}"
        )

    previous = llm.scheduler.block_manager
    manager = ExternalBlockManager(
        len(previous.blocks), previous.block_size, external, runner.kv_cache
    )
    llm.scheduler.block_manager = manager
    _wrap_run(runner, manager)
    return manager


def detach(llm: Any) -> BlockManager:
    if not llm.scheduler.is_finished():
        raise RuntimeError("detach while the engine has requests")

    runner = llm.model_runner
    previous = llm.scheduler.block_manager
    plain = BlockManager(len(previous.blocks), previous.block_size)
    llm.scheduler.block_manager = plain
    original = getattr(runner, "_nano_lmcache_original_run", None)
    if original is not None:
        runner.run = original
    return plain


def _wrap_run(runner: Any, manager: ExternalBlockManager) -> None:
    original: Callable[[Sequence[Any], bool], list[int]] = getattr(
        runner, "_nano_lmcache_original_run", runner.run
    )
    runner._nano_lmcache_original_run = original

    def run(seqs: Sequence[Any], is_prefill: bool) -> list[int]:
        pending_loads = manager.take_pending_loads()
        if pending_loads:
            import_blocks(
                runner.kv_cache,
                [block_id for block_id, _ in pending_loads],
                [payload for _, payload in pending_loads],
            )
        if is_prefill:
            manager.stats.prefill_tokens += sum(
                seq.num_scheduled_tokens for seq in seqs
            )
        outputs = original(seqs, is_prefill)
        return outputs

    runner.run = run