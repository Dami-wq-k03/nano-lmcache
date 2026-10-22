"""Install nano-lmcache as the L2 KV cache of a running nano-vLLM engine.

Roadmap tasks N2 (restore) and N3 (store). nano-vLLM stays an unmodified, pinned
dependency (commit `bb823b3`): instead of patching its source, `attach()`
swaps in a `BlockManager` subclass and wraps `ModelRunner.run`, which are
the only two places that need to know about L2.

    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    layout = KVBlockLayout.from_hf_config(
        llm.model_runner.config.hf_config, model="Qwen/Qwen3-0.6B", revision=...
    )
    attach(llm, NanoVLLMExternalCache(layout, storage))
    llm.generate(...)

What happens for a new request:

1. `can_allocate` runs nano-vLLM's own L1 lookup, then asks L2 how many of
   the following blocks it holds (membership only: this runs on every
   scheduling step while the request waits).
2. `allocate` attaches the L1 blocks, allocates fresh GPU blocks for the
   rest, fetches the L2 payloads, registers L1 hashes for the restored
   blocks, and sets `num_cached_tokens` to cover L1 and L2 hits.
3. The wrapped `run` copies the fetched payloads into their GPU blocks
   before the forward pass, which then computes only the uncached tail.
4. After the step, `hash_blocks` registers the blocks it completed, and
   any complete block L2 does not hold yet is exported and stored. That
   includes blocks completed by chunked prefill and by decode.

This module imports nano-vLLM, so it needs `uv sync --group nanovllm`; the
rest of `nano_lmcache.integrations.nanovllm` does not. The rules each step
keeps are numbered in `docs/nanovllm-integration.md`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from nanovllm.engine.block_manager import BlockManager

from nano_lmcache.integrations.nanovllm.external_cache import NanoVLLMExternalCache
from nano_lmcache.integrations.nanovllm.kv_blocks import export_block, import_block
from nano_lmcache.integrations.nanovllm.layout import KVBlockLayout


@dataclass
class EngineCacheStats:
    """Where the prompt blocks of admitted requests came from."""

    # Blocks reused from nano-vLLM's own GPU prefix cache.
    l1_hit_blocks: int = 0
    # Blocks restored from nano-lmcache into fresh GPU blocks.
    l2_hit_blocks: int = 0
    # Prompt tokens the model actually computed during prefill.
    prefill_tokens: int = 0
    # Completed blocks exported from the GPU and kept by L2.
    stored_blocks: int = 0


class ExternalBlockManager(BlockManager):  # type: ignore[misc]
    """nano-vLLM's BlockManager, extended to restore blocks from L2 and fill it.

    `kv_cache` is the engine's `ModelRunner.kv_cache`; without it the
    manager restores but never stores.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        external: NanoVLLMExternalCache,
        kv_cache: torch.Tensor | None = None,
    ) -> None:
        super().__init__(num_blocks, block_size)
        if external.layout.block_size != block_size:
            raise ValueError(
                f"layout block_size {external.layout.block_size} does not match "
                f"nano-vLLM's kvcache_block_size {block_size}"
            )
        self.external = external
        self.kv_cache = kv_cache
        self.stats = EngineCacheStats()
        # seq_id -> L1 hits found by the latest can_allocate for that request.
        self._l1_hits: dict[int, int] = {}
        # (GPU block ID, payload) pairs fetched in allocate, waiting to be
        # copied into the KV cache before the next forward pass.
        self._pending_loads: list[tuple[int, torch.Tensor]] = []

    def can_allocate(self, seq: Any) -> int:
        # The original check also makes sure there are enough free GPU
        # blocks; it counts L2 blocks as new, which they are (rule 4).
        l1_hits: int = super().can_allocate(seq)
        if l1_hits == -1:
            return -1
        self._l1_hits[seq.seq_id] = l1_hits
        # Membership checks only (rule 3); never past the last block (rule 1).
        return l1_hits + self.external.count_hits(seq.token_ids, l1_hits)

    def allocate(self, seq: Any, num_cached_blocks: int) -> None:
        try:
            l1_hits = self._l1_hits.pop(seq.seq_id)
        except KeyError:
            raise RuntimeError(
                "allocate() without a preceding can_allocate() for this sequence"
            ) from None

        super().allocate(seq, l1_hits)
        self.stats.l1_hit_blocks += l1_hits

        wanted = num_cached_blocks - l1_hits
        payloads = self.external.fetch(seq.token_ids, l1_hits, wanted) if wanted else []
        restored_end = l1_hits + len(payloads)
        block_ids = seq.block_table[l1_hits:restored_end]
        self._pending_loads.extend(zip(block_ids, payloads, strict=True))

        # Only blocks that will really receive their KV get an L1 hash
        # (rule 2), and the request counts as cached up to there (rule 5).
        self._register_l1_hashes(seq, l1_hits, restored_end)
        seq.num_cached_tokens = restored_end * self.block_size
        self.stats.l2_hit_blocks += len(payloads)

    def hash_blocks(self, seq: Any) -> None:
        """Register the blocks this step completed, then store new ones in L2.

        `Scheduler.postprocess` calls this before it frees a finished
        sequence's blocks, so the blocks still hold this sequence's KV
        (rule 7). Only complete, computed blocks are stored, each once
        (rule 8).
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        super().hash_blocks(seq)
        # Most decode steps complete no block; skip them before hashing
        # any keys, the same shortcut the original method takes.
        if start == end or self.kv_cache is None:
            return

        # Usually starts at `start`, but can start earlier: if L2 lost
        # blocks that L1 still holds, they are stored again from the GPU.
        missing = self.external.missing_blocks(seq.token_ids, end)
        if not missing:
            return
        payloads = [
            export_block(self.kv_cache, seq.block_table[index]) for index in missing
        ]
        self.stats.stored_blocks += self.external.store(
            seq.token_ids, missing.start, payloads
        )

    def take_pending_loads(self) -> list[tuple[int, torch.Tensor]]:
        """Payloads to copy into the KV cache before the next forward pass."""
        pending, self._pending_loads = self._pending_loads, []
        return pending

    def _register_l1_hashes(self, seq: Any, start: int, end: int) -> None:
        """Hash restored blocks exactly as `hash_blocks` would have.

        `hash_blocks` seeds each later block's hash with the hash stored on
        the block before it, so a restored block left with the reset hash
        -1 would give every later block of the sequence a wrong L1 hash.
        """
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for index in range(start, end):
            token_ids = seq.block(index)
            h = self.compute_hash(token_ids, h)
            block = self.blocks[seq.block_table[index]]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


def attach(llm: Any, external: NanoVLLMExternalCache) -> ExternalBlockManager:
    """Make a single-GPU nano-vLLM engine use `external` as its L2 KV cache.

    The engine restores L2 hits before prefill and stores every complete
    block it computes.

    Call it after creating the engine and before adding requests. Calling
    it again installs a fresh block manager, which empties L1 (useful in
    tests and demos to show restores from L2).
    """
    runner = llm.model_runner
    if getattr(runner, "world_size", 1) != 1:
        raise NotImplementedError(
            "tensor parallelism is not supported yet: every rank would have to "
            "load its own slice of the KV heads"
        )
    if not llm.scheduler.is_finished():
        raise RuntimeError("attach() before adding requests to the engine")

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
            f"the engine's KV cache layout {actual} does not match the cache "
            f"layout {expected}"
        )

    previous = llm.scheduler.block_manager
    manager = ExternalBlockManager(
        len(previous.blocks), previous.block_size, external, runner.kv_cache
    )
    llm.scheduler.block_manager = manager
    _wrap_run(runner, manager)
    return manager



def detach(llm: Any) -> BlockManager:
    """Give the engine back plain nano-vLLM behavior, with an empty L1.

    Installs a fresh nano-vLLM `BlockManager` and the original
    `ModelRunner.run`, so nothing is restored from or stored into L2. The
    L2 storage itself is untouched; `attach` it again to use it. Useful as
    the no-cache baseline in tests and benchmarks: every prompt is then
    computed in full.
    """
    if not llm.scheduler.is_finished():
        raise RuntimeError("detach() while the engine has requests")

    runner = llm.model_runner
    previous = llm.scheduler.block_manager
    plain = BlockManager(len(previous.blocks), previous.block_size)
    llm.scheduler.block_manager = plain
    original = getattr(runner, "_nano_lmcache_original_run", None)
    if original is not None:
        runner.run = original
    return plain


def _wrap_run(runner: Any, manager: ExternalBlockManager) -> None:
    """Copy fetched L2 payloads into the KV cache before every forward pass.

    nano-vLLM's `ModelRunner.call` looks methods up on the instance, so an
    instance attribute named `run` takes over. The original method is kept
    so that attaching again replaces the wrapper instead of nesting it.
    """
    original: Callable[[Sequence[Any], bool], list[int]] = getattr(
        runner, "_nano_lmcache_original_run", runner.run
    )
    runner._nano_lmcache_original_run = original

    def run(seqs: Sequence[Any], is_prefill: bool) -> list[int]:
        # Same CUDA stream as the forward pass, so attention cannot read a
        # block before its copy has finished (rule 6).
        for block_id, payload in manager.take_pending_loads():
            import_block(runner.kv_cache, block_id, payload)
        if is_prefill:
            manager.stats.prefill_tokens += sum(
                seq.num_scheduled_tokens for seq in seqs
            )
        return original(seqs, is_prefill)

    runner.run = run
