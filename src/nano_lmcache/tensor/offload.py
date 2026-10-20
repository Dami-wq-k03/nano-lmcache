"""Offload KV chunks from GPU memory to CPU memory, and bring them back.

GPU memory is the scarcest resource in LLM serving: the model weights and
the KV cache of running requests compete for it. Offloading finished
chunks to CPU memory, which is usually several times larger, lets a server
keep far more reusable prefixes than would fit on the GPU. On a hit, the
chunk is copied back to the GPU; that copy is the cost of reuse, and
roadmap task G3 measures when it beats recomputing.

Two details make the copies fast:

- Pinned (page-locked) CPU memory. The GPU can DMA directly into it. With
  ordinary pageable memory the driver first copies into a hidden pinned
  staging buffer, which roughly halves bandwidth.
- `non_blocking=True`. With pinned memory the copy is queued on the CUDA
  stream and the Python thread returns immediately. Both directions are
  queued on the same stream, so a copy back to the GPU always runs after
  the copy to the CPU it depends on; only code that reads the CPU buffer
  directly (see `peek`) has to wait for the copy to finish.

`benchmarks/bench_offload.py` measures both effects on your GPU.
"""

from collections.abc import Iterator

import torch

from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.stats import StorageStats


class CPUOffloadKVCache:
    """Stores chunk tensors in CPU memory and returns them on `device`.

    Satisfies `CacheStorage`, so it drops into `ChunkedPrefixCache` or into
    `TieredStorage` as the fast tier. Capacity limits use the same
    prefix-safe eviction as every other backend (`PrefixLRUIndex`).
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        max_entries: int | None = None,
        pin_memory: bool = True,
        non_blocking: bool = True,
    ) -> None:
        self._device = torch.device(device)
        self._pin_memory = pin_memory
        self._non_blocking = non_blocking
        self._index = PrefixLRUIndex(max_entries)
        self._tensors: dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        """Where `get` returns tensors."""
        return self._device

    @property
    def max_entries(self) -> int | None:
        return self._index.max_entries

    @property
    def stats(self) -> StorageStats:
        return self._index.stats

    @property
    def nbytes(self) -> int:
        """CPU memory held by cached payloads."""
        return sum(t.nelement() * t.element_size() for t in self._tensors.values())

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, key: object) -> bool:
        """Whether `key` is stored; unlike `get`, does not count as a use."""
        return key in self._tensors

    def keys(self) -> Iterator[str]:
        """Stored keys, least recently used first."""
        return self._index.keys_by_recency()

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for a stored `key`; KeyError if absent."""
        if key not in self._tensors:
            raise KeyError(key)
        return self._index.parent_of(key)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        # Check before touching the index, so a bad payload evicts nothing.
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"CPUOffloadKVCache stores tensors, got {type(value).__name__}"
            )

        decision = self._index.admit(key, parent_key)
        for victim in decision.evicted:
            # The caching host allocator only reuses this pinned buffer once
            # any copy still using it has finished, so dropping it is safe.
            del self._tensors[victim]

        if not decision.admitted:
            return

        held = torch.empty(
            value.shape,
            dtype=value.dtype,
            device="cpu",
            pin_memory=self._pin_memory,
        )
        # A fresh buffer, not `value.to("cpu")`: that would return `value`
        # itself when it is already on the CPU, and the caller could then
        # change the cached data by modifying their own tensor.
        held.copy_(value, non_blocking=self._non_blocking and self._pin_memory)
        self._tensors[key] = held

    def get(self, key: str) -> torch.Tensor | None:
        held = self._tensors.get(key)
        if held is None:
            return None

        self._index.touch(key)
        # Always a new tensor on `device`, so callers can modify what they
        # get without corrupting the cache.
        restored = torch.empty_like(held, device=self._device)
        restored.copy_(held, non_blocking=self._non_blocking and self._pin_memory)
        return restored

    def peek(self, key: str) -> torch.Tensor | None:
        """The CPU copy itself, without counting a use. Do not modify it.

        Waits for pending copies first: with `non_blocking`, the copy into
        the buffer may still be running when `put` returns.
        """
        held = self._tensors.get(key)
        if held is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        return held
