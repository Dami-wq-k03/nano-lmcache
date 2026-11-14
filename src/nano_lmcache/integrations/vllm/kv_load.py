"""Engine-independent chunk load plans and paged KV scatter (optional torch)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nano_lmcache.core.chunking import chunk_tokens, make_chunk_key
from nano_lmcache.storage.protocol import CacheStorage

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ChunkLoadPlan:
    request_id: str
    key: str
    slot_mapping: tuple[int, ...]


def chunk_keys(
    tokens: Sequence[int], chunk_size: int, namespace: str
) -> tuple[str, ...]:
    """Use the same chained keys as ChunkedPrefixCache, without fetching KV."""
    keys = []
    parent = None
    for chunk in chunk_tokens(tokens, chunk_size):
        if len(chunk) < chunk_size:
            break
        parent = make_chunk_key(chunk, namespace=namespace, prefix_key=parent)
        keys.append(parent)
    return tuple(keys)


class KVLoadAdapter:
    """Scatter complete CPU chunks into registered [blocks, heads, tokens, 2D] views.

    Copies run on each device's current stream. A per-layer event lets attention
    on another stream wait without synchronizing the host. CPU loads are immediate.
    """

    def __init__(self, storage: CacheStorage, block_size: int) -> None:
        self.storage = storage
        self.block_size = block_size
        self.layers: dict[str, torch.Tensor] = {}
        self.events: dict[str, Any] = {}
        self.errors: set[int] = set()
        self.failed_requests: set[str] = set()
        self.started = False

    def reset(self) -> None:
        # Drain writes before rebinding metadata or releasing registered buffers.
        for event in self.events.values():
            event.synchronize()
        self.events.clear()
        self.errors.clear()
        self.failed_requests.clear()
        self.started = False

    def start(self, plans: Sequence[ChunkLoadPlan]) -> None:
        import torch

        if self.started or not plans:
            return
        self.started = True
        if not self.layers:
            raise RuntimeError("KV cache layers must be registered before loading")
        for plan in plans:
            # Deserialization is an untrusted cache boundary. Invalid/missing
            # payloads are load failures, not engine exceptions.
            try:
                chunk = self.storage.get(plan.key)
                self._validate(chunk, plan.slot_mapping)
            except Exception:
                self.errors.update(
                    slot // self.block_size for slot in plan.slot_mapping
                )
                self.failed_requests.add(plan.request_id)
                continue
            assert isinstance(chunk, torch.Tensor)
            for index, layer in enumerate(self.layers.values()):
                slots = torch.tensor(plan.slot_mapping, device=layer.device)
                source = torch.cat((chunk[index, 0], chunk[index, 1]), dim=-1)
                layer[slots // self.block_size, :, slots % self.block_size] = (
                    source.permute(1, 0, 2).to(layer.device)
                )
        for name, layer in self.layers.items():
            if layer.device.type == "cuda":
                with torch.cuda.device(layer.device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(layer.device))
                    self.events[name] = event

    def wait(self, layer_name: str) -> None:
        import torch

        event = self.events.get(layer_name)
        if event is not None:
            torch.cuda.current_stream(self.layers[layer_name].device).wait_event(event)

    def _validate(self, chunk: object, slots: tuple[int, ...]) -> None:
        import torch

        first = next(iter(self.layers.values()))
        expected = (
            len(self.layers),
            2,
            first.shape[1],
            self.block_size,
            first.shape[-1] // 2,
        )
        if (
            not isinstance(chunk, torch.Tensor)
            or chunk.device.type != "cpu"
            or tuple(chunk.shape) != expected
            or chunk.dtype != first.dtype
            or not bool(torch.isfinite(chunk).all())
        ):
            raise ValueError("Missing or incompatible KV chunk")
        if len(slots) != self.block_size or len(set(slots)) != len(slots):
            raise ValueError("Load plan must address one complete chunk")
        for layer in self.layers.values():
            if (
                layer.ndim != 4
                or layer.shape[1:] != first.shape[1:]
                or layer.shape[2] != self.block_size
                or layer.shape[-1] % 2
                or layer.dtype != first.dtype
                or any(
                    slot < 0 or slot >= layer.shape[0] * self.block_size
                    for slot in slots
                )
            ):
                raise ValueError("Incompatible paged KV view or slots")
