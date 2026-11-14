"""Two-GPU KV transport with explicit peer versus pinned-host staging."""

from dataclasses import dataclass
from typing import Literal

import torch

from nano_lmcache.tensor.transfer import (
    AsyncTransferEngine,
    PinnedBufferPool,
    TransferHandle,
)


@dataclass
class PeerTransfer:
    handle: TransferHandle
    transport: str
    _pool: PinnedBufferPool | None = None
    _host: torch.Tensor | None = None

    @property
    def tensor(self) -> torch.Tensor:
        return self.handle.tensor

    def wait(self) -> torch.Tensor:
        value = self.handle.wait()
        self.release()
        return value

    def release(self) -> None:
        """Drain the copy before returning an internal bounce buffer."""
        self.handle.wait()
        if self._host is not None and self._pool is not None:
            self._pool.release(self._host)
            self._host = None
        self.handle._keepalive = ()


class PeerTransferEngine:
    """Queue cross-GPU copies; never silently label host staging as P2P.

    Sources must remain immutable until completion. GPU consumers call
    ``transfer.handle.wait_stream()``; CPU consumers call ``transfer.wait()``.
    Pending copies are bounded; close drains outstanding work.
    """

    def __init__(
        self, source_device: int, target_device: int, *, max_pending: int = 8
    ) -> None:
        if source_device == target_device or max_pending <= 0:
            raise ValueError("need different GPUs and a positive pending limit")
        self.source = torch.device("cuda", source_device)
        self.target = torch.device("cuda", target_device)
        self.peer_available = bool(
            torch.cuda.can_device_access_peer(source_device, target_device)
        )
        self._source_engine = AsyncTransferEngine(device=self.source)
        self._stream = torch.cuda.Stream(device=self.target)  # type: ignore[no-untyped-call]
        self._pending: list[PeerTransfer] = []
        self._max_pending = max_pending
        self._closed = False

    def copy(
        self, source: torch.Tensor, *, mode: Literal["auto", "peer", "host"] = "auto"
    ) -> PeerTransfer:
        if self._closed:
            raise RuntimeError("peer transfer engine is closed")
        if source.device != self.source or not source.is_contiguous():
            raise ValueError("source must be contiguous on the configured source GPU")
        if mode not in ("auto", "peer", "host"):
            raise ValueError("mode must be auto, peer or host")
        if mode == "peer" and not self.peer_available:
            raise RuntimeError("CUDA peer access is unavailable on this GPU pair")
        active = []
        for transfer in self._pending:
            if transfer.handle.done:
                transfer.release()
            else:
                active.append(transfer)
        self._pending = active
        if len(active) >= self._max_pending:
            raise RuntimeError("peer transfer queue is full")
        direct = mode != "host" and self.peer_available
        host = None
        if direct:
            producer = torch.cuda.Event()  # type: ignore[no-untyped-call]
            producer.record(torch.cuda.current_stream(self.source))
            self._stream.wait_event(producer)
            input_tensor = source
        else:
            d2h = self._source_engine.to_cpu(source)
            host = d2h.tensor
            assert d2h.event is not None
            self._stream.wait_event(d2h.event)
            input_tensor = host
        with torch.cuda.device(self.target), torch.cuda.stream(self._stream):
            destination = torch.empty_like(source, device=self.target)
            destination.copy_(input_tensor, non_blocking=True)
            event = torch.cuda.Event()  # type: ignore[no-untyped-call]
            event.record(self._stream)
        transfer = PeerTransfer(
            TransferHandle(destination, event, (source, input_tensor, destination)),
            "peer" if direct else "pinned_host",
            self._source_engine.pool if host is not None else None,
            host,
        )
        self._pending.append(transfer)
        return transfer

    def close(self) -> None:
        for transfer in self._pending:
            transfer.release()
        self._pending.clear()
        self._closed = True

    def __enter__(self) -> "PeerTransferEngine":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
