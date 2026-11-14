"""Asynchronous CUDA copies and a small pinned-memory reuse pool."""

from collections import defaultdict
from dataclasses import dataclass

import torch

from nano_lmcache.experimental.dataplane import KVPageRef, PageLocation, PageState


@dataclass(slots=True)
class TransferHandle:
    """A copy queued on a CUDA stream."""

    tensor: torch.Tensor
    event: torch.cuda.Event | None
    _keepalive: tuple[torch.Tensor, ...] = ()
    page_ref: KVPageRef | None = None

    @property
    def done(self) -> bool:
        """Whether the queued operation has completed."""
        return self.event is None or bool(self.event.query())

    def wait(self) -> torch.Tensor:
        """Wait for this copy only and return its destination tensor."""
        if self.event is not None:
            self.event.synchronize()
        return self.tensor

    def wait_stream(self, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
        """Make a consuming CUDA stream wait without blocking the host."""
        if self.tensor.device.type != "cuda":
            raise ValueError("CPU consumers must use wait() before reading")
        consumer = stream or torch.cuda.current_stream(self.tensor.device)
        if self.event is not None:
            consumer.wait_event(self.event)
        self.tensor.record_stream(consumer)
        return self.tensor


class PinnedBufferPool:
    """Reuse page-locked CPU buffers by shape and dtype."""

    def __init__(self, *, max_bytes: int | None = None) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive or None")
        self.max_bytes = max_bytes
        self._free: defaultdict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = defaultdict(list)
        self._free_bytes = 0
        self._allocated_bytes = 0
        self._owned: dict[int, torch.Tensor] = {}
        self._in_use: set[int] = set()

    @property
    def free_bytes(self) -> int:
        return self._free_bytes

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    def acquire(self, source: torch.Tensor) -> torch.Tensor:
        """Return a pinned CPU buffer with the same shape and dtype."""
        if source.device.type == "meta":
            raise ValueError("cannot allocate a transfer buffer for a meta tensor")
        key = (tuple(source.shape), source.dtype)
        if self._free[key]:
            buffer = self._free[key].pop()
            self._free_bytes -= buffer.nbytes
            if not self._free[key]:
                del self._free[key]
            self._in_use.add(buffer.data_ptr())
            return buffer
        if not self._free[key]:
            del self._free[key]
        while (
            self.max_bytes is not None
            and self._allocated_bytes + source.nbytes > self.max_bytes
        ):
            if not self._free:
                raise MemoryError("pinned buffer pool is full")
            old_key = next(iter(self._free))
            old = self._free[old_key].pop()
            self._free_bytes -= old.nbytes
            self._allocated_bytes -= old.nbytes
            self._owned.pop(old.data_ptr())
            if not self._free[old_key]:
                del self._free[old_key]
        buffer = torch.empty(
            source.shape, dtype=source.dtype, device="cpu", pin_memory=True
        )
        self._allocated_bytes += buffer.nbytes
        self._owned[buffer.data_ptr()] = buffer
        self._in_use.add(buffer.data_ptr())
        return buffer

    def release(self, buffer: torch.Tensor) -> None:
        """Return a completed buffer; the caller must first wait for its event."""
        if buffer.device.type != "cpu" or not buffer.is_pinned():
            raise ValueError("only pinned CPU tensors can be returned")
        pointer = buffer.data_ptr()
        if id(self._owned.get(pointer)) != id(buffer) or pointer not in self._in_use:
            raise ValueError("buffer is not checked out from this pool")
        self._in_use.remove(pointer)
        self._free[(tuple(buffer.shape), buffer.dtype)].append(buffer)
        self._free_bytes += buffer.nbytes


class AsyncTransferEngine:
    """Queue D2H/H2D copies without synchronizing the whole device."""

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
        pool: PinnedBufferPool | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("AsyncTransferEngine requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if stream is not None and stream.device != self.device:
            raise ValueError("stream must belong to the transfer device")
        self.stream = stream or torch.cuda.Stream(
            device=self.device
        )
        self.pool = pool or PinnedBufferPool()

    def to_cpu(self, source: torch.Tensor) -> TransferHandle:
        """Queue a device-to-pinned-host copy and return immediately."""
        if source.device.type != self.device.type or (
            self.device.index is not None and source.device.index != self.device.index
        ):
            raise ValueError(f"source is on {source.device}, expected {self.device}")
        destination = self.pool.acquire(source)
        self.stream.wait_stream(torch.cuda.current_stream(source.device))
        source.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            destination.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        return TransferHandle(destination, event, (source, destination))

    def to_device(self, source: torch.Tensor) -> TransferHandle:
        """Queue a pinned-host-to-device copy and return immediately."""
        if source.device.type != "cpu" or not source.is_pinned():
            raise ValueError("to_device requires a pinned CPU tensor")
        destination = torch.empty_like(source, device=self.device)
        with torch.cuda.stream(self.stream):
            destination.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        return TransferHandle(destination, event, (source, destination))

    def to_cpu_page(self, ref: KVPageRef, source: torch.Tensor) -> TransferHandle:
        """Queue D2H and mark ``ref`` transferring until completion."""
        if ref.location not in (PageLocation.CPU, PageLocation.PINNED_CPU):
            raise ValueError("a D2H page must target a CPU location")
        if ref.state is not PageState.RESERVED:
            raise ValueError(f"cannot transfer a {ref.state.value} page")
        handle = self.to_cpu(source)
        ref.begin_transfer(ready_event=handle.event)
        handle.page_ref = ref
        return handle

    def to_device_page(self, ref: KVPageRef, source: torch.Tensor) -> TransferHandle:
        """Queue H2D and mark ``ref`` transferring until completion."""
        if ref.location is not PageLocation.GPU:
            raise ValueError("an H2D page must target the GPU location")
        if ref.state is not PageState.RESERVED:
            raise ValueError(f"cannot transfer a {ref.state.value} page")
        handle = self.to_device(source)
        ref.begin_transfer(ready_event=handle.event)
        handle.page_ref = ref
        return handle

    @staticmethod
    def complete_page(ref: KVPageRef, handle: TransferHandle) -> bool:
        """Publish a page after its event; stale or cancelled refs are ignored."""
        if (
            ref.state is not PageState.TRANSFERRING
            or handle.page_ref is not ref
            or ref.ready_event is not handle.event
        ):
            return False
        handle.wait()
        ref.mark_ready()
        return True

    def cancel_page(self, ref: KVPageRef, handle: TransferHandle) -> bool:
        """Drain a cancelled copy before releasing its pinned destination."""
        if ref.state is not PageState.TRANSFERRING or handle.page_ref is not ref:
            return False
        ref.mark_failed()
        handle.wait()
        if ref.location in (PageLocation.CPU, PageLocation.PINNED_CPU):
            self.pool.release(handle.tensor)
        return True
