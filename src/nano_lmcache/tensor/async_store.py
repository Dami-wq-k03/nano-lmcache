"""Bounded CUDA-to-host write-behind with owned snapshots and event ordering."""

from __future__ import annotations

import threading

import torch

from nano_lmcache.storage.async_write import (
    AsyncWriteBehindStorage,
    WriteBehindBackpressure,
)
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.tensor.transfer import (
    AsyncTransferEngine,
    PinnedBufferPool,
    TransferHandle,
)


class _TensorSink:
    def __init__(
        self, backend: CacheStorage, pool: PinnedBufferPool, lock: threading.Lock
    ) -> None:
        self.backend = backend
        self.pool = pool
        self.lock = lock

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        if isinstance(value, TransferHandle):
            try:
                # Backends may retain tensors; never publish a reusable pool buffer.
                self.backend.put(key, value.wait().clone(), parent_key=parent_key)
            finally:
                value.wait()
                with self.lock:
                    self.pool.release(value.tensor)
        else:
            self.backend.put(key, value, parent_key=parent_key)

    def get(self, key: str) -> object | None:
        return self.backend.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self.backend if hasattr(self.backend, "__contains__") else False


class AsyncTensorWriteStorage:
    """Snapshot tensors, enqueue D2H, and serialize/store them on a worker.

    The queue and pinned bytes are bounded. Saturation drains accepted writes
    before retrying rather than dropping a prefix. Host snapshots are owned;
    CUDA snapshots run on the producer stream before asynchronous export.
    Backend reads/writes must support concurrency. Explicit close is required.
    GPU gather/snapshot work is still submitted on the request's stream.
    """

    def __init__(
        self,
        backend: CacheStorage,
        *,
        max_pending: int = 32,
        max_pinned_bytes: int = 256 << 20,
    ) -> None:
        self.backend = backend
        self.pool = PinnedBufferPool(max_bytes=max_pinned_bytes)
        self._engine: AsyncTransferEngine | None = None
        self._lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._closed = False
        self.stalls = 0
        self._writer = AsyncWriteBehindStorage(
            _TensorSink(backend, self.pool, self._pool_lock), max_pending=max_pending
        )

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("async tensor storage requires a tensor")
        with self._lock:
            if self._closed:
                raise RuntimeError("async tensor storage is closed")
            snapshot = value.detach().clone()
            payload: object = snapshot
            if snapshot.is_cuda:
                if self._engine is None:
                    self._engine = AsyncTransferEngine(
                        device=snapshot.device, pool=self.pool
                    )
                if snapshot.device != self._engine.device:
                    raise ValueError("one async tensor store supports one CUDA device")
                try:
                    payload = self._export(snapshot)
                except MemoryError:
                    self.stalls += 1
                    self.flush()
                    payload = self._export(snapshot)
            try:
                try:
                    self._writer.put(key, payload, parent_key=parent_key)
                except WriteBehindBackpressure:
                    self.stalls += 1
                    self.flush()
                    self._writer.put(key, payload, parent_key=parent_key)
            except BaseException:
                if isinstance(payload, TransferHandle):
                    payload.wait()
                    with self._pool_lock:
                        self.pool.release(payload.tensor)
                raise

    def _export(self, snapshot: torch.Tensor) -> TransferHandle:
        assert self._engine is not None
        with self._pool_lock:
            return self._engine.to_cpu(snapshot)

    def get(self, key: str) -> object | None:
        return self._writer.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._writer

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._writer.close()
