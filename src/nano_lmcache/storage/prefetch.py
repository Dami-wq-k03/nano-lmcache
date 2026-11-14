"""Bounded asynchronous host reads using ThreadPoolExecutor, not io_uring."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from nano_lmcache.storage.protocol import CacheStorage


@runtime_checkable
class _BatchReader(Protocol):
    def get_many(self, keys: Sequence[str]) -> list[object | None]: ...


class AsyncPrefetchStorage:
    """Submit bounded read batches and return caller-owned result futures.

    ``max_pending`` includes running AND queued tasks, even cancelled tasks
    until the executor dequeues them. ``max_batch_size`` bounds keys per task.
    Cancel succeeds only before a read starts; running I/O cannot be interrupted.
    Exceptions propagate through Future.result(), while ordinary misses are None.
    No results are retained here, so synchronous get/put always use the backend,
    rather than an old prefetched replica. The backend must support concurrency.

    Close stops submission and optionally cancels queued reads, then joins unless
    wait=False. The caller owns the backend and closes it AFTER a waiting close.
    Caller-held futures retain their results until released by the caller.
    """

    def __init__(
        self,
        backend: CacheStorage,
        *,
        max_workers: int = 2,
        max_pending: int = 8,
        max_batch_size: int = 128,
    ) -> None:
        if max_workers <= 0 or max_pending <= 0 or max_batch_size <= 0:
            raise ValueError("worker, pending and batch limits must be positive")
        self.backend = backend
        self._max_pending = max_pending
        self._max_batch_size = max_batch_size
        self._lock = threading.RLock()
        self._pending: set[Future[list[object | None]]] = set()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="nano-lmcache-prefetch"
        )

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def prefetch(self, keys: Sequence[str]) -> Future[list[object | None]]:
        """Snapshot keys and submit, or raise RuntimeError on backpressure."""
        with self._lock:
            self._check_open()
            if len(keys) > self._max_batch_size:
                raise ValueError("prefetch batch exceeds max_batch_size")
            if len(self._pending) >= self._max_pending:
                raise RuntimeError("prefetch queue is full")
            future: Future[list[object | None]] = Future()
            self._pending.add(future)
            try:
                task = self._executor.submit(self._read, tuple(keys), future)
            except BaseException:
                self._pending.remove(future)
                raise
            task.add_done_callback(lambda task: self._finished(task, future))
            return future

    def _read(self, keys: Sequence[str], future: Future[list[object | None]]) -> None:
        # Do not cancel the executor task when the PUBLIC future is cancelled:
        # keeping its slot until dequeued prevents an unbounded cancelled queue.
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = self._load(keys)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    def _finished(
        self, task: Future[None], future: Future[list[object | None]]
    ) -> None:
        if task.cancelled():
            future.cancel()
        with self._lock:
            self._pending.discard(future)

    def _load(self, keys: Sequence[str]) -> list[object | None]:
        if isinstance(self.backend, _BatchReader):
            return self.backend.get_many(keys)
        return [self.backend.get(key) for key in keys]

    def get_many(self, keys: Sequence[str]) -> list[object | None]:
        with self._lock:
            self._check_open()
        return self._load(keys)

    def get(self, key: str) -> object | None:
        with self._lock:
            self._check_open()
        return self.backend.get(key)

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        with self._lock:
            self._check_open()
        self.backend.put(key, value, parent_key=parent_key)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("prefetch storage is closed")

    def close(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        """Running reads finish; repeat with wait=True to join a nonwaiting close."""
        with self._lock:
            self._closed = True
        # Never hold the lock while joining: completion callbacks need it.
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def __enter__(self) -> AsyncPrefetchStorage:
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
