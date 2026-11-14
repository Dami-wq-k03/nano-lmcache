"""Bounded write-behind storage for moving cache writes off a request path."""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from nano_lmcache.storage.protocol import CacheStorage


class WriteBehindBackpressure(RuntimeError):
    """The bounded write queue has no room for another payload."""


@dataclass(frozen=True, slots=True)
class _Write:
    key: str
    value: object
    parent_key: str | None


class AsyncWriteBehindStorage:
    """Queue immutable writes and publish them only after backend completion.

    ``prepare`` runs on the caller thread and must return an owned value. The
    default uses the payload as-is, which is appropriate for immutable bytes
    and strings; tensor callers should pass a copy-to-host function.
    """

    def __init__(
        self,
        backend: CacheStorage,
        *,
        max_pending: int = 64,
        prepare: Callable[[object], object] | None = None,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.backend = backend
        self._queue: queue.Queue[_Write | None] = queue.Queue(max_pending)
        self._prepare = prepare or (lambda value: value)
        self._errors: list[BaseException] = []
        self._error_lock = threading.Lock()
        self._pending_keys: dict[str, int] = {}
        self._pending_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="nano-lmcache-write-behind",
            daemon=True,
        )
        self._worker.start()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._error_lock:
            return tuple(self._errors)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        if value is None:
            raise ValueError("None is reserved for a cache miss")
        prepared = self._prepare(value)
        with self._pending_lock:
            if self._closed:
                raise RuntimeError("write-behind storage is closed")
            self._pending_keys[key] = self._pending_keys.get(key, 0) + 1
            try:
                self._queue.put_nowait(_Write(key, prepared, parent_key))
            except queue.Full as error:
                self._decrement_pending(key)
                raise WriteBehindBackpressure(
                    "write-behind queue is full; call flush or increase max_pending"
                ) from error

    def get(self, key: str) -> object | None:
        """Wait for a pending key, then return only committed data."""
        with self._pending_lock:
            pending = key in self._pending_keys
        if pending:
            self.flush()
        return self.backend.get(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._pending_lock:
            if key in self._pending_keys:
                return True
        return key in self.backend if hasattr(self.backend, "__contains__") else False

    def flush(self) -> None:
        """Wait for all queued writes and surface worker failures."""
        self._queue.join()
        errors = self.errors
        if errors:
            raise RuntimeError("write-behind worker failed") from errors[0]

    def close(self, *, flush: bool = True) -> None:
        """Reject new writes and stop the worker even when a write failed.

        With flush=False, skip the initial flush but still drain accepted writes
        before the FIFO stop marker. Backend I/O must eventually return.
        """
        with self._close_lock:
            with self._pending_lock:
                if self._closed:
                    return
                self._closed = True
            try:
                if flush:
                    self.flush()
            finally:
                self._queue.put(None)
                self._worker.join()
            if self.errors:
                raise RuntimeError("write-behind worker failed") from self.errors[0]

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self.backend.put(
                    item.key, item.value, parent_key=item.parent_key
                )
            except BaseException as error:
                with self._error_lock:
                    self._errors.append(error)
            finally:
                if item is not None:
                    with self._pending_lock:
                        self._decrement_pending(item.key)
                self._queue.task_done()

    def _decrement_pending(self, key: str) -> None:
        count = self._pending_keys[key] - 1
        if count:
            self._pending_keys[key] = count
        else:
            del self._pending_keys[key]
