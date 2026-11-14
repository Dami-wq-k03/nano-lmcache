"""Append-only page bytes with an atomic, checksummed offset index.

One writer owns a directory. Data is fsynced before publishing the index, then
its rename is directory-fsynced. A crash before publication leaves unreachable
append bytes, never a partial indexed write. Invalid metadata fails closed.
Capacity counts LIVE payload bytes; eviction does not reclaim the append file.
Read recency persists on the next mutation or clean close, not on every hit.
This is buffered host I/O, not io_uring, zero-copy, or GPUDirect Storage.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from nano_lmcache.storage.disk import _parents_first, _payload_crc32
from nano_lmcache.storage.eviction import PrefixIndex
from nano_lmcache.storage.eviction_policies import PolicyLike
from nano_lmcache.storage.serializers import Serializer
from nano_lmcache.storage.stats import StorageStats


class BytesSerializer:
    """Store already-packed page bytes without tensor dependencies."""

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError("page payload must be bytes")
        return value

    def loads(self, data: bytes) -> bytes:
        return data


@dataclass(frozen=True)
class _Extent:
    parent: str | None
    offset: int
    length: int
    checksum: int


class CoalescedDiskKVCache:
    """Prefix-safe packed disk storage; serializers remain caller-selectable.

    Adjacent requested extents share a pread, up to ``max_read_bytes`` (a single
    larger page still needs one read). Order and duplicates are preserved.
    Corruption, short reads and decode failures invalidate entire subtrees.
    After a write/checkpoint I/O error, close and reopen to recover disk state.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        serializer: Serializer | None = None,
        eviction_policy: PolicyLike = "lru",
        max_read_bytes: int = 8 << 20,
    ) -> None:
        if max_read_bytes <= 0:
            raise ValueError("max_read_bytes must be positive")
        self.directory = Path(directory)
        self.serializer = serializer if serializer is not None else BytesSerializer()
        self._index = PrefixIndex(
            max_entries, max_bytes=max_bytes, eviction_policy=eviction_policy
        )
        self._lock = threading.RLock()
        self._extents: dict[str, _Extent] = {}
        self._max_read_bytes = max_read_bytes
        self._closed = False
        self._failed = False
        self.read_ops = 0
        self.read_bytes = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest = self.directory / "pages.index"
        self._temp = self.directory / "pages.index.tmp"
        self._file = open(self.directory / "pages.data", "a+b")
        try:
            self._temp.unlink(missing_ok=True)
            self._recover()
        except BaseException:
            self._file.close()
            raise

    @property
    def nbytes(self) -> int:
        """Live serialized payload bytes, excluding metadata and dead extents."""
        with self._lock:
            return self._index.nbytes

    @property
    def stats(self) -> StorageStats:
        with self._lock:
            return replace(self._index.stats)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._index

    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(tuple(self._index.keys_by_recency()))

    def parent_of(self, key: str) -> str | None:
        with self._lock:
            return self._index.parent_of(key)

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        self.try_put(key, value, parent_key=parent_key)

    def try_put(
        self, key: str, value: object, *, parent_key: str | None = None
    ) -> bool:
        return self.put_many([(key, value, parent_key)])[0]

    def put_many(self, entries: Sequence[tuple[str, object, str | None]]) -> list[bool]:
        """Append a parent-first batch with one data fsync/index publication.

        Flags indicate each admission, not survival of later batch evictions.
        All encoding happens before admission; encoding failure evicts nothing.
        """
        with self._lock:
            self._check_open()
            encoded = []
            for key, value, parent in entries:
                if value is None:
                    raise ValueError("None is reserved for a cache miss")
                encoded.append((key, self.serializer.dumps(value), parent))
            admitted: list[bool] = []
            payloads: list[bytes] = []
            offset = self._file.seek(0, os.SEEK_END)
            for key, payload, parent in encoded:
                if key in self._index:
                    parent = self._index.parent_of(key)
                decision = self._index.admit(key, parent, size_bytes=len(payload))
                admitted.append(decision.admitted)
                if not decision.admitted:
                    continue
                for victim in decision.evicted:
                    self._extents.pop(victim)
                self._extents[key] = _Extent(
                    parent, offset, len(payload), _payload_crc32(payload)
                )
                offset += len(payload)
                payloads.append(payload)
            if payloads:
                try:
                    packed = b"".join(payloads)
                    if self._file.write(packed) != len(packed):
                        raise OSError("short page write")
                    self._file.flush()
                    os.fsync(self._file.fileno())
                    self._checkpoint()
                except BaseException:
                    self._failed = True
                    raise
            return admitted

    def get(self, key: str) -> object | None:
        return self.get_many([key])[0]

    def get_many(self, keys: Sequence[str]) -> list[object | None]:
        with self._lock:
            self._check_open()
            ordered = sorted(
                {key for key in keys if key in self._extents},
                key=lambda key: self._extents[key].offset,
            )
            values: dict[str, object] = {}
            bad: list[str] = []
            cursor = 0
            while cursor < len(ordered):
                start = self._extents[ordered[cursor]].offset
                end = start
                stop = cursor
                while stop < len(ordered):
                    extent = self._extents[ordered[stop]]
                    if stop > cursor and (
                        extent.offset != end
                        or extent.offset + extent.length - start > self._max_read_bytes
                    ):
                        break
                    end = extent.offset + extent.length
                    stop += 1
                try:
                    self.read_ops += 1
                    data = os.pread(self._file.fileno(), end - start, start)
                    self.read_bytes += len(data)
                except OSError:
                    data = b""
                for key in ordered[cursor:stop]:
                    extent = self._extents[key]
                    payload = data[
                        extent.offset - start : extent.offset - start + extent.length
                    ]
                    try:
                        if (
                            len(payload) != extent.length
                            or _payload_crc32(payload) != extent.checksum
                        ):
                            raise ValueError("short or corrupt page")
                        value = self.serializer.loads(payload)
                        if value is None:
                            raise ValueError("decoded None is reserved")
                        values[key] = value
                    except (OSError, ValueError, TypeError):
                        bad.append(key)
                cursor = stop
            for key in bad:
                if key in self._index:
                    self._index.stats.corrupted_entries += 1
                    for victim in self._index.discard_subtree(key):
                        self._extents.pop(victim)
            # Filter after ALL invalidations, even if a child preceded its parent.
            for key in keys:
                if key in values and key in self._index:
                    self._index.touch(key)
            if bad:
                self._checkpoint()
            return [values.get(key) if key in self._index else None for key in keys]

    def _checkpoint(self) -> None:
        try:
            rows = [
                [key, e.parent, e.offset, e.length, e.checksum]
                for key in self._index.keys_by_recency()
                for e in [self._extents[key]]
            ]
            payload = json.dumps(
                {"version": 1, "entries": rows}, separators=(",", ":")
            ).encode()
            with open(self._temp, "wb") as handle:
                handle.write(str(_payload_crc32(payload)).encode() + b"\n" + payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._temp, self._manifest)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            self._failed = True
            raise

    def _recover(self) -> None:
        try:
            checksum, separator, payload = self._manifest.read_bytes().partition(b"\n")
            if not separator or int(checksum) != _payload_crc32(payload):
                return
            metadata = json.loads(payload)
            if metadata["version"] != 1 or not isinstance(metadata["entries"], list):
                return
            size = os.fstat(self._file.fileno()).st_size
            found: dict[str, _Extent] = {}
            for key, parent, offset, length, crc in metadata["entries"]:
                if (
                    not isinstance(key, str)
                    or key in found
                    or (parent is not None and not isinstance(parent, str))
                    or any(type(n) is not int for n in (offset, length, crc))
                    or offset < 0
                    or length < 0
                    or not 0 <= crc <= 0xFFFFFFFF
                ):
                    return
                found[key] = _Extent(parent, offset, length, crc)
            # Short/missing extents and their descendants are unreachable.
            restorable = {
                key: (e.parent, rank, e.length)
                for rank, (key, e) in enumerate(found.items(), 1)
                if e.offset + e.length <= size
            }
        except (OSError, ValueError, TypeError, KeyError):
            return
        for key in _parents_first(restorable):
            self._index.restore(key, found[key].parent, size_bytes=found[key].length)
            self._extents[key] = found[key]
        for key, (_, rank, _) in restorable.items():
            if key in self._index:
                self._index.restore_access(key, rank)
        # Payload checksums are validated lazily by get/get_many.

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("disk storage is closed or failed; reopen to recover")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if not self._failed:
                    self._checkpoint()
            finally:
                self._closed = True
                self._file.close()

    def __enter__(self) -> CoalescedDiskKVCache:
        self._check_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
