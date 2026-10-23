"""Disk-backed cache: one file per entry, surviving process restarts.

This is what makes an external KV cache more than vLLM's built-in prefix
caching: entries outlive the process that computed them, so a restarted
server, or another process on the same machine, can reuse them.

File layout, one file per key inside `directory`:

    <sha256(key)>.entry   = one JSON header line + "\\n" + payload bytes
    header                = {"key": ..., "parent_key": ..., "payload_crc32": ...}

The file name is a hash of the key rather than the key itself, so no key
(not even "../x") can address a path outside the directory. The header
records the key and its parent, which lets a new process rebuild the
prefix chains. A file's modification time records when it was last used,
so least-recently-used order also survives a restart.

The directory belongs to the cache: on open, the cache deletes leftover
temporary files, entries with missing or mismatched payload checksums, and
entries whose parent is gone. Entries written before checksums were added are
discarded on open and regenerated as cache misses.

Only one process should write to a directory at a time: each process
keeps its own index, so two writers would evict without seeing each
other's entries.
"""

import hashlib
import json
import os
import time
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.serializers import JsonSerializer, Serializer
from nano_lmcache.storage.stats import StorageStats

ENTRY_SUFFIX = ".entry"
TEMP_SUFFIX = ".tmp"
EntryHeader = dict[str, str | int | None]


def _payload_crc32(payload: bytes) -> int:
    """Return the unsigned CRC32 recorded for one entry payload."""
    return zlib.crc32(payload) & 0xFFFFFFFF


def _has_matching_payload_crc32(header: EntryHeader, payload: bytes) -> bool:
    """Whether an entry has an exact integer CRC32 matching its payload."""
    expected_crc32 = header.get("payload_crc32")
    return (
        type(expected_crc32) is int
        and expected_crc32 == _payload_crc32(payload)
    )


class DiskKVCache:
    """Optionally bounded, prefix-safe cache persisted in a directory.

    Eviction follows the same rules as `MemoryKVCache`, through a shared
    `PrefixLRUIndex`; evicting an entry deletes its file.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_entries: int | None = None,
        serializer: Serializer | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._serializer = serializer if serializer is not None else JsonSerializer()
        self._index = PrefixLRUIndex(max_entries)
        # Last-used times are written as file mtimes. Keeping them strictly
        # increasing makes the order exact even when two accesses land in
        # the same clock tick.
        self._last_tick = 0
        self._recover()

    @property
    def directory(self) -> Path:
        """Where the entry files live."""
        return self._directory

    @property
    def serializer(self) -> Serializer:
        """How payloads are encoded on disk."""
        return self._serializer

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when unbounded."""
        return self._index.max_entries

    @property
    def stats(self) -> StorageStats:
        """Eviction, rejection, and corruption totals for this object."""
        return self._index.stats

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        """Whether `key` is stored; unlike `get`, does not count as a use."""
        return key in self._index

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for a stored `key`; KeyError if absent."""
        if key not in self._index:
            raise KeyError(key)
        return self._index.parent_of(key)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        # Encode first: a payload that cannot be stored must not change the
        # index or evict anything.
        payload = self._serializer.dumps(value)

        decision = self._index.admit(key, parent_key)
        # Victims are always leaves, so deleting them one at a time never
        # leaves an orphan on disk, even if the process dies halfway.
        for victim in decision.evicted:
            self._path(victim).unlink(missing_ok=True)

        if not decision.admitted:
            return

        # The chain position of an existing key never changes, so keep the
        # parent recorded at first admission.
        header = {
            "key": key,
            "parent_key": self._index.parent_of(key),
            "payload_crc32": _payload_crc32(payload),
        }
        self._write_atomically(self._path(key), header, payload)
        self._mark_used(self._path(key))

    def get(self, key: str) -> object | None:
        if key not in self._index:
            return None

        path = self._path(key)
        try:
            header, payload = self._read(path)
        except FileNotFoundError:
            self._discard_subtree(key)
            return None
        except (OSError, ValueError, UnicodeDecodeError):
            self._discard_subtree(key)
            self._index.stats.corrupted_entries += 1
            return None

        if not _has_matching_payload_crc32(header, payload):
            self._discard_subtree(key)
            if "payload_crc32" in header:
                self._index.stats.corrupted_entries += 1
            return None

        self._index.touch(key)
        self._mark_used(path)
        return self._serializer.loads(payload)

    def _discard_subtree(self, key: str) -> None:
        """Remove an unusable entry and every descendant from disk and index."""
        for victim in self._index.discard_subtree(key):
            self._path(victim).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._directory / f"{name}{ENTRY_SUFFIX}"

    def _write_atomically(
        self, path: Path, header: EntryHeader, payload: bytes
    ) -> None:
        """Write to a temp file, then rename it into place.

        A rename within one directory is atomic, so a crash leaves either
        the old file, the new file, or a stray temp file (cleaned up on the
        next open), never a half-written entry.
        """
        temp = path.with_name(f"{path.name}.{os.getpid()}{TEMP_SUFFIX}")
        with open(temp, "wb") as handle:
            handle.write(json.dumps(header, separators=(",", ":")).encode("utf-8"))
            handle.write(b"\n")
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _read(self, path: Path) -> tuple[EntryHeader, bytes]:
        data = path.read_bytes()
        header_line, separator, payload = data.partition(b"\n")
        if not separator:
            raise ValueError(f"{path.name}: missing header")

        header = json.loads(header_line)
        if not isinstance(header, dict) or not isinstance(header.get("key"), str):
            raise ValueError(f"{path.name}: malformed header")
        return header, payload

    def _mark_used(self, path: Path) -> None:
        self._last_tick = max(time.time_ns(), self._last_tick + 1)
        os.utime(path, ns=(self._last_tick, self._last_tick))

    def _recover(self) -> None:
        """Rebuild the index from the files left by earlier processes."""
        for temp in self._directory.glob(f"*{TEMP_SUFFIX}"):
            temp.unlink(missing_ok=True)

        # key -> (parent_key, last-used time)
        found: dict[str, tuple[str | None, int]] = {}
        for path in self._directory.glob(f"*{ENTRY_SUFFIX}"):
            try:
                header, payload = self._read(path)
            except (OSError, ValueError, UnicodeDecodeError):
                path.unlink(missing_ok=True)
                continue
            key = header["key"]
            assert isinstance(key, str)
            # A file must live at the path its own key hashes to; anything
            # else is a stray copy that the cache would never find or delete.
            if path != self._path(key):
                path.unlink(missing_ok=True)
                continue
            if not _has_matching_payload_crc32(header, payload):
                path.unlink(missing_ok=True)
                if "payload_crc32" in header:
                    self._index.stats.corrupted_entries += 1
                continue
            mtime = path.stat().st_mtime_ns
            found[key] = (cast(str | None, header.get("parent_key")), mtime)
            self._last_tick = max(self._last_tick, mtime)

        restorable = list(_parents_first(found))
        if self.max_entries is not None and len(restorable) > self.max_entries:
            raise ValueError(
                f"{self._directory} holds {len(restorable)} entries, more than "
                f"max_entries={self.max_entries}; open it with a larger limit "
                "or clear the directory"
            )

        for key in restorable:
            self._index.restore(key, found[key][0])
        # Restore had to go parents-first; now replay true recency.
        for key in sorted(restorable, key=lambda k: found[k][1]):
            self._index.touch(key)

        # Anything not restorable has a missing ancestor: unreachable.
        for key in found.keys() - set(restorable):
            self._path(key).unlink(missing_ok=True)


def _parents_first(found: dict[str, tuple[str | None, int]]) -> Iterator[str]:
    """Yield keys whose whole ancestor chain is present, parents first."""
    children: dict[str | None, list[str]] = {}
    for key, (parent_key, _) in found.items():
        children.setdefault(parent_key, []).append(key)

    pending = list(children.get(None, []))
    while pending:
        key = pending.pop()
        yield key
        pending.extend(children.get(key, []))
