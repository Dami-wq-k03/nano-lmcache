"""How a handoff travels from a prefill worker to a decode worker.

The transport also defines **ownership**: after `send`, the handoff
belongs to the transport; `receive` *claims* it, so exactly one decode
worker can ever continue a request. A cancelled request is `discard`ed so
its KV does not linger. Both implementations move bytes, not Python
objects, so the same checks apply whether the next hop is memory, a
shared directory, or (in a real system) the network or RDMA.
"""

import hashlib
import os
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from nano_lmcache.disaggregation.handoff import KVHandoff

HANDOFF_SUFFIX = ".handoff"
_TEMP_SUFFIX = ".tmp"
_CLAIMED_SUFFIX = ".claimed"


class DuplicateHandoffError(ValueError):
    """A handoff with this request ID is already waiting to be received."""


@runtime_checkable
class KVTransport(Protocol):
    def send(self, handoff: KVHandoff) -> int:
        """Hand over ownership; return the number of bytes transferred."""
        ...

    def receive(self, request_id: str) -> KVHandoff | None:
        """Claim the handoff, or None if absent or already claimed.

        Raises CorruptHandoffError if the bytes are damaged; the damaged
        handoff is removed either way.
        """
        ...

    def discard(self, request_id: str) -> bool:
        """Drop a pending handoff (request cancelled); True if one existed."""
        ...


class InMemoryTransport:
    """Pending handoffs as bytes in a dict, safe to share between threads."""

    def __init__(self) -> None:
        self._pending: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.bytes_sent = 0

    def send(self, handoff: KVHandoff) -> int:
        data = handoff.to_bytes()
        with self._lock:
            if handoff.request_id in self._pending:
                raise DuplicateHandoffError(handoff.request_id)
            self._pending[handoff.request_id] = data
            self.bytes_sent += len(data)
        return len(data)

    def receive(self, request_id: str) -> KVHandoff | None:
        with self._lock:
            data = self._pending.pop(request_id, None)
        return None if data is None else KVHandoff.from_bytes(data)

    def discard(self, request_id: str) -> bool:
        with self._lock:
            return self._pending.pop(request_id, None) is not None

    def pending_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._pending)


class DirectoryTransport:
    """One file per pending handoff in a directory both workers can see.

    Separate processes can share it. Writes go to a temporary file and
    are renamed into place, so a receiver never sees half a handoff.
    Receiving renames the file to a claimed name first; a rename succeeds
    for only one process, which is what makes the claim exclusive.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self.bytes_sent = 0

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, request_id: str) -> Path:
        """Where a pending handoff lives; hashed so IDs cannot escape the directory."""
        name = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return self._directory / f"{name}{HANDOFF_SUFFIX}"

    def send(self, handoff: KVHandoff) -> int:
        data = handoff.to_bytes()
        final = self.path_for(handoff.request_id)
        temp = final.with_name(
            f"{final.name}.{os.getpid()}.{threading.get_ident()}{_TEMP_SUFFIX}"
        )
        temp.write_bytes(data)
        try:
            # link() fails if `final` exists, unlike replace(): a second send
            # for the same pending request is an error, not an overwrite.
            os.link(temp, final)
        except FileExistsError:
            raise DuplicateHandoffError(handoff.request_id) from None
        finally:
            temp.unlink(missing_ok=True)
        self.bytes_sent += len(data)
        return len(data)

    def receive(self, request_id: str) -> KVHandoff | None:
        path = self.path_for(request_id)
        claimed = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}{_CLAIMED_SUFFIX}"
        )
        try:
            path.rename(claimed)
        except FileNotFoundError:
            return None
        try:
            return KVHandoff.from_bytes(claimed.read_bytes())
        finally:
            claimed.unlink(missing_ok=True)

    def discard(self, request_id: str) -> bool:
        try:
            self.path_for(request_id).unlink()
        except FileNotFoundError:
            return False
        return True

    def pending_count(self) -> int:
        return sum(1 for _ in self._directory.glob(f"*{HANDOFF_SUFFIX}"))
