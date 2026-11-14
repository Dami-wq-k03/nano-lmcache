"""Optional Linux io_uring bulk pread; no threads, liburing, or fallback.

A per-process cached library is built in a private temporary directory. Reads
use registered native buffers: RLIMIT_MEMLOCK and syscall permissions apply.
Payloads are bounded to 64 MiB/call (native buffer + Python output + results).
This returns host bytes, not zero-copy/GDS buffers or an NVMe performance claim.
"""

from __future__ import annotations

import ctypes as ct
import errno
import functools
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

MAX_BYTES = 64 << 20
MAX_EXTENTS = 1024
MAX_QUEUE_DEPTH = 256


class UringUnavailableError(OSError):
    """Platform, compiler, or kernel permissions do not permit io_uring."""


class ShortReadError(OSError):
    """An extent did not return its exact requested length; no partial results."""


class _ReaderLibrary(ct.CDLL):
    _build_directory: tempfile.TemporaryDirectory[str]


def _check_linux() -> None:
    if sys.platform != "linux":
        raise UringUnavailableError(errno.ENOSYS, "io_uring requires Linux")


def _error(code: int, context: str) -> OSError:
    cls = (
        UringUnavailableError
        if code in (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP)
        else OSError
    )
    return cls(code, f"{context}: {os.strerror(code)}")


@functools.lru_cache(maxsize=1)
def _library() -> ct.CDLL:
    _check_linux()
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        raise UringUnavailableError(errno.ENOENT, "io_uring needs a C compiler")
    directory = tempfile.TemporaryDirectory(prefix="nano-lmcache-uring-")
    target = Path(directory.name) / "reader.so"
    try:
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-O2",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(Path(__file__).with_name("uring_reader.c")),
                "-o",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        library = _ReaderLibrary(str(target))
    except (OSError, subprocess.SubprocessError) as error:
        directory.cleanup()
        raise UringUnavailableError(
            errno.ENOSYS, f"io_uring build/load failed: {error}"
        ) from error
    library._build_directory = directory  # keep the private build alive with the DLL
    library.nlc_uring_probe.argtypes = []
    library.nlc_uring_probe.restype = ct.c_int
    library.nlc_uring_read.argtypes = [
        ct.c_int,
        ct.c_uint,
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint32),
        ct.c_uint,
        ct.c_void_p,
        ct.POINTER(ct.c_int32),
    ]
    library.nlc_uring_read.restype = ct.c_int
    return library


def capability() -> tuple[bool, str]:
    """Probe compilation + io_uring_setup only; reads also require registration."""
    try:
        result = _library().nlc_uring_probe()
        if result < 0:
            raise _error(-result, "io_uring_setup")
    except OSError as error:
        return False, str(error)
    return True, "io_uring_setup permitted; registered reads still require resources"


def bulk_read(
    fd: int, extents: Sequence[tuple[int, int]], queue_depth: int = 32
) -> list[bytes]:
    """Read (offset, length) extents in input order, without changing file offset.

    Any setup/submission/completion error raises OSError. Short reads raise
    ShortReadError; no partial list is published. There is no async cancellation
    API. Each call owns/closes its ring; SQ/CQ depth and allocation are bounded.
    """
    if type(fd) is not int or not 0 <= fd <= 0x7FFFFFFF:
        raise ValueError("fd must be a non-negative C integer")
    if type(queue_depth) is not int or not 1 <= queue_depth <= MAX_QUEUE_DEPTH:
        raise ValueError("queue_depth must be in 1..256")
    if len(extents) > MAX_EXTENTS:
        raise ValueError("too many extents")
    offsets, lengths = [], []
    for offset, length in extents:
        if (
            type(offset) is not int
            or type(length) is not int
            or not 0 <= offset <= 0x7FFFFFFFFFFFFFFF
            or not 0 <= length <= 0x7FFFFFFF
        ):
            raise ValueError("invalid extent offset/length")
        offsets.append(offset)
        lengths.append(length)
    total = sum(lengths)
    if total > MAX_BYTES:
        raise ValueError("bulk read exceeds 64 MiB")
    _check_linux()
    if not extents:
        return []
    count = len(offsets)
    output = (ct.c_char * max(total, 1))()
    results = (ct.c_int32 * count)()
    result = _library().nlc_uring_read(
        fd,
        count,
        (ct.c_uint64 * count)(*offsets),
        (ct.c_uint32 * count)(*lengths),
        queue_depth,
        output,
        results,
    )
    if result < 0:
        raise _error(-result, "io_uring bulk read")
    for index, length in enumerate(lengths):
        if results[index] < 0:
            raise _error(-results[index], f"extent {index}")
        if results[index] != length:
            raise ShortReadError(
                errno.EIO, f"extent {index}: got {results[index]}, expected {length}"
            )
    values, position = [], 0
    for length in lengths:
        values.append(ct.string_at(ct.addressof(output) + position, length))
        position += length
    return values
