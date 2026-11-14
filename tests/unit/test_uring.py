import errno
import os
import sys

import pytest

from nano_lmcache.storage import uring


@pytest.mark.parametrize(
    "fd,extents,depth",
    [
        (-1, [], 1),
        (True, [], 1),
        (0, [], 0),
        (0, [], 257),
        (0, [(-1, 1)], 1),
        (0, [(0, -1)], 1),
        (0, [(0, 1 << 31)], 1),
        (0, [(0, 1)] * 1025, 1),
        (0, [(0, uring.MAX_BYTES + 1)], 1),
    ],
)
def test_invalid_reads_are_rejected_before_native_call(monkeypatch, fd, extents, depth):
    def forbidden():
        raise AssertionError("invalid input must not load the native reader")

    monkeypatch.setattr(uring, "_library", forbidden)
    with pytest.raises(ValueError):
        uring.bulk_read(fd, extents, depth)


def test_non_linux_capability_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    # Clear a possibly cached Linux library before testing platform detection.
    uring._library.cache_clear()
    available, reason = uring.capability()
    assert not available
    assert "Linux" in reason
    with pytest.raises(uring.UringUnavailableError):
        uring.bulk_read(0, [(0, 1)])


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux io_uring")
def test_registered_reads_order_duplicates_short_reads_and_fd_position(tmp_path):
    available, reason = uring.capability()
    if not available:
        pytest.skip(reason)
    path = tmp_path / "pages"
    path.write_bytes(bytes(range(256)) * 4)
    with path.open("rb") as handle:
        handle.seek(11)
        extents = [(500, 9), (0, 16), (500, 9), (256, 0), (1000, 24)]
        try:
            result = uring.bulk_read(handle.fileno(), extents, queue_depth=2)
        except OSError as error:
            if error.errno in (errno.EPERM, errno.ENOMEM, errno.ENOBUFS):
                pytest.skip(f"registered buffers unavailable: {error}")
            raise
        expected = path.read_bytes()
        assert result == [expected[offset:offset + size] for offset, size in extents]
        assert os.lseek(handle.fileno(), 0, os.SEEK_CUR) == 11
        assert uring.bulk_read(handle.fileno(), []) == []
        with pytest.raises(uring.ShortReadError):
            uring.bulk_read(handle.fileno(), [(0, 4), (1020, 16)])
