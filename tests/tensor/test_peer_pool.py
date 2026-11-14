import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.tensor.kernels import (  # noqa: E402
    gather_pages_gpu,
    scatter_pages_gpu_,
)
from nano_lmcache.tensor.peer import PeerTransferEngine  # noqa: E402
from nano_lmcache.tensor.transfer import AsyncTransferEngine, PinnedBufferPool  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.gpu]


def test_pool_is_bounded_and_rejects_double_release():
    source = torch.empty(16, device="cuda")
    pool = PinnedBufferPool(max_bytes=source.nbytes)
    buffer = pool.acquire(source)
    with pytest.raises(MemoryError):
        pool.acquire(source)
    pool.release(buffer)
    with pytest.raises(ValueError, match="checked out"):
        pool.release(buffer)
    for _ in range(10_000):
        buffer = pool.acquire(source)
        pool.release(buffer)
    assert pool.allocated_bytes == source.nbytes
    assert pool.free_bytes == source.nbytes
    small = pool.acquire(torch.empty(8, device="cuda"))
    assert pool.allocated_bytes == small.nbytes
    pool.release(small)


def test_transfer_waits_for_producer_stream(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("device-wide synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    producer = torch.cuda.Stream()
    engine = AsyncTransferEngine(device=f"cuda:{torch.cuda.current_device()}")
    with torch.cuda.stream(producer):
        torch.cuda._sleep(1_000_000)
        source = torch.full((8192,), 19.0, device="cuda")
        transfer = engine.to_cpu(source)
    assert torch.equal(transfer.wait(), torch.full((8192,), 19.0))


@pytest.mark.parametrize("direction", [(0, 1), (1, 0)])
def test_cross_gpu_pages_remap_without_global_sync(direction, monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")

    def forbidden(*args, **kwargs):
        raise AssertionError("device-wide synchronization is forbidden")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    src, dst = direction
    with torch.cuda.device(src):
        pages = torch.randn(32, 2, 2, 8, 16, device=f"cuda:{src}")
        ids = torch.tensor([17, 2, 31, 0], device=f"cuda:{src}")
        packed = gather_pages_gpu(pages, ids)
    with PeerTransferEngine(src, dst) as engine:
        transfer = engine.copy(packed)
        assert transfer.transport == (
            "peer" if engine.peer_available else "pinned_host"
        )
        with torch.cuda.device(dst):
            ready = transfer.handle.wait_stream()
            destination = torch.zeros_like(pages, device=f"cuda:{dst}")
            target_ids = torch.tensor([4, 11, 9, 20], device=f"cuda:{dst}")
            scatter_pages_gpu_(destination, target_ids, ready)
            actual = destination[target_ids].cpu()
        torch.testing.assert_close(actual, packed.cpu(), rtol=0, atol=0)
        transfer.release()
        if not engine.peer_available:
            with pytest.raises(RuntimeError, match="unavailable"):
                engine.copy(packed, mode="peer")
