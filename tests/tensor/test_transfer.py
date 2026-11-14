import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.experimental.dataplane import (  # noqa: E402
    KVLayout,
    KVPageRef,
    PageLocation,
)
from nano_lmcache.tensor.transfer import (  # noqa: E402
    AsyncTransferEngine,
    PinnedBufferPool,
)

pytestmark = [pytest.mark.torch, pytest.mark.gpu]


def test_async_transfer_round_trip_without_global_sync():
    source = torch.randn(2, 3, 16, device="cuda")
    pool = PinnedBufferPool(max_bytes=source.nbytes * 2)
    engine = AsyncTransferEngine(device="cuda", pool=pool)

    host_handle = engine.to_cpu(source)
    assert not host_handle.done or host_handle.done
    host = host_handle.wait()
    assert host.is_pinned()
    assert torch.equal(host, source.cpu())

    restored = engine.to_device(host).wait()
    assert restored.is_cuda
    assert torch.equal(restored, source)

    pool.release(host)
    reused = pool.acquire(source)
    assert reused.data_ptr() == host.data_ptr()
    pool.release(reused)


def test_page_ref_becomes_ready_only_after_copy_event():
    source = torch.randn(2, 3, 16, device="cuda")
    ref = KVPageRef(
        logical_key="page-0",
        page_id=0,
        token_range=(0, 3),
        location=PageLocation.PINNED_CPU,
        layout=KVLayout(
            model_namespace="test",
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            page_tokens=3,
            dtype="float32",
        ),
    )
    engine = AsyncTransferEngine(device="cuda")

    handle = engine.to_cpu_page(ref, source)
    assert not ref.is_ready
    assert engine.complete_page(ref, handle)
    assert ref.is_ready
    assert not engine.complete_page(ref, handle)


def test_pool_rejects_non_pinned_release():
    pool = PinnedBufferPool()
    with pytest.raises(ValueError, match="pinned CPU"):
        pool.release(torch.empty(4))
