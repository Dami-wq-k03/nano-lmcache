import pytest

from nano_lmcache.experimental.dataplane import (
    KVLayout,
    PageLocation,
    ReferencePageStore,
)


def layout():
    return KVLayout(
        model_namespace="test",
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_tokens=4,
        dtype="uint8",
    )


def test_reference_store_publishes_and_falls_back_between_replicas():
    store = ReferencePageStore()
    descriptor = layout()
    gpu = store.reserve(
        logical_key="page-0",
        page_id=1,
        token_range=(0, 4),
        location=PageLocation.GPU,
        layout=descriptor,
    )
    cpu = store.reserve(
        logical_key="page-0",
        page_id=2,
        token_range=(0, 4),
        location=PageLocation.PINNED_CPU,
        layout=descriptor,
    )
    payload = bytes(range(descriptor.page_nbytes))
    assert store.complete(gpu, payload)
    assert store.complete(cpu, payload)
    assert store.load("page-0") == payload

    store.evict(gpu.ref)
    assert store.load("page-0") == payload
    assert store.directory.authoritative("page-0") is cpu.ref


def test_stale_generation_completion_is_ignored():
    store = ReferencePageStore()
    descriptor = layout()
    old = store.reserve(
        logical_key="page-0",
        page_id="old",
        token_range=(0, 4),
        location=PageLocation.NVME,
        layout=descriptor,
        generation=0,
    )
    new = store.reserve(
        logical_key="page-0",
        page_id="new",
        token_range=(0, 4),
        location=PageLocation.NVME,
        layout=descriptor,
        generation=1,
    )
    payload = bytes(descriptor.page_nbytes)
    assert not store.complete(old, payload)
    assert store.complete(new, payload)


def test_partial_page_byte_count_and_bad_payload():
    store = ReferencePageStore()
    reservation = store.reserve(
        logical_key="tail",
        page_id=0,
        token_range=(4, 6),
        location=PageLocation.CPU,
        layout=layout(),
    )
    assert reservation.expected_nbytes == layout().nbytes_for_tokens(2)
    with pytest.raises(ValueError, match="expected"):
        store.complete(reservation, b"bad")
