import pytest

from nano_lmcache.experimental.dataplane import (
    KVLayout,
    KVPageDirectory,
    KVPageRef,
    PageLocation,
    PageState,
)


def make_layout(**overrides: object) -> KVLayout:
    values: dict[str, object] = {
        "model_namespace": "model/v1",
        "num_layers": 2,
        "num_kv_heads": 4,
        "head_dim": 8,
        "page_tokens": 16,
        "dtype": "float16",
    }
    values.update(overrides)
    return KVLayout(**values)  # type: ignore[arg-type]


def make_ref(
    logical_key: str = "page-0",
    *,
    page_id: int = 0,
    location: PageLocation = PageLocation.GPU,
    layout: KVLayout | None = None,
    state: PageState = PageState.READY,
) -> KVPageRef:
    return KVPageRef(
        logical_key=logical_key,
        page_id=page_id,
        token_range=(0, 16),
        location=location,
        layout=layout or make_layout(),
        state=state,
    )


def test_layout_describes_both_supported_attention_axis_orders():
    nhd = make_layout(key_layout="NHD", value_layout="NHD")
    hnd = make_layout(key_layout="HND", value_layout="HND")

    assert nhd.tensor_shape == (16, 4, 8)
    assert hnd.tensor_shape == (4, 16, 8)
    assert nhd.storage_shape == (2, 2, 16, 4, 8)
    assert nhd.page_nbytes == 2 * 2 * 16 * 4 * 8 * 2
    assert nhd.nbytes_for_tokens(3) == 2 * 2 * 3 * 4 * 8 * 2


def test_layout_rejects_unknown_or_invalid_dimensions():
    with pytest.raises(ValueError, match="page_tokens"):
        make_layout(page_tokens=0)

    with pytest.raises(ValueError, match="key_layout"):
        make_layout(key_layout="bad")

    with pytest.raises(ValueError, match="dtype"):
        make_layout(dtype="made_up")


def test_page_ref_has_explicit_transfer_lifecycle():
    ref = make_ref(state=PageState.RESERVED)

    ref.begin_transfer(ready_event="copy-event")
    assert ref.state is PageState.TRANSFERRING
    assert ref.ready_event == "copy-event"

    ref.mark_ready()
    assert ref.state is PageState.READY
    assert ref.is_ready

    ref.begin_eviction()
    ref.mark_evicted()
    assert ref.state is PageState.EVICTED
    assert not ref.is_ready

    with pytest.raises(ValueError, match="cannot transition"):
        ref.mark_ready()


def test_page_ref_rejects_invalid_token_range():
    with pytest.raises(ValueError, match="token_range"):
        KVPageRef(
            logical_key="page-0",
            page_id=0,
            token_range=(0, 17),
            location=PageLocation.GPU,
            layout=make_layout(),
        )


def test_directory_keeps_logical_page_separate_from_physical_replicas():
    directory = KVPageDirectory()
    gpu = make_ref(page_id=3, location=PageLocation.GPU)
    pinned = make_ref(page_id=9, location=PageLocation.PINNED_CPU)
    nvme = make_ref(page_id="segment-1", location=PageLocation.NVME)

    directory.register(gpu, authoritative=True)
    directory.register(pinned)
    directory.register(nvme)

    assert directory.get("page-0") is gpu
    assert directory.get("page-0", location=PageLocation.PINNED_CPU) is pinned
    assert directory.get("page-0", location=PageLocation.REMOTE) is None
    assert directory.authoritative("page-0") is gpu
    assert directory.replicas("page-0") == (gpu, pinned, nvme)
    assert directory.replica_count == 3

    gpu.begin_eviction()
    gpu.mark_evicted()
    assert directory.get("page-0") is pinned

    directory.set_authoritative(nvme)
    assert directory.get("page-0") is nvme


def test_directory_rejects_incompatible_replicas():
    directory = KVPageDirectory()
    directory.register(make_ref())

    with pytest.raises(ValueError, match="layout"):
        directory.register(
            make_ref(
                page_id=1,
                location=PageLocation.PINNED_CPU,
                layout=make_layout(head_dim=16),
            )
        )

    with pytest.raises(ValueError, match="authoritative"):
        directory.set_authoritative(
            make_ref(
                page_id=2,
                location=PageLocation.NVME,
                state=PageState.RESERVED,
            )
        )


def test_directory_promotes_a_ready_replica_when_authoritative_is_removed():
    directory = KVPageDirectory()
    gpu = make_ref(page_id=0, location=PageLocation.GPU)
    cpu = make_ref(page_id=1, location=PageLocation.CPU)
    directory.register(gpu, authoritative=True)
    directory.register(cpu)

    directory.remove(gpu)

    assert directory.get("page-0") is cpu
    assert directory.authoritative("page-0") is cpu
    assert directory.logical_page_count == 1
