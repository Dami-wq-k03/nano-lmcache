"""CPU tests for the actual vLLM load adapter, without importing the engine."""

import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nano_lmcache.integrations.vllm.kv_load import ChunkLoadPlan, chunk_keys


def test_adapter_import_does_not_import_torch_or_vllm():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import nano_lmcache.integrations.vllm.kv_load; "
                "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.fixture
def setup_load():
    torch = pytest.importorskip("torch")
    from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
    from nano_lmcache.integrations.vllm.kv_load import KVLoadAdapter
    from nano_lmcache.storage.remote import RemoteCacheClient, RemoteCacheServer
    from nano_lmcache.tensor.serializers import TensorSerializer

    server = RemoteCacheServer()
    server.start()
    host, port = server.address
    writer = RemoteCacheClient(host, port, serializer=TensorSerializer())
    reader = RemoteCacheClient(host, port, serializer=TensorSerializer())
    cache = ChunkedPrefixCache(chunk_size=4, namespace="load-test", storage=writer)
    chunks = [
        torch.arange(3 * 2 * 2 * 4 * 3, dtype=torch.float32).reshape(3, 2, 2, 4, 3)
        + index * 10000
        for index in range(3)
    ]
    cache.store(list(range(13)), chunks)
    keys = chunk_keys(list(range(13)), 4, "load-test")
    assert keys == cache.lookup(list(range(13))).keys
    adapter = KVLoadAdapter(reader, 4)
    adapter.layers = {
        f"layer.{index}": torch.full((10, 2, 4, 6), -1.0) for index in range(3)
    }
    plans = tuple(
        ChunkLoadPlan("req", key, tuple(range(block * 4, block * 4 + 4)))
        for key, block in zip(keys, (7, 2, 9), strict=True)
    )
    try:
        yield SimpleNamespace(
            adapter=adapter, plans=plans, chunks=chunks, cache=cache, server=server
        )
    finally:
        adapter.reset()
        server.close()


def test_pickled_plans_restore_all_layers_and_preserve_other_slots(setup_load):
    import torch

    data = setup_load
    plans = pickle.loads(pickle.dumps(data.plans))
    assert plans == data.plans
    data.adapter.start(plans)
    assert data.adapter.errors == set()
    assert data.adapter.events == {}  # CPU must never touch CUDA.
    for index, (name, layer) in enumerate(data.adapter.layers.items()):
        data.adapter.wait(name)
        for chunk, block in zip(data.chunks, (7, 2, 9), strict=True):
            assert torch.equal(layer[block, :, :, :3], chunk[index, 0])
            assert torch.equal(layer[block, :, :, 3:], chunk[index, 1])
        assert torch.all(layer[[0, 1, 3, 4, 5, 6, 8]] == -1)
        layer.fill_(5)
    data.adapter.start(plans)
    assert all(torch.all(layer == 5) for layer in data.adapter.layers.values())


@pytest.mark.parametrize("damage", ["missing", "bytes", "shape", "dtype", "nan", "inf"])
def test_bad_chunk_falls_back_without_partial_layer_writes(setup_load, damage):
    import torch

    from nano_lmcache.tensor.serializers import TensorSerializer

    data = setup_load
    key = data.plans[1].key
    if damage == "missing":
        data.server.storage.invalidate(key)
    else:
        chunk = data.chunks[1].clone()
        if damage == "shape":
            chunk = chunk[:2]
        elif damage == "dtype":
            chunk = chunk.to(torch.float16)
        elif damage == "nan":
            chunk[-1, 1, 0, 0, 0] = float("nan")
        elif damage == "inf":
            chunk[-1, 1, 0, 0, 0] = float("inf")
        payload = (
            b"not a tensor archive"
            if damage == "bytes"
            else TensorSerializer().dumps(chunk)
        )
        data.server.storage.put(key, payload)
    data.adapter.start(data.plans)
    assert data.adapter.errors == ({2, 9} if damage == "missing" else {2})
    assert data.adapter.failed_requests == {"req"}
    for layer in data.adapter.layers.values():
        assert torch.all(layer[2] == -1)
    data.adapter.reset()
    assert data.adapter.errors == set()
    assert data.adapter.failed_requests == set()
    assert not data.adapter.started


def test_only_additional_prefix_is_loaded(setup_load):
    import torch

    data = setup_load
    data.adapter.start(data.plans[1:])
    assert all(torch.all(layer[7] == -1) for layer in data.adapter.layers.values())
    assert data.adapter.errors == set()


@pytest.mark.parametrize(
    "slots", [(40, 41, 42, 43), (-4, -3, -2, -1), (0, 0, 1, 2), (0, 1)]
)
def test_invalid_slot_plans_do_not_mutate_buffers(setup_load, slots):
    import torch

    data = setup_load
    data.adapter.start([ChunkLoadPlan("bad", data.plans[0].key, slots)])
    assert data.adapter.failed_requests == {"bad"}
    assert all(torch.all(layer == -1) for layer in data.adapter.layers.values())


def test_load_failure_does_not_affect_other_request(setup_load):
    data = setup_load
    data.adapter.start(
        [ChunkLoadPlan("missing", "unknown", (0, 1, 2, 3)), data.plans[0]]
    )
    assert data.adapter.errors == {0}
    assert data.adapter.failed_requests == {"missing"}


def test_layer_wait_enqueues_event_and_reset_drains_it(setup_load, monkeypatch):
    import torch

    calls = []
    event = SimpleNamespace(synchronize=lambda: calls.append("drain"))
    stream = SimpleNamespace(wait_event=lambda actual: calls.append(actual))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    adapter = setup_load.adapter
    adapter.events["layer.0"] = event
    adapter.wait("layer.0")
    adapter.wait("layer.1")
    assert calls == [event]
    adapter.reset()
    assert calls == [event, "drain"]


def test_scatter_preserves_noncontiguous_registered_views(setup_load):
    import torch

    data = setup_load
    data.adapter.layers = {
        name: torch.full((10, 2, 6, 4), -1.0).transpose(-1, -2)
        for name in data.adapter.layers
    }
    data.adapter.start(data.plans)
    assert data.adapter.errors == set()
    for index, layer in enumerate(data.adapter.layers.values()):
        assert not layer.is_contiguous()
        assert torch.equal(layer[7, :, :, :3], data.chunks[0][index, 0])
        assert torch.equal(layer[2, :, :, 3:], data.chunks[1][index, 1])


def test_empty_step_does_not_record_cuda_events(setup_load, monkeypatch):
    import torch

    def unexpected_event():
        raise AssertionError("empty load step must not record an event")

    monkeypatch.setattr(torch.cuda, "Event", unexpected_event)
    setup_load.adapter.start(())
    assert setup_load.adapter.events == {}
    assert not setup_load.adapter.started


@pytest.mark.gpu
def test_cuda_load_event_orders_attention_on_another_stream(setup_load):
    import torch

    data = setup_load
    data.adapter.layers = {
        name: layer.cuda() for name, layer in data.adapter.layers.items()
    }
    data.adapter.start(data.plans)
    assert set(data.adapter.events) == set(data.adapter.layers)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for name, layer in data.adapter.layers.items():
            data.adapter.wait(name)
            index = int(name.split(".")[-1])
            assert torch.equal(layer[7, :, :, :3].cpu(), data.chunks[0][index, 0])
