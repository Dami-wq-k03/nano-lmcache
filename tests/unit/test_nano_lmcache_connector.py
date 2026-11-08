"""Call the vLLM connector directly, without an LLM, model weights or GPU.

Run with:
    .venv/bin/python -m pytest -q tests/integration/vllm/test_nano_lmcache_connector.py

Tests exercise the no-op contract and lifecycle tracing against real vLLM.
"""

# Optional dependencies must be checked before importing the connector.
# ruff: noqa: E402

import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# The tests/integration/vllm directory can appear as a namespace package.
# Check an engine submodule so CPU-only collection skips without real vLLM.
pytest.importorskip("vllm.config")


from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig

from nano_lmcache.integrations.vllm.nano_lmcache_connector import (
    NanoLMCacheConnector,
    NanoLMCacheConnectorMetadata,
)


@pytest.fixture
def make_connector():
    """Create independent connector instances with minimal configuration.

    Accept either a role name ("SCHEDULER" / "WORKER") or KVConnectorRole.
    Configuration objects are created per call; no model or engine is loaded.
    """

    def create(role: str | KVConnectorRole) -> NanoLMCacheConnector:
        if isinstance(role, str):
            role = KVConnectorRole[role]

        config = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="NanoLMCacheConnector",
                kv_role="kv_both",
            )
        )
        cache_config = KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
        )
        return NanoLMCacheConnector(config, role, cache_config)

    return create


@pytest.fixture
def scheduler(make_connector):
    return make_connector("SCHEDULER")


@pytest.fixture
def worker(make_connector):
    connector = make_connector("WORKER")
    connector.bind_connector_metadata(NanoLMCacheConnectorMetadata())
    try:
        yield connector
    finally:
        connector.clear_connector_metadata()


@pytest.fixture
def request_stub():
    return SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(12)),
        num_computed_tokens=0,
    )


@pytest.mark.parametrize(
    "role", ["SCHEDULER", "WORKER", KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER]
)
def test_connector_can_be_constructed(make_connector, role):
    connector = make_connector(role)

    expected_role = KVConnectorRole[role] if isinstance(role, str) else role
    assert connector.role == expected_role
    assert not connector.has_connector_metadata()


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_factory_loads_external_connector(role):
    """Exercise the external-module route used by --kv-transfer-config."""
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NanoLMCacheConnector",
            kv_connector_module_path=(
                "nano_lmcache.integrations.vllm.nano_lmcache_connector"
            ),
            kv_role="kv_both",
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )
    cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    connector = KVConnectorFactory.create_connector(config, role, cache_config)

    assert isinstance(connector, NanoLMCacheConnector)
    assert connector.role == role
    assert not connector.has_connector_metadata()


@pytest.mark.parametrize("num_computed_tokens", [0, 4, 8])
def test_scheduler_reports_zero_matches(scheduler, request_stub, num_computed_tokens):
    request_stub.num_computed_tokens = num_computed_tokens
    original_tokens = list(request_stub.prompt_token_ids)

    for _ in range(2):
        assert scheduler.get_num_new_matched_tokens(
            request_stub, num_computed_tokens
        ) == (0, False)
    assert request_stub.prompt_token_ids == original_tokens
    assert request_stub.num_computed_tokens == num_computed_tokens


def test_scheduler_accepts_allocation(scheduler, request_stub):
    blocks = KVCacheBlocks(blocks=([KVCacheBlock(7), KVCacheBlock(2)],))
    original_tokens = list(request_stub.prompt_token_ids)
    original_block_ids = blocks.get_block_ids()

    assert scheduler.update_state_after_alloc(request_stub, blocks, 0) is None
    assert request_stub.prompt_token_ids == original_tokens
    assert request_stub.num_computed_tokens == 0
    assert blocks.get_block_ids() == original_block_ids
    assert all(block.ref_cnt == 0 for block in blocks.blocks[0])


def test_scheduler_builds_metadata(scheduler):
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        finished_req_ids=set(),
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert isinstance(metadata, NanoLMCacheConnectorMetadata)
    restored = pickle.loads(pickle.dumps(metadata))
    assert isinstance(restored, NanoLMCacheConnectorMetadata)
    assert vars(restored) == vars(metadata)


def test_metadata_is_serializable():
    metadata = NanoLMCacheConnectorMetadata()

    restored = pickle.loads(pickle.dumps(metadata))

    assert isinstance(restored, NanoLMCacheConnectorMetadata)
    assert vars(restored) == vars(metadata)


def test_worker_binds_and_clears_metadata(make_connector):
    connector = make_connector("WORKER")
    assert not connector.has_connector_metadata()

    for _ in range(2):
        metadata = NanoLMCacheConnectorMetadata()
        connector.bind_connector_metadata(metadata)
        assert connector.has_connector_metadata()
        assert connector._get_connector_metadata() is metadata

        connector.clear_connector_metadata()
        assert not connector.has_connector_metadata()


def test_connector_instances_do_not_share_metadata(make_connector):
    scheduler = make_connector("SCHEDULER")
    worker = make_connector("WORKER")
    metadata = NanoLMCacheConnectorMetadata()

    worker.bind_connector_metadata(metadata)

    assert not scheduler.has_connector_metadata()
    assert worker._get_connector_metadata() is metadata
    worker.clear_connector_metadata()


def test_worker_start_load_is_noop(worker):
    kv_layer = torch.arange(64, dtype=torch.float32, device="cpu").reshape(2, 1, 4, 8)
    before = kv_layer.clone()
    forward_context = SimpleNamespace(
        attn_metadata=None,
        no_compile_layers={
            "model.layers.0.self_attn": SimpleNamespace(kv_cache=kv_layer),
        },
    )

    assert worker.start_load_kv(forward_context) is None
    assert torch.equal(kv_layer, before)


def test_worker_wait_for_layer_is_noop(worker):
    assert worker.wait_for_layer_load("model.layers.0.self_attn") is None


def test_worker_save_preserves_kv(worker):
    kv_layer = torch.arange(64, dtype=torch.float32, device="cpu").reshape(2, 1, 4, 8)
    before = kv_layer.clone()
    attn_metadata = SimpleNamespace(num_actual_tokens=4)

    assert (
        worker.save_kv_layer("model.layers.0.self_attn", kv_layer, attn_metadata)
        is None
    )
    assert torch.equal(kv_layer, before)


def test_worker_wait_for_save_is_noop(worker):
    assert worker.wait_for_save() is None


def test_request_finished_does_not_retain_blocks(scheduler, request_stub):
    block_ids = [7, 2]

    assert scheduler.request_finished(request_stub, block_ids) == (False, None)
    assert block_ids == [7, 2]


@pytest.mark.parametrize("finished_req_ids", [set(), {"req-1"}])
def test_worker_reports_no_transfers_or_load_errors(worker, finished_req_ids):
    original_ids = set(finished_req_ids)

    assert worker.get_finished(finished_req_ids) == (None, None)
    results = worker.get_transfer_results(finished_req_ids)

    assert results.finished_sending == set()
    assert results.finished_recving == set()
    assert results.failed_recving == set()
    assert worker.get_block_ids_with_load_errors() == set()
    assert finished_req_ids == original_ids


def test_transfer_results_preserve_base_aggregation(worker, monkeypatch):
    received_ids = []

    def finished(ids):
        received_ids.append(ids)
        return {"sent"}, {"received"}

    monkeypatch.setattr(worker, "get_finished", finished)
    ids = {"sent"}
    result = worker.get_transfer_results(ids)
    assert received_ids == [ids]
    assert result.finished_sending == {"sent"}
    assert result.finished_recving == {"received"}
    assert result.failed_recving == set()


def test_empty_step_drops_previous_request_ids(scheduler):
    scheduler.build_connector_meta(SimpleNamespace(num_scheduled_tokens={"old": 4}))
    metadata = scheduler.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))
    assert metadata.step_id == 2
    assert metadata.request_ids == ()
