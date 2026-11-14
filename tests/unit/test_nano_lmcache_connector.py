"""Call the vLLM connector directly, without an LLM, model weights or GPU.

Run with:
    .venv/bin/python -m pytest -q tests/unit/test_nano_lmcache_connector.py

Tests exercise scheduler lookup, worker saves and lifecycle tracing against
real vLLM. Scheduler-only lookup payloads are placeholders, not usable KV.
"""

# Optional dependencies must be checked before importing the connector.
# ruff: noqa: E402

import pickle
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("vllm.config")


from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import (
    CachedRequestData,
    KVConnectorBlockState,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

from nano_lmcache.integrations.vllm.nano_lmcache_connector import (
    NanoLMCacheConnector,
    NanoLMCacheConnectorMetadata,
    ReqMeta,
    SaveSpec,
    _cache_namespace,
)


def _test_model_config():
    return SimpleNamespace(
        model="test/tiny-qwen3",
        revision="weights-v1",
        dtype=torch.bfloat16,
        quantization=None,
        hf_config=SimpleNamespace(to_dict=lambda: {"num_hidden_layers": 3}),
    )


@pytest.mark.parametrize("changed", ["model", "revision", "dtype", "kv_dtype", "rope"])
def test_namespace_isolates_incompatible_kv(changed):
    from nano_lmcache.core.prefix_cache import ChunkedPrefixCache

    config = SimpleNamespace(
        model_config=_test_model_config(),
        cache_config=SimpleNamespace(block_size=4, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    namespace = _cache_namespace(config)
    first = ChunkedPrefixCache(chunk_size=4, namespace=namespace)
    first.store([1, 2, 3, 4], [b"kv"])
    compatible = ChunkedPrefixCache(
        chunk_size=4, namespace=_cache_namespace(config), storage=first.storage
    )
    assert compatible.lookup([1, 2, 3, 4]).matched_tokens == 4
    if changed == "model":
        config.model_config.model = "test/other-model"
    elif changed == "revision":
        config.model_config.revision = "weights-v2"
    elif changed == "dtype":
        config.model_config.dtype = torch.float32
    elif changed == "kv_dtype":
        config.cache_config.cache_dtype = "fp8"
    else:
        config.model_config.hf_config.to_dict = lambda: {
            "num_hidden_layers": 3,
            "rope_theta": 500000,
        }
    incompatible = ChunkedPrefixCache(
        chunk_size=4, namespace=_cache_namespace(config), storage=first.storage
    )
    assert incompatible.lookup([1, 2, 3, 4]).matched_tokens == 0


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
            ),
            cache_config=SimpleNamespace(block_size=4, cache_dtype="auto"),
            model_config=_test_model_config(),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1
            ),
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
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NanoLMCacheConnector",
            kv_connector_module_path=(
                "nano_lmcache.integrations.vllm.nano_lmcache_connector"
            ),
            kv_role="kv_both",
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        cache_config=SimpleNamespace(block_size=4, cache_dtype="auto"),
        model_config=_test_model_config(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    cache_config = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])

    connector = KVConnectorFactory.create_connector(config, role, cache_config)

    assert isinstance(connector, NanoLMCacheConnector)
    assert connector.role == role
    assert not connector.has_connector_metadata()


@pytest.mark.parametrize("num_computed_tokens", [0, 4, 8])
def test_scheduler_reports_zero_matches_with_empty_cache(
    scheduler, request_stub, num_computed_tokens
):
    request_stub.num_computed_tokens = num_computed_tokens
    original_tokens = list(request_stub.prompt_token_ids)

    for _ in range(2):
        assert scheduler.get_num_new_matched_tokens(
            request_stub, num_computed_tokens
        ) == (0, False)
    assert request_stub.prompt_token_ids == original_tokens
    assert request_stub.num_computed_tokens == num_computed_tokens


def test_scheduler_reports_matches_only_after_cache_is_populated(
    scheduler, request_stub
):
    request_stub.prompt_token_ids = list(range(13))
    cache = scheduler._lookup_cache

    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (0, False)
    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (0, False)

    cache.store(request_stub.prompt_token_ids, [b"kv-0", b"kv-1", b"kv-2"])
    # A different request can reuse the same prefix; IDs are not cache keys.
    request_stub.request_id = "req-2"
    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (12, False)
    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (12, False)


@pytest.mark.parametrize(
    ("cached_tokens", "local_tokens", "expected"),
    [
        (12, 0, 12),
        (12, 4, 8),
        (12, 8, 4),
        (12, 12, 0),
        (8, 4, 4),
        (8, 8, 0),
        (4, 8, 0),
    ],
)
def test_scheduler_reports_only_additional_matches(
    scheduler, request_stub, cached_tokens, local_tokens, expected
):
    tokens = list(range(13))
    request_stub.prompt_token_ids = tokens.copy()
    request_stub.num_computed_tokens = 0
    scheduler._lookup_cache.store(
        tokens[:cached_tokens], [b"kv"] * (cached_tokens // 4)
    )

    assert scheduler.get_num_new_matched_tokens(request_stub, local_tokens) == (
        expected,
        False,
    )
    assert request_stub.prompt_token_ids == tokens
    assert request_stub.num_computed_tokens == 0


@pytest.mark.parametrize(
    ("prompt_length", "expected"),
    [(1, 0), (3, 0), (4, 0), (5, 4), (7, 4), (8, 4), (12, 8), (13, 12)],
)
def test_scheduler_reuses_complete_chunks_and_leaves_token_for_logits(
    scheduler, request_stub, prompt_length, expected
):
    scheduler._lookup_cache.store(list(range(16)), [b"kv"] * 4)
    request_stub.prompt_token_ids = list(range(prompt_length))

    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (expected, False)


@pytest.mark.parametrize("token_ids", [None, []], ids=["missing", "empty"])
def test_scheduler_reports_zero_without_prompt_tokens(
    scheduler, request_stub, token_ids
):
    scheduler._lookup_cache.store(list(range(12)), [b"kv"] * 3)
    request_stub.prompt_token_ids = token_ids

    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (0, False)
    assert request_stub.prompt_token_ids is token_ids


@pytest.mark.parametrize(
    ("changed_index", "expected"), [(0, 0), (4, 4), (8, 8), (12, 12)]
)
def test_scheduler_matches_only_contiguous_shared_prefix(
    scheduler, request_stub, changed_index, expected
):
    tokens = list(range(13))
    scheduler._lookup_cache.store(tokens, [b"kv"] * 3)
    request_stub.prompt_token_ids = tokens.copy()
    request_stub.prompt_token_ids[changed_index] = 99

    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (expected, False)


def test_scheduler_rechecks_cache_after_invalidation(scheduler, request_stub):
    request_stub.prompt_token_ids = list(range(13))
    cache = scheduler._lookup_cache
    cache.store(request_stub.prompt_token_ids, [b"kv"] * 3)
    keys = cache.lookup(request_stub.prompt_token_ids).keys
    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (12, False)

    cache.storage.invalidate(keys[1])
    assert scheduler.get_num_new_matched_tokens(request_stub, 0) == (4, False)
    assert scheduler.get_num_new_matched_tokens(request_stub, 4) == (0, False)


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
    scheduler_output = SchedulerOutput.make_empty()

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


@pytest.fixture
def save_layers(worker):
    layers = {
        f"model.layers.{index}.self_attn": (
            torch.arange(10 * 2 * 4 * 6, dtype=torch.float32).reshape(10, 2, 4, 6)
            + index * 10000
        )
        for index in (10, 0, 2)
    }
    worker.register_kv_caches(layers)
    return layers


@pytest.mark.parametrize("prompt_length", [3, 8, 10])
def test_worker_stores_complete_cpu_chunks(worker, save_layers, prompt_length):
    slots = tuple(range(28, 32)) + tuple(range(8, 12)) + tuple(range(36, 40))
    tokens = tuple(range(prompt_length))
    worker.bind_connector_metadata(
        NanoLMCacheConnectorMetadata(
            requests=(ReqMeta("req", tokens, slots[:prompt_length]),)
        )
    )
    originals = {name: value.clone() for name, value in save_layers.items()}
    for index, (name, value) in enumerate(save_layers.items()):
        worker.save_kv_layer(name, value, SimpleNamespace())
        if index < len(save_layers) - 1:
            assert worker._save_cache.lookup(tokens).matched_tokens == 0
        assert torch.equal(value, originals[name])
        value.fill_(-1)
    worker.wait_for_save()
    result = worker._save_cache.lookup(tokens)
    assert result.matched_tokens == prompt_length // 4 * 4
    for chunk_index, chunk in enumerate(result.values):
        assert chunk.shape == (3, 2, 2, 4, 3)
        assert chunk.device.type == "cpu"
        assert not chunk.requires_grad
        for layer_index, model_index in enumerate((0, 2, 10)):
            original = originals[f"model.layers.{model_index}.self_attn"]
            for offset in range(4):
                token_index = chunk_index * 4 + offset
                block = (7, 2, 9)[token_index // 4]
                expected = original[block, :, token_index % 4]
                assert torch.equal(chunk[layer_index, 0, :, offset], expected[:, :3])
                assert torch.equal(chunk[layer_index, 1, :, offset], expected[:, 3:])


def test_worker_does_not_combine_layers_from_different_steps(worker, save_layers):
    tokens = tuple(range(4))
    metadata = NanoLMCacheConnectorMetadata(
        requests=(ReqMeta("req", tokens, (28, 29, 30, 31)),)
    )
    worker.bind_connector_metadata(metadata)
    names = list(save_layers)
    worker.save_kv_layer(names[0], save_layers[names[0]], SimpleNamespace())
    with pytest.raises(RuntimeError, match="missing layers"):
        worker.wait_for_save()
    worker.clear_connector_metadata()
    worker.bind_connector_metadata(metadata)
    for name in names[1:]:
        worker.save_kv_layer(name, save_layers[name], SimpleNamespace())
    assert worker._save_cache.lookup(tokens).matched_tokens == 0
    with pytest.raises(RuntimeError, match="missing layers"):
        worker.wait_for_save()


def test_worker_keeps_requests_separate(worker, save_layers):
    tokens_a, tokens_b = (0, 1, 2, 3), (10, 11, 12, 13)
    worker.bind_connector_metadata(
        NanoLMCacheConnectorMetadata(
            requests=(
                ReqMeta("a", tokens_a, (28, 29, 30, 31)),
                ReqMeta("b", tokens_b, (8, 9, 10, 11)),
            )
        )
    )
    for name, value in save_layers.items():
        worker.save_kv_layer(name, value, SimpleNamespace())
    worker.wait_for_save()
    chunk_a = worker._save_cache.lookup(tokens_a).values[0]
    chunk_b = worker._save_cache.lookup(tokens_b).values[0]
    original = save_layers["model.layers.0.self_attn"]
    assert torch.equal(chunk_a[0, 0], original[7, :, :, :3])
    assert torch.equal(chunk_b[0, 0], original[2, :, :, :3])


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
    scheduler.build_connector_meta(
        _save_step(
            new=[_new_save_request("old", list(range(4)), [7])], counts={"old": 4}
        )
    )
    metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
    assert metadata.step_id == 2
    assert metadata.request_ids == ()
    assert metadata.requests == []


def _new_save_request(request_id, tokens, blocks, computed=0):
    return NewRequestData(
        req_id=request_id,
        prompt_token_ids=tokens,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=(blocks,),
        num_computed_tokens=computed,
        lora_request=None,
    )


def _save_step(*, new=(), cached=(), counts=None, resumed=(), finished=()):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = list(new)
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=[req_id for req_id, _, _ in cached],
        resumed_req_ids=set(resumed),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[blocks for _, blocks, _ in cached],
        num_computed_tokens=[computed for _, _, computed in cached],
        num_output_tokens=[0 for _ in cached],
    )
    output.num_scheduled_tokens = counts or {}
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    output.finished_req_ids = set(finished)
    return output


@pytest.mark.parametrize(
    ("tokens", "computed", "scheduled", "expected"),
    [
        (list(range(10)), 0, 6, 4),
        (list(range(10)), 4, 6, 8),  
        (list(range(8)), 0, 8, 8), 
        (list(range(10)), 8, 5, 8),  
        (list(range(10)), 10, 1, 0),  
        (list(range(10)), 0, 0, 0),
        (list(range(3)), 0, 3, 0),
        ([], 0, 1, 0),
        (None, 0, 1, 0),
    ],
)
def test_scheduler_plans_only_complete_computed_prompt_chunks(
    scheduler, tokens, computed, scheduled, expected
):
    output = _save_step(
        new=[_new_save_request("req", tokens, [7, 2, 9], computed)],
        counts={"req": scheduled},
    )
    original = pickle.dumps(output)
    metadata = scheduler.build_connector_meta(output)
    assert pickle.dumps(output) == original
    assert pickle.loads(pickle.dumps(metadata)) == metadata
    if expected == 0:
        assert metadata.requests == []
    else:
        assert metadata.requests == [
            ReqMeta(
                "req",
                tuple(tokens[:expected]),
                (28, 29, 30, 31, 8, 9, 10, 11)[:expected],
            ),
        ]


def test_scheduler_plans_only_newly_completed_chunks(scheduler):
    first = _save_step(
        new=[_new_save_request("req", list(range(8)), [7])], counts={"req": 4}
    )
    assert scheduler.build_connector_meta(first).requests == [
        ReqMeta("req", (0, 1, 2, 3), (28, 29, 30, 31))
    ]

    second = _save_step(cached=[("req", ([2],), 4)], counts={"req": 4})
    assert scheduler.build_connector_meta(second).requests == [
        ReqMeta(
            "req",
            tuple(range(8)),
            (28, 29, 30, 31, 8, 9, 10, 11),
            SaveSpec(skip_leading_tokens=4),
        )
    ]

    third = _save_step(cached=[("req", None, 8)], counts={"req": 1})
    assert scheduler.build_connector_meta(third).requests == []


def test_worker_saves_only_the_new_token_span(worker, save_layers):
    originals = {name: value.clone() for name, value in save_layers.items()}

    worker.bind_connector_metadata(
        NanoLMCacheConnectorMetadata(
            requests=(ReqMeta("req", (0, 1, 2, 3), (28, 29, 30, 31)),)
        )
    )
    for name, value in save_layers.items():
        worker.save_kv_layer(name, value, SimpleNamespace())
    worker.wait_for_save()
    worker.clear_connector_metadata()
    assert worker._save_cache.lookup((0, 1, 2, 3)).matched_tokens == 4

    worker.bind_connector_metadata(
        NanoLMCacheConnectorMetadata(
            requests=(
                ReqMeta(
                    "req",
                    tuple(range(8)),
                    (28, 29, 30, 31, 8, 9, 10, 11),
                    SaveSpec(skip_leading_tokens=4),
                ),
            )
        )
    )
    for name, value in save_layers.items():
        value.fill_(float("nan"))
        for position in range(4, 8):
            block, offset = (7, 2)[position // 4], position % 4
            value[block, :, offset] = originals[name][block, :, offset]
        worker.save_kv_layer(name, value, SimpleNamespace())
        value.fill_(-1)
    worker.wait_for_save()

    result = worker._save_cache.lookup(tuple(range(8)))
    assert result.matched_tokens == 8
    original = originals["model.layers.0.self_attn"]
    assert torch.equal(result.values[1][0, 0], original[2, :, :, :3])


def test_connector_builds_role_specific_caches(make_connector):
    scheduler = make_connector("SCHEDULER")
    worker = make_connector("WORKER")

    assert scheduler._lookup_cache is not None
    assert scheduler._save_cache is None
    assert worker._save_cache is not None
    assert worker._lookup_cache is None


def test_scheduler_plans_reach_worker_across_prefill_steps(
    scheduler, worker, save_layers
):
    tokens = list(range(10))
    originals = {name: value.clone() for name, value in save_layers.items()}
    computed_end = 0
    steps = [
        (_save_step(new=[_new_save_request("req", tokens, [7])], counts={"req": 3}), 0),
        (_save_step(cached=[("req", ([2],), 3)], counts={"req": 3}), 4),
        (_save_step(cached=[("req", ([9],), 6)], counts={"req": 2}), 8),
        (_save_step(cached=[("req", None, 8)], counts={"req": 2}), 8),
    ]
    for output, expected_tokens in steps:
        computed_end += output.num_scheduled_tokens["req"]
        original = pickle.dumps(output)
        metadata = scheduler.build_connector_meta(output)
        assert pickle.dumps(output) == original
        worker.bind_connector_metadata(pickle.loads(pickle.dumps(metadata)))
        for name, value in save_layers.items():
            value.fill_(float("nan"))
            for position in range(computed_end):
                block, offset = (7, 2, 9)[position // 4], position % 4
                value[block, :, offset] = originals[name][block, :, offset]
            worker.save_kv_layer(name, value, SimpleNamespace())
            value.fill_(-1) 
        worker.wait_for_save()
        worker.clear_connector_metadata()
        result = worker._save_cache.lookup(tokens)
        assert result.matched_tokens == expected_tokens
        for chunk_index, chunk in enumerate(result.values):
            for layer_index, model_index in enumerate((0, 2, 10)):
                source = originals[f"model.layers.{model_index}.self_attn"]
                block = (7, 2)[chunk_index]
                assert torch.equal(chunk[layer_index, 0], source[block, :, :, :3])
                assert torch.equal(chunk[layer_index, 1], source[block, :, :, 3:])

    decode = _save_step(cached=[("req", None, 10)], counts={"req": 1})
    assert scheduler.build_connector_meta(decode).requests == []


def test_scheduler_replaces_blocks_on_resume_and_keeps_requests_separate(scheduler):
    scheduler.build_connector_meta(
        _save_step(
            new=[
                _new_save_request("a", list(range(10)), [7, 2]),
                _new_save_request("b", list(range(20, 30)), [3, 4]),
            ],
            counts={"a": 6, "b": 6},
        )
    )
    preempted = SchedulerOutput.make_empty()
    preempted.preempted_req_ids = {"a"}
    assert scheduler.build_connector_meta(preempted).requests == []
    metadata = scheduler.build_connector_meta(
        _save_step(
            cached=[("a", ([9, 1],), 0), ("b", None, 6)],
            counts={"a": 8, "b": 2},
            resumed={"a"},
        )
    )
    assert metadata.requests == [
        ReqMeta(
            "a",
            tuple(range(8)),
            (36, 37, 38, 39, 4, 5, 6, 7),
            SaveSpec(skip_leading_tokens=4),
        ),
        ReqMeta(
            "b",
            tuple(range(20, 28)),
            (12, 13, 14, 15, 16, 17, 18, 19),
            SaveSpec(skip_leading_tokens=4),
        ),
    ]


def test_scheduler_uses_current_blocks_after_copy_on_write(scheduler):
    output = _save_step(
        new=[_new_save_request("req", list(range(8)), [7, 2])], counts={"req": 8}
    )
    output.kv_connector_block_state = KVConnectorBlockState(
        req_ids={"req"},
        resolve_block_ids=lambda _: ([3, 9],),
        boundary_state_offloads={},
    )
    metadata = scheduler.build_connector_meta(output)
    assert metadata.requests[0].slot_mapping == (12, 13, 14, 15, 36, 37, 38, 39)


@pytest.mark.parametrize("via_callback", [False, True])
def test_scheduler_releases_finished_save_state(scheduler, via_callback):
    scheduler.build_connector_meta(
        _save_step(
            new=[_new_save_request("req", list(range(8)), [7, 2])], counts={"req": 6}
        )
    )
    if via_callback:
        scheduler.request_finished(SimpleNamespace(request_id="req"), [7, 2])
    else:
        scheduler.build_connector_meta(_save_step(finished={"req"}))
    assert "req" not in scheduler._request_trackers
    metadata = scheduler.build_connector_meta(
        _save_step(
            new=[_new_save_request("req", list(range(20, 24)), [9])], counts={"req": 4}
        )
    )
    assert metadata.requests == [ReqMeta("req", (20, 21, 22, 23), (36, 37, 38, 39))]
