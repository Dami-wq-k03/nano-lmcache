import pytest

from nano_lmcache.experimental import (
    LoadDecision,
    PrefetchState,
    PrefetchStateMachine,
    RecomputeLoadCostModel,
)
from nano_lmcache.experimental.dataplane import KVLayout


def test_layout_round_trips_and_has_stable_fingerprint():
    layout = KVLayout(
        model_namespace="qwen/rev",
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        page_tokens=16,
        dtype="float16",
        model_revision="commit-1",
        tokenizer_fingerprint="tokenizer-1",
        key_layout="HND",
        value_layout="NHD",
        layer_layout="page-major",
    )

    restored = KVLayout.from_dict(layout.to_dict())

    assert restored == layout
    assert restored.fingerprint == layout.fingerprint
    assert restored.tensor_shapes == ((4, 16, 8), (16, 4, 8))


def test_cost_model_switches_at_the_measured_break_even():
    model = RecomputeLoadCostModel(
        recompute_ms_per_token=0.1,
        bandwidth_gbps={"pinned_cpu": 10.0, "nvme": 1.0},
        fixed_latency_ms={"pinned_cpu": 0.1, "nvme": 2.0},
    )

    fast = model.estimate(tier="pinned_cpu", token_count=256, nbytes=1_000_000)
    slow = model.estimate(tier="nvme", token_count=16, nbytes=100_000_000)

    assert fast.decision is LoadDecision.LOAD
    assert fast.savings_ms > 0
    assert slow.decision is LoadDecision.RECOMPUTE


def test_prefetch_state_machine_deduplicates_and_cleans_up():
    machine = PrefetchStateMachine()
    planned = machine.plan("page-0", tier="pinned_cpu")
    assert machine.plan("page-0", tier="nvme") is planned
    assert machine.active == (planned,)

    machine.start("page-0")
    ready = machine.ready("page-0")
    assert ready.state is PrefetchState.READY
    assert machine.active == ()

    machine.plan("page-1", tier="nvme")
    machine.start("page-1")
    failed = machine.fail("page-1", "short read")
    assert failed.error == "short read"
    with pytest.raises(ValueError, match="cannot transition"):
        machine.ready("page-1")
