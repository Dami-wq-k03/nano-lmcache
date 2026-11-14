import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from nano_lmcache.experimental.policy import (
    CacheValue,
    CostAwareEvictionPolicy,
    PrefetchDepthTuner,
    admission_score,
    reuse_distances,
)
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.eviction import PrefixIndex
from nano_lmcache.storage.memory import MemoryKVCache


def test_admission_score_is_expected_savings_per_byte():
    value = CacheValue(100, 10.0, 2.0)
    assert admission_score(value, reuse_probability=0.5, write_ms=1) == 0.03
    assert admission_score(value, reuse_probability=0.5, write_ms=4) == 0
    assert admission_score(value, reuse_probability=0, write_ms=1) == -0.01
    assert admission_score(CacheValue(100, 1, 2)) < 0


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 0),
        (-1, 1, 0),
        (True, 1, 0),
        (1.0, 1, 0),
        (1, -1, 0),
        (1, 1, -1),
        (1, float("nan"), 0),
        (1, 1, float("inf")),
    ],
)
def test_invalid_value_metadata(values):
    with pytest.raises(ValueError):
        CacheValue(*values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reuse_probability": -0.1},
        {"reuse_probability": 1.1},
        {"reuse_probability": float("nan")},
        {"reuse_probability": float("inf")},
        {"write_ms": -1},
        {"write_ms": float("inf")},
    ],
)
def test_invalid_admission_inputs(kwargs):
    with pytest.raises(ValueError):
        admission_score(CacheValue(1, 1, 0), **kwargs)


def test_cost_policy_retains_expensive_old_leaf_unlike_lru():
    values = {"a": CacheValue(100, 10, 0), "b": CacheValue(100, 1, 0)}
    cache = MemoryKVCache(
        max_entries=2, eviction_policy=lambda: CostAwareEvictionPolicy(values.get)
    )
    control = MemoryKVCache(max_entries=2)
    for backend in (cache, control):
        backend.put("a", "a")
        backend.put("b", "b")
        backend.put("c", "c")
    assert cache.eviction_policy == "cost-aware"
    assert set(cache.keys()) == {"a", "c"}
    assert set(control.keys()) == {"b", "c"}


def test_cost_density_frequency_and_dynamic_callback():
    values = {"a": CacheValue(100, 10, 0), "b": CacheValue(10, 2, 0)}
    policy = CostAwareEvictionPolicy(values.get)
    policy.on_insert("a", is_leaf=True)
    policy.on_insert("b", is_leaf=True)
    assert policy.choose_victim(None) == "a"  # 0.1/byte versus 0.2/byte
    policy.on_touch("a", is_leaf=True)
    policy.on_touch("a", is_leaf=True)
    assert policy.choose_victim(None) == "b"  # frequency 3 * 0.1 > 0.2
    values["b"] = CacheValue(10, 100, 0)
    assert policy.choose_victim(None) == "a"  # fresh metadata, no stale ranking
    policy.on_write("a", is_leaf=True)
    assert policy.frequency_of("a") == 3  # overwrite is not another hit


def test_unknown_and_nonbeneficial_values_use_lru_ties():
    values = {"slow": CacheValue(1, 1, 2)}
    policy = CostAwareEvictionPolicy(values.get)
    policy.on_insert("unknown", is_leaf=True)
    policy.on_insert("slow", is_leaf=True)
    assert policy.choose_victim(None) == "unknown"
    policy.on_touch("unknown", is_leaf=True)
    assert policy.choose_victim(None) == "slow"
    policy.on_remove("slow")
    assert policy.choose_victim(None) == "unknown"
    with pytest.raises(KeyError):
        policy.frequency_of("slow")


def test_leaf_eligibility_protection_and_exposure_do_not_duplicate_tree():
    policy = CostAwareEvictionPolicy(lambda key: None)
    policy.on_insert("internal", is_leaf=False)
    policy.on_insert("leaf", is_leaf=True)
    assert policy.choose_victim(None) == "leaf"
    assert policy.choose_victim("leaf") is None
    policy.on_touch("internal", is_leaf=False)
    policy.on_leaf_status_change("internal", is_leaf=True)
    assert policy.frequency_of("internal") == 2
    assert policy.choose_victim("leaf") == "internal"
    policy.on_leaf_status_change("internal", is_leaf=False)
    assert policy.choose_victim("leaf") is None


def test_protected_chain_rejection_and_internal_frequency_survive():
    values = {key: CacheValue(1, 1, 0) for key in "abcx"}
    index = PrefixIndex(
        max_entries=3, eviction_policy=lambda: CostAwareEvictionPolicy(values.get)
    )
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("c", "b")
    index.touch("b")
    assert not index.admit("d", "c").admitted
    assert tuple(index.keys_by_recency()) == ("a", "c", "b")
    assert index.admit("x", None).evicted == ("c",)
    assert index.frequency_of("b") == 2
    assert index.parent_of("b") == "a"


def test_restore_access_has_no_fake_frequency_and_deterministic_key_ties():
    policy = CostAwareEvictionPolicy(lambda key: CacheValue(1, 1, 0))
    policy.on_restore("b", is_leaf=True)
    policy.on_restore("a", is_leaf=True)
    policy.on_restore_access("a", 10)
    policy.on_restore_access("b", 10)
    assert policy.frequency_of("a") == policy.frequency_of("b") == 1
    assert policy.choose_victim(None) == "a"
    assert policy.choose_victim("a") == "b"


def test_disk_policy_recovery_reuses_existing_mechanics(tmp_path):
    values = {key: CacheValue(1, 1, 0) for key in "abc"}

    def factory():
        return CostAwareEvictionPolicy(values.get)

    first = DiskKVCache(tmp_path, max_entries=2, eviction_policy=factory)
    first.put("a", "a")
    first.put("b", "b")
    first.get("a")
    first.get("a")
    reopened = DiskKVCache(tmp_path, max_entries=2, eviction_policy=factory)
    assert reopened.frequency_of("a") == 1
    reopened.put("c", "c")
    assert set(reopened.keys()) == {"a", "c"}


def test_delegate_heap_remains_bounded_and_policy_instances_are_independent():
    policy = CostAwareEvictionPolicy(lambda key: CacheValue(1, 1, 0))
    first = MemoryKVCache(max_entries=2, eviction_policy=policy)
    second = MemoryKVCache(max_entries=2, eviction_policy=policy)
    for backend in (first, second):
        backend.put("a", "a")
        backend.put("b", "b")
    for _ in range(1000):
        first.get("a")
    assert first.frequency_of("a") == 1001
    assert second.frequency_of("a") == 1
    assert first._index.policy.heap_size <= 64
    first.put("c", "c")
    assert set(first.keys()) == {"a", "c"}
    assert set(second.keys()) == {"a", "b"}


def test_random_operations_keep_prefix_and_byte_invariants():
    parents = {"a": None, "b": "a", "c": "b", "x": None, "y": "x", "z": "y"}
    values = {key: CacheValue(i + 1, 10 - i, 0.1) for i, key in enumerate(parents)}
    cache = MemoryKVCache(
        max_entries=5,
        max_bytes=15,
        size_fn=lambda value: value.nbytes,
        eviction_policy=lambda: CostAwareEvictionPolicy(values.get),
    )
    rng = random.Random(17)
    for _ in range(500):
        key = rng.choice(list(parents))
        operation = rng.randrange(3)
        if operation == 0:
            cache.put(key, values[key], parent_key=parents[key])
        elif operation == 1:
            cache.get(key)
        else:
            cache.invalidate(key)
        keys = set(cache.keys())
        assert len(cache) <= 5
        assert cache.nbytes == sum(values[k].nbytes for k in keys) <= 15
        assert keys == set(cache._values)
        for stored in keys:
            parent = cache.parent_of(stored)
            while parent is not None:
                assert parent in keys
                parent = cache.parent_of(parent)


def test_exact_reuse_distance_is_distinct_keys_not_elapsed_accesses():
    assert reuse_distances("abbcaba") == [None, None, 0, None, 2, 2, 1]
    assert reuse_distances(["a"] + ["b"] * 20 + ["a"])[-1] == 1
    assert reuse_distances(iter([])) == []
    assert reuse_distances([None, None, (1, 2), (1, 2)]) == [None, 0, None, 0]


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_reuse_distances_match_independent_interval_reference(seed):
    rng = random.Random(seed)
    trace = [rng.randrange(8) for _ in range(200)]
    last = {}
    expected = []
    for index, key in enumerate(trace):
        expected.append(len(set(trace[last[key] + 1 : index])) if key in last else None)
        last[key] = index
    assert reuse_distances(iter(trace)) == expected
    for capacity in range(1, 9):
        control = MemoryKVCache(max_entries=capacity)
        misses = 0
        for key in trace:
            if control.get(str(key)) is None:
                misses += 1
                control.put(str(key), key)
        assert misses == sum(d is None or d >= capacity for d in expected)


def test_prefetch_tuner_reports_wasted_bytes_and_uses_window_feedback():
    tuner = PrefetchDepthTuner(initial_depth=4, max_depth=8)
    assert tuner.observe(prefetched_bytes=100, used_bytes=25, stall_ms=1) == 2
    assert tuner.observe(prefetched_bytes=400, used_bytes=350, stall_ms=3) == 3
    assert tuner.prefetched_bytes == 500
    assert tuner.used_bytes == 375
    assert tuner.wasted_bytes == 125
    assert tuner.waste_ratio == 0.25
    # High cumulative waste must not prevent growth after a useful new window.
    assert tuner.observe(prefetched_bytes=100, used_bytes=100, stall_ms=1) == 4


def test_prefetch_tuner_can_disable_probe_and_stays_bounded():
    tuner = PrefetchDepthTuner(initial_depth=3, max_depth=3)
    assert tuner.observe(prefetched_bytes=100, used_bytes=0) == 1
    assert tuner.observe(prefetched_bytes=100, used_bytes=0) == 0
    assert tuner.observe(prefetched_bytes=0, used_bytes=0) == 0
    assert tuner.observe(prefetched_bytes=0, used_bytes=0, stall_ms=1) == 1
    for _ in range(10):
        tuner.observe(prefetched_bytes=100, used_bytes=100, stall_ms=1)
    assert tuner.depth == 3
    assert tuner.wasted_bytes == 200


def test_tuner_no_stall_no_growth_and_tolerance_boundary():
    tuner = PrefetchDepthTuner(initial_depth=2, waste_tolerance=0.25)
    assert tuner.waste_ratio == 0
    assert tuner.observe(prefetched_bytes=100, used_bytes=75) == 2
    assert tuner.observe(prefetched_bytes=100, used_bytes=74) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_depth": -1},
        {"initial_depth": 9},
        {"initial_depth": True},
        {"max_depth": 0},
        {"max_depth": 2.0},
        {"waste_tolerance": -0.1},
        {"waste_tolerance": 1.1},
        {"waste_tolerance": float("nan")},
    ],
)
def test_invalid_tuner_configuration(kwargs):
    with pytest.raises(ValueError):
        PrefetchDepthTuner(**kwargs)


@pytest.mark.parametrize(
    "prefetched,used,stall",
    [
        (-1, 0, 0),
        (1, 2, 0),
        (1, -1, 0),
        (1.0, 1, 0),
        (1, True, 0),
        (1, 1, -1),
        (1, 1, float("inf")),
        (1, 1, float("nan")),
    ],
)
def test_invalid_tuner_feedback_does_not_change_state(prefetched, used, stall):
    tuner = PrefetchDepthTuner()
    with pytest.raises(ValueError):
        tuner.observe(prefetched_bytes=prefetched, used_bytes=used, stall_ms=stall)
    assert tuner.prefetched_bytes == tuner.used_bytes == tuner.wasted_bytes == 0
    assert tuner.depth == 1


@pytest.fixture
def bench(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_cache_policy.py"
    spec = importlib.util.spec_from_file_location("research_policy_bench", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclass lookup
    spec.loader.exec_module(module)
    return module


def test_benchmark_is_deterministic_and_reports_lru_controls(bench):
    result = bench.benchmark(requests=100)
    assert result == bench.benchmark(requests=100)
    assert set(result["traces"]) == {"uniform", "shared_system_prompt", "low_reuse"}
    assert "hardware" not in result
    for trace in result["traces"].values():
        assert set(trace["policies"]) == {"lru", "cost-aware", "cost-aware+admission"}
        for name, metrics in trace["policies"].items():
            assert 0 <= metrics["hit_bytes"] <= metrics["requested_bytes"]
            assert metrics["estimated_latency_ms"] > 0
            if name == "lru":
                assert metrics["estimated_latency_vs_lru"] == 1
        reuse = trace["reuse_distance"]
        assert (
            reuse["cold_accesses"] + sum(reuse["histogram"].values())
            == (trace["policies"]["lru"]["accesses"])
        )


def test_model_latency_and_hit_bytes_have_exact_accounting(bench):
    value = CacheValue(1024, 2, 0.1)
    trace = [bench.Access("a", value)] * 2
    control = bench.simulate(trace, mode="lru", capacity_bytes=1024, write_ms=0.15)
    assert control["hits"] == 1
    assert control["hit_bytes"] == 1024
    assert control["write_bytes"] == 1024
    assert control["estimated_latency_ms"] == pytest.approx(2 + 0.15 + 0.1)
    research = bench.simulate(
        trace, mode="cost-aware+admission", capacity_bytes=1024, write_ms=0.15
    )
    assert research["hits"] == 0  # not every workload improves
    assert research["estimated_latency_ms"] > control["estimated_latency_ms"]


def test_cost_policy_can_lose_to_lru_on_a_different_trace(bench):
    expensive = CacheValue(1024, 10, 0.1)
    cheap = CacheValue(1024, 1, 0.1)
    trace = [
        bench.Access("a", expensive),
        bench.Access("b", cheap),
        bench.Access("c", cheap),
        bench.Access("b", cheap),
    ]
    control = bench.simulate(trace, mode="lru", capacity_bytes=2048, write_ms=0.15)
    research = bench.simulate(
        trace, mode="cost-aware", capacity_bytes=2048, write_ms=0.15
    )
    assert control["hits"] == 1 and research["hits"] == 0
    assert research["estimated_latency_ms"] > control["estimated_latency_ms"]


def test_admission_helper_does_not_bypass_missing_parent(bench):
    value = CacheValue(1024, 2, 0.1)
    trace = [bench.Access("child", value, "missing")] * 3
    result = bench.simulate(
        trace, mode="cost-aware+admission", capacity_bytes=2048, write_ms=0.15
    )
    assert result["hits"] == 0
    assert result["write_bytes"] == 0
    assert result["capacity_or_parent_rejections"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests": 0},
        {"capacity_bytes": 0},
        {"write_ms": -1},
    ],
)
def test_invalid_benchmark_configuration(bench, kwargs):
    with pytest.raises(ValueError):
        bench.benchmark(**kwargs)


def test_benchmark_cli_emits_model_only_json():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_cache_policy.py"
    completed = subprocess.run(
        [sys.executable, str(path), "--requests", "30"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    assert output["requests"] == 30
    assert "synthetic estimates" in output["scope"]
    assert "environment" not in output
