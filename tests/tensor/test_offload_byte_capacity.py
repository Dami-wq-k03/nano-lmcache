import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.storage.stats import StorageStats  # noqa: E402
from nano_lmcache.storage.tiered import TieredStorage, TierStats  # noqa: E402
from nano_lmcache.tensor.offload import CPUOffloadKVCache  # noqa: E402
from nano_lmcache.tensor.serializers import TensorSerializer  # noqa: E402

pytestmark = pytest.mark.torch


def make_cache(**limits):
    return CPUOffloadKVCache(device="cpu", pin_memory=False, **limits)


def payload(size, dtype=torch.uint8):
    return torch.ones(size, dtype=dtype, device="cpu", pin_memory=False)


def assert_contents(cache, **sizes):
    assert list(cache.keys()) == list(sizes)
    assert len(cache) == len(sizes)
    assert cache.nbytes == sum(sizes.values())
    for key, size in sizes.items():
        assert torch.equal(cache.peek(key), payload(size))
        parent = cache.parent_of(key)
        assert parent is None or parent in cache


def test_multiple_evictions_follow_lru():
    cache = make_cache(max_bytes=6)
    for key in "abc":
        cache.put(key, payload(2))
    cache.get("a")
    assert list(cache.keys()) == ["b", "c", "a"]
    cache.put("d", payload(4))
    assert_contents(cache, a=2, d=4)
    assert cache.stats == StorageStats(evictions=2)


@pytest.mark.parametrize("key,size,parent", [
    ("new", 7, None), ("a", 7, None), ("b", 5, None), ("c", 3, "b"),
])
def test_rejected_write_preserves_payloads_and_recency(key, size, parent):
    cache = make_cache(max_bytes=6)
    cache.put("a", payload(2))
    cache.put("b", payload(2), parent_key="a")
    cache.put("other", payload(2))
    cache.put(key, payload(size), parent_key=parent)
    assert_contents(cache, a=2, b=2, other=2)
    assert cache.parent_of("b") == "a"
    assert cache.stats == StorageStats(rejected_no_room=1)


def test_overwrites_resize_payload_and_protect_parent():
    cache = make_cache(max_bytes=6)
    cache.put("a", payload(2))
    cache.put("b", payload(2), parent_key="a")
    cache.put("c", payload(2), parent_key="b")
    cache.put("b", payload(4), parent_key="missing")
    assert_contents(cache, a=2, b=4)
    assert cache.parent_of("b") == "a"
    cache.put("b", payload(1))
    assert_contents(cache, a=2, b=1)
    cache.put("c", payload(3), parent_key="b")
    assert_contents(cache, a=2, b=1, c=3)
    assert cache.stats == StorageStats(evictions=1)
    cache.put("x", payload(3))  # Evict the leaf c, never the older parent a.
    assert_contents(cache, a=2, b=1, x=3)
    assert cache.stats.evictions == 2


@pytest.mark.parametrize("entries,budget,sizes,expected", [
    (None, None, [2, 100], {"a": 2, "b": 100}),
    (1, None, [2, 100], {"b": 100}),
    (2, 10, [1, 1, 1], {"b": 1, "c": 1}),
    (4, 5, [2, 2, 4], {"c": 4}),
    (2, 5, [1, 1, 5], {"c": 5}),
    (1, 1, [0, 0], {"b": 0}),
])
def test_capacity_limits_and_defaults(entries, budget, sizes, expected):
    limits = {} if budget is None else {"max_bytes": budget}
    cache = make_cache(max_entries=entries, **limits)
    assert cache.max_entries == entries
    assert cache.max_bytes == budget
    with pytest.raises(AttributeError):
        cache.max_bytes = 16
    for key, size in zip("abc"[:len(sizes)], sizes, strict=True):
        cache.put(key, payload(size))
    assert_contents(cache, **expected)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "10"])
def test_invalid_max_bytes(bad):
    with pytest.raises(ValueError, match="max_bytes"):
        make_cache(max_bytes=bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_view_payload_bytes_and_copy_isolation(dtype):
    original = payload(100, dtype)[::25]
    expected = original.clone()
    cache = make_cache(max_bytes=original.nbytes)
    cache.put("a", original)
    original.zero_()
    assert cache.nbytes == expected.nbytes
    restored = cache.get("a")
    assert restored.device.type == "cpu"
    assert not cache.peek("a").is_pinned()
    assert torch.equal(restored, expected)
    restored.zero_()
    assert torch.equal(cache.get("a"), expected)
    with pytest.raises(TypeError):
        cache.put("b", "not a tensor")
    assert torch.equal(cache.peek("a"), expected)
    assert cache.stats == StorageStats()
    cache.put("b", payload(1))
    assert_contents(cache, b=1)


@pytest.mark.parametrize("budget", [1, 4, 6])
def test_tiered_disk_fallback_and_prefix_safe_promotion(tmp_path, budget):
    fast = make_cache(max_bytes=budget)
    slow = DiskKVCache(tmp_path, serializer=TensorSerializer())
    tiered = TieredStorage(fast=fast, slow=slow)
    for key, parent in [("a", None), ("b", "a"), ("c", "b")]:
        tiered.put(key, payload(2), parent_key=parent)
        assert torch.equal(slow.get(key), payload(2))
    assert len(fast) == budget // 2
    tiered.put("x", payload(budget))
    assert_contents(fast, x=budget)
    assert fast.stats.evictions == budget // 2
    # A child remains available on disk even without its parent in the fast tier.
    assert torch.equal(tiered.get("c"), payload(2))
    assert "c" not in fast
    assert tiered.tier_stats == TierStats(slow_hits=1)
    for key in "abc":  # Prefix lookups promote parents before children.
        assert torch.equal(tiered.get(key), payload(2))
        for tier in (fast, slow):
            for stored in "abcx":
                if stored in tier:
                    parent = tier.parent_of(stored)
                    assert parent is None or parent in tier
    assert tiered.tier_stats == TierStats(slow_hits=4, promotions=budget // 2)
    expected = {"x": 1} if budget == 1 else dict.fromkeys("abc"[:budget // 2], 2)
    assert_contents(fast, **expected)
    if "b" in fast:
        assert fast.parent_of("b") == "a"
    for key in "abc":
        assert torch.equal(tiered.get(key), payload(2))
    assert tiered.tier_stats.fast_hits == budget // 2
    assert tiered.get("missing") is None
    assert tiered.tier_stats.misses == 1
