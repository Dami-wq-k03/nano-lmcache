import os
import random

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.tiered import TieredStorage, TierStats


def make_tiered(tmp_path, *, fast=None, slow=None):
    return TieredStorage(
        fast=MemoryKVCache(max_entries=fast),
        slow=DiskKVCache(tmp_path / "disk", max_entries=slow),
    )


def test_satisfies_storage_protocol(tmp_path):
    assert isinstance(make_tiered(tmp_path), CacheStorage)


def test_put_writes_both_tiers(tmp_path):
    tiered = make_tiered(tmp_path)

    tiered.put("a", "kv-a")

    assert "a" in tiered.fast
    assert "a" in tiered.slow


def test_get_from_the_fast_tier(tmp_path):
    tiered = make_tiered(tmp_path)
    tiered.put("a", "kv-a")

    assert tiered.get("a") == "kv-a"
    assert tiered.tier_stats == TierStats(fast_hits=1)


def test_miss_in_both_tiers(tmp_path):
    tiered = make_tiered(tmp_path)

    assert tiered.get("missing") is None
    assert tiered.tier_stats == TierStats(misses=1)


def test_entry_evicted_from_memory_still_hits_on_disk(tmp_path):
    tiered = make_tiered(tmp_path, fast=1)
    tiered.put("a", "kv-a")
    tiered.put("b", "kv-b")  # memory evicts a; disk keeps both

    assert "a" not in tiered.fast

    assert tiered.get("a") == "kv-a"
    assert tiered.tier_stats.slow_hits == 1


def test_slow_hit_is_promoted_to_the_fast_tier(tmp_path):
    tiered = make_tiered(tmp_path, fast=1)
    tiered.put("a", "kv-a")
    tiered.put("b", "kv-b")

    tiered.get("a")

    assert "a" in tiered.fast
    assert tiered.tier_stats.promotions == 1
    tiered.get("a")
    assert tiered.tier_stats.fast_hits == 1


def test_promotion_keeps_the_parent_link(tmp_path):
    tiered = make_tiered(tmp_path, fast=2)
    tiered.put("a", 1)
    tiered.put("b", 2, parent_key="a")
    tiered.put("x", 0)
    tiered.put("y", 0)  # memory now holds only x and y

    # A lookup walks root first, so a is promoted before b.
    tiered.get("a")
    tiered.get("b")

    assert tiered.fast.parent_of("b") == "a"
    # b depends on a, so filling memory evicts b, never a.
    tiered.put("z", 0)
    assert "a" in tiered.fast


def test_child_is_not_promoted_without_its_parent(tmp_path):
    tiered = make_tiered(tmp_path, fast=2)
    tiered.put("a", 1)
    tiered.put("b", 2, parent_key="a")
    tiered.put("x", 0)
    tiered.put("y", 0)

    # Reading b without a: served from disk, but memory must refuse it,
    # since b would be unreachable there without a.
    assert tiered.get("b") == 2
    assert "b" not in tiered.fast
    assert tiered.tier_stats.promotions == 0


def test_cache_hits_after_a_restart_then_warms_memory(tmp_path):
    tokens = [1, 2, 3, 4, 5, 6]
    writer = ChunkedPrefixCache(
        chunk_size=2, namespace="model-a", storage=make_tiered(tmp_path)
    )
    writer.store(tokens, ["kv-0", "kv-1", "kv-2"])

    # New process: memory is empty, the disk directory is not.
    storage = make_tiered(tmp_path)
    reader = ChunkedPrefixCache(chunk_size=2, namespace="model-a", storage=storage)

    assert reader.lookup(tokens).matched_chunks == 3
    assert storage.tier_stats.slow_hits == 3

    assert reader.lookup(tokens).matched_chunks == 3
    assert storage.tier_stats.fast_hits == 3


def test_memory_smaller_than_the_prompt_still_serves_it_from_disk(tmp_path):
    storage = make_tiered(tmp_path, fast=2)
    cache = ChunkedPrefixCache(chunk_size=2, namespace="model-a", storage=storage)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    cache.store(tokens, ["kv-0", "kv-1", "kv-2", "kv-3"])
    result = cache.lookup(tokens)

    assert result.matched_chunks == 4
    assert result.values == ("kv-0", "kv-1", "kv-2", "kv-3")


def test_tier_stats_hit_rates():
    stats = TierStats(fast_hits=6, slow_hits=3, misses=1)

    assert stats.lookups == 10
    assert stats.fast_hit_rate == 0.6
    assert stats.hit_rate == 0.9
    assert TierStats().hit_rate == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_no_tier_ever_holds_an_orphan(tmp_path, monkeypatch, seed):
    """Random workload; after every step each tier keeps whole chains."""
    # This checks chain bookkeeping, not durability; skipping fsync keeps
    # hundreds of disk writes fast.
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    rng = random.Random(seed)
    tiered = make_tiered(tmp_path / str(seed), fast=4, slow=7)
    roots = [f"r{i}" for i in range(4)]
    written: dict[str, str | None] = {}

    for step in range(200):
        if rng.random() < 0.6:
            parent = rng.choice([None, *written]) if written else None
            key = f"k{step}"
            tiered.put(key, step, parent_key=parent)
            written[key] = parent
        elif written:
            tiered.get(rng.choice([*written, *roots]))

        for tier in (tiered.fast, tiered.slow):
            for key in written:
                if key in tier:
                    parent = tier.parent_of(key)
                    assert parent is None or parent in tier, (step, key)


def test_contains_checks_both_tiers_without_counting_a_use(tmp_path):
    tiered = make_tiered(tmp_path, fast=1)
    tiered.put("a", "kv-a")
    tiered.put("b", "kv-b")  # a is now only on disk

    assert "a" in tiered
    assert "b" in tiered
    assert "missing" not in tiered
    assert tiered.tier_stats == TierStats()
    assert "a" not in tiered.fast  # membership does not promote
