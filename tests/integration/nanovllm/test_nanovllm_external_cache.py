"""N1: external prefix lookup for nano-vLLM blocks.

CPU tensors in nano-vLLM's KV cache layout stand in for the GPU cache;
N0's export_block / import_block move blocks in and out, exactly as
ModelRunner will in N2 and N3.
"""

import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.integrations.nanovllm.external_cache import (  # noqa: E402
    NanoVLLMExternalCache,
    max_reusable_blocks,
)
from nano_lmcache.integrations.nanovllm.kv_blocks import (  # noqa: E402
    export_block,
    import_block,
)
from nano_lmcache.integrations.nanovllm.layout import KVBlockLayout  # noqa: E402
from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.storage.memory import MemoryKVCache  # noqa: E402
from nano_lmcache.storage.tiered import TieredStorage  # noqa: E402
from nano_lmcache.tensor.serializers import TensorSerializer  # noqa: E402

pytestmark = pytest.mark.torch

BLOCK = 4
LAYOUT = KVBlockLayout(
    model="Qwen/Qwen3-0.6B",
    revision="abc123",
    dtype="bfloat16",
    num_layers=3,
    num_kv_heads=2,
    head_dim=4,
    block_size=BLOCK,
)


def make_kv_cache(seed=0, num_blocks=16, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    shape = (2, LAYOUT.num_layers, num_blocks, BLOCK, LAYOUT.num_kv_heads,
             LAYOUT.head_dim)
    return torch.randn(shape, generator=generator).to(torch.bfloat16).to(device)


def tokens(num_blocks, extra=0, first=100):
    return list(range(first, first + num_blocks * BLOCK + extra))


def save(cache, token_ids, kv_cache, block_table, num_completed):
    """What N3 will do after a step: export and store the missing blocks."""
    missing = cache.missing_blocks(token_ids, num_completed)
    payloads = [export_block(kv_cache, block_table[i]) for i in missing]
    return cache.store(token_ids, missing.start, payloads)


def restore(cache, token_ids, kv_cache, block_ids, num_gpu_cached=0):
    """What N2 will do in allocate and load_kv_blocks."""
    found = cache.count_hits(token_ids, num_gpu_cached)
    payloads = cache.fetch(token_ids, num_gpu_cached, min(found, len(block_ids)))
    for block_id, payload in zip(block_ids, payloads, strict=False):
        import_block(kv_cache, block_id, payload)
    return len(payloads)


class RecordingStorage(MemoryKVCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gets, self.puts = [], []

    def get(self, key):
        self.gets.append(key)
        return super().get(key)

    def put(self, key, value, *, parent_key=None):
        self.puts.append(key)
        super().put(key, value, parent_key=parent_key)


@pytest.fixture
def cache():
    return NanoVLLMExternalCache(LAYOUT, MemoryKVCache())


# ---- keys -------------------------------------------------------------


def test_storage_must_support_membership_checks():
    class GetPutOnly:
        def put(self, key, value, *, parent_key=None): ...

        def get(self, key):
            return None

    with pytest.raises(TypeError, match="__contains__"):
        NanoVLLMExternalCache(LAYOUT, GetPutOnly())


def test_block_keys_are_nano_lmcache_chunk_keys(cache):
    prompt = tokens(3, extra=2)
    reference = ChunkedPrefixCache(chunk_size=BLOCK, namespace=LAYOUT.namespace())
    reference.store(prompt, ["a", "b", "c"])

    assert cache.block_keys(prompt) == reference.lookup(prompt).keys


def test_block_keys_cover_complete_blocks_only(cache):
    assert len(cache.block_keys(tokens(3, extra=3))) == 3
    assert len(cache.block_keys(tokens(3, extra=3), num_blocks=2)) == 2
    with pytest.raises(ValueError):
        cache.block_keys(tokens(3, extra=3), num_blocks=4)


def test_a_block_key_depends_on_everything_before_it(cache):
    tail = tokens(1, first=900)
    first = cache.block_keys(tokens(1, first=100) + tail)
    second = cache.block_keys(tokens(1, first=500) + tail)

    assert first[1] != second[1]


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(0, 0), (1, 0), (BLOCK, 0), (BLOCK + 1, 1), (2 * BLOCK, 1),
     (2 * BLOCK + 1, 2)],
)
def test_max_reusable_blocks_never_includes_the_last_block(num_tokens, expected):
    # nano-vLLM's can_allocate walks range(num_blocks - 1).
    assert max_reusable_blocks(num_tokens, BLOCK) == expected


# ---- counting hits ----------------------------------------------------


def test_cold_cache_has_no_hits(cache):
    assert cache.count_hits(tokens(3, extra=1), 0) == 0


def test_hits_follow_the_l1_hits(cache):
    prompt = tokens(4, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2, 3], 4)

    assert cache.count_hits(prompt, 0) == 4
    # nano-vLLM already has blocks 0 and 1 on the GPU.
    assert cache.count_hits(prompt, 2) == 2


def test_a_prompt_of_whole_blocks_leaves_its_last_block_to_compute(cache):
    prompt = tokens(3)  # exactly 3 full blocks
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)

    assert len(cache.storage) == 3
    assert cache.count_hits(prompt, 0) == 2


def test_a_longer_prompt_can_use_a_stored_last_block(cache):
    save(cache, tokens(3), make_kv_cache(), [0, 1, 2], 3)

    assert cache.count_hits(tokens(3, extra=1), 0) == 3


def test_counting_stops_at_the_first_missing_block(cache):
    shared = tokens(2)
    save(cache, shared + tokens(2, first=500), make_kv_cache(), [0, 1, 2, 3], 4)

    prompt = shared + tokens(2, extra=1, first=900)

    assert cache.count_hits(prompt, 0) == 2


def test_counting_never_reads_a_payload():
    storage = RecordingStorage()
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    prompt = tokens(3, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)

    for _ in range(10):  # nano-vLLM asks on every step while a request waits
        cache.count_hits(prompt, 0)

    assert storage.gets == []


def test_more_l1_hits_than_reusable_blocks_is_an_error(cache):
    with pytest.raises(ValueError):
        cache.count_hits(tokens(2), 2)


# ---- fetching ---------------------------------------------------------


def test_fetch_returns_payloads_in_block_order(cache):
    kv_cache = make_kv_cache(seed=1)
    prompt = tokens(3, extra=1)
    save(cache, prompt, kv_cache, [5, 2, 7], 3)

    payloads = cache.fetch(prompt, 0, 3)

    for payload, block_id in zip(payloads, [5, 2, 7], strict=True):
        assert torch.equal(payload, export_block(kv_cache, block_id))


def test_fetch_from_a_start_block_skips_the_l1_hits():
    storage = RecordingStorage()
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    prompt = tokens(4, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2, 3], 4)

    payloads = cache.fetch(prompt, 2, 2)

    assert len(payloads) == 2
    assert storage.gets == list(cache.block_keys(prompt)[2:4])


def test_fetch_returns_fewer_blocks_if_some_were_evicted():
    storage = MemoryKVCache(max_entries=8)
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    prompt = tokens(3, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)
    promised = cache.count_hits(prompt, 0)

    # Unrelated traffic evicts the tail of the chain before allocate runs.
    for i in range(8):
        storage.put(f"other-{i}", i)

    assert promised == 3
    assert len(cache.fetch(prompt, 0, promised)) < promised


def test_fetch_never_crosses_the_last_block_limit(cache):
    prompt = tokens(3)
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)

    with pytest.raises(ValueError):
        cache.fetch(prompt, 0, 3)


@pytest.mark.parametrize(("start", "count"), [(-1, 1), (0, -1)])
def test_fetch_rejects_negative_arguments(cache, start, count):
    with pytest.raises(ValueError):
        cache.fetch(tokens(3, extra=1), start, count)


def test_fetching_nothing_reads_nothing(cache):
    assert cache.fetch(tokens(3, extra=1), 1, 0) == []


# ---- what to store ----------------------------------------------------


def test_missing_blocks_on_a_cold_cache_is_everything(cache):
    assert cache.missing_blocks(tokens(3, extra=1), 3) == range(0, 3)


def test_missing_blocks_starts_after_what_is_stored(cache):
    prompt = tokens(4, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2, 3], 2)

    assert cache.missing_blocks(prompt, 4) == range(2, 4)
    assert cache.missing_blocks(prompt, 2) == range(2, 2)


def test_missing_blocks_cannot_exceed_complete_blocks(cache):
    with pytest.raises(ValueError):
        cache.missing_blocks(tokens(3, extra=1), 4)


def test_store_links_each_block_to_the_previous_one(cache):
    prompt = tokens(3, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)

    keys = cache.block_keys(prompt)
    assert cache.storage.parent_of(keys[0]) is None
    assert cache.storage.parent_of(keys[1]) == keys[0]
    assert cache.storage.parent_of(keys[2]) == keys[1]


def test_store_after_a_missing_prefix_is_dropped(cache):
    kv_cache = make_kv_cache()
    prompt = tokens(3, extra=1)

    # Block 0 was never stored, so block 1 would be unreachable.
    stored = cache.store(prompt, 1, [export_block(kv_cache, 1)])

    assert stored == 0
    assert len(cache.storage) == 0


def test_store_checks_its_arguments(cache):
    kv_cache = make_kv_cache()
    payload = export_block(kv_cache, 0)
    prompt = tokens(2, extra=1)

    with pytest.raises(ValueError):
        cache.store(prompt, 0, [payload] * 3)  # only 2 complete blocks
    with pytest.raises(ValueError):
        cache.store(prompt, -1, [payload])
    with pytest.raises(ValueError):
        cache.store(prompt, 0, [payload, None])
    with pytest.raises(ValueError):
        cache.store(prompt, 0, [torch.zeros(2, 2)])  # not a chunk


def test_chunked_prefill_stores_every_block_exactly_once():
    storage = RecordingStorage()
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    kv_cache = make_kv_cache()
    prompt = tokens(4, extra=1)
    table = [3, 1, 4, 5, 9]

    # hash_blocks reports 1, then 3, then 4 completed blocks.
    for completed in (1, 3, 4, 4):
        save(cache, prompt, kv_cache, table, completed)

    assert sorted(storage.puts) == sorted(cache.block_keys(prompt))


def test_store_reports_what_capacity_kept():
    storage = MemoryKVCache(max_entries=2)
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    prompt = tokens(3, extra=1)

    assert save(cache, prompt, make_kv_cache(), [0, 1, 2], 3) == 2


def test_stats_track_what_happened(cache):
    prompt = tokens(3, extra=1)
    save(cache, prompt, make_kv_cache(), [0, 1, 2], 3)
    cache.count_hits(prompt, 1)
    cache.fetch(prompt, 1, 2)

    assert cache.stats.stored_blocks == 3
    assert cache.stats.hit_blocks == 2
    assert cache.stats.fetched_blocks == 2


# ---- isolation --------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [{"model": "Qwen/Qwen3-1.7B"}, {"revision": "other"}, {"dtype": "float16"},
     {"num_kv_heads": 4}, {"config_fingerprint": "other"}],
)
def test_a_different_model_or_layout_never_hits(change):
    import dataclasses

    storage = MemoryKVCache()
    prompt = tokens(3, extra=1)
    save(NanoVLLMExternalCache(LAYOUT, storage), prompt, make_kv_cache(),
         [0, 1, 2], 3)

    other = NanoVLLMExternalCache(dataclasses.replace(LAYOUT, **change), storage)

    assert other.count_hits(prompt, 0) == 0


# ---- the full save -> restore path on every backend --------------------


BACKENDS = {
    "memory": lambda path: MemoryKVCache(),
    "disk": lambda path: DiskKVCache(path, serializer=TensorSerializer()),
    "tiered": lambda path: TieredStorage(
        fast=MemoryKVCache(max_entries=1),
        slow=DiskKVCache(path, serializer=TensorSerializer()),
    ),
}


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_blocks_saved_by_one_engine_restore_into_another(backend, tmp_path):
    engine_a = make_kv_cache(seed=7)
    prompt = tokens(3, extra=3)
    writer = NanoVLLMExternalCache(LAYOUT, BACKENDS[backend](tmp_path))
    save(writer, prompt, engine_a, [2, 8, 5], 3)

    # Disk-backed storage is reopened from its directory, as after a
    # restart; memory storage is shared, as inside one process.
    storage = writer.storage if backend == "memory" else BACKENDS[backend](tmp_path)
    reader = NanoVLLMExternalCache(LAYOUT, storage)
    engine_b = torch.zeros_like(engine_a)

    assert restore(reader, prompt, engine_b, [1, 3, 4]) == 3
    for a, b in [(2, 1), (8, 3), (5, 4)]:
        assert torch.equal(engine_b[:, :, b], engine_a[:, :, a])


@pytest.mark.gpu
def test_cpu_offload_restores_into_a_gpu_kv_cache():
    from nano_lmcache.tensor.offload import CPUOffloadKVCache

    engine_a = make_kv_cache(seed=5, device="cuda")
    storage = CPUOffloadKVCache(device="cuda")
    cache = NanoVLLMExternalCache(LAYOUT, storage)
    prompt = tokens(3, extra=1)
    save(cache, prompt, engine_a, [6, 2, 4], 3)

    assert all(storage.peek(k).device.type == "cpu" for k in storage.keys())
    engine_b = torch.zeros_like(engine_a)
    assert restore(cache, prompt, engine_b, [1, 3, 5]) == 3
    for a, b in [(6, 1), (2, 3), (4, 5)]:
        assert torch.equal(engine_b[:, :, b], engine_a[:, :, a])
