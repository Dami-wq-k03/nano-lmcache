"""N2 and N3: restore L2 hits into nano-vLLM's GPU blocks, and store new ones.

Two levels:

- Scheduler level (CPU only): nano-vLLM's real `Scheduler`,
  `BlockManager`, and `Sequence`, with a fake model runner. The fake
  writes deterministic KV for every token it computes, and picks each
  next token from the KV actually sitting in the sequence's GPU blocks,
  so a restored block with wrong contents changes the output.
- Engine level (GPU and a real model): `LLM.generate` with nano-lmcache
  attached. Set NANO_LMCACHE_TEST_MODEL to a local model directory, for
  example Qwen3-0.6B, and pick a free GPU with CUDA_VISIBLE_DEVICES.
"""

import atexit
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nanovllm")

from nanovllm.engine.block_manager import BlockManager  # noqa: E402
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

from nano_lmcache.integrations.nanovllm import (  # noqa: E402
    KVBlockLayout,
    NanoVLLMExternalCache,
    export_block,
)
from nano_lmcache.integrations.nanovllm.engine import (  # noqa: E402
    attach,
    detach,
)
from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.storage.memory import MemoryKVCache  # noqa: E402
from nano_lmcache.tensor.serializers import TensorSerializer  # noqa: E402

pytestmark = pytest.mark.nanovllm

BLOCK = 256  # nano-vLLM requires a multiple of 256
LAYOUT = KVBlockLayout(
    model="fake", revision="r", dtype="float32", num_layers=2,
    num_kv_heads=2, head_dim=4, block_size=BLOCK,
)


# ---- a fake engine around nano-vLLM's real scheduler -------------------


def fake_kv(token: int, position: int) -> float:
    """KV value a 'model' writes for one token; depends on token and position."""
    return float((token * 31 + position * 7) % 1009)


class FakeModelRunner:
    """Stands in for nano-vLLM's ModelRunner, without a model or a GPU."""

    world_size = 1

    def __init__(self, num_blocks: int = 24):
        self.kv_cache = torch.zeros(
            2, LAYOUT.num_layers, num_blocks, BLOCK, LAYOUT.num_kv_heads,
            LAYOUT.head_dim,
        )
        self.computed_tokens = 0

    def run(self, seqs, is_prefill):
        next_tokens = []
        for seq in seqs:
            start = seq.num_cached_tokens
            for position in range(start, start + seq.num_scheduled_tokens):
                block, offset = divmod(position, BLOCK)
                value = fake_kv(seq.token_ids[position], position)
                self.kv_cache[:, :, seq.block_table[block], offset] = value
            self.computed_tokens += seq.num_scheduled_tokens
            # The next token depends on every cached K/V of the sequence,
            # including restored blocks.
            end = start + seq.num_scheduled_tokens
            total = sum(
                float(self.kv_cache[0, 0, seq.block_table[p // BLOCK], p % BLOCK, 0, 0])
                for p in range(end)
            )
            next_tokens.append(int(total) % 50000 + 1)
        return next_tokens

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)


class FakeLLM:
    """The parts of nano-vLLM's LLMEngine that `attach` and `generate` use."""

    def __init__(self, num_blocks=24, max_num_batched_tokens=16384,
                 max_num_seqs=8):
        Sequence.block_size = BLOCK
        config = SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos=-1,
            kvcache_block_size=BLOCK,
            num_kvcache_blocks=num_blocks,
        )
        self.scheduler = Scheduler(config)
        self.model_runner = FakeModelRunner(num_blocks)

    def generate(self, prompts, max_tokens=4):
        seqs = [Sequence(p, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
                for p in prompts]
        for seq in seqs:
            self.scheduler.add(seq)
        while not self.scheduler.is_finished():
            batch, is_prefill = self.scheduler.schedule()
            tokens = self.model_runner.call("run", batch, is_prefill)
            self.scheduler.postprocess(batch, tokens, is_prefill)
        return [seq.completion_token_ids for seq in seqs]


def prompt(num_blocks, extra=0, first=1000):
    return list(range(first, first + num_blocks * BLOCK + extra))


def expected_payload(token_ids, block):
    """The chunk a correct model run would have written for this block."""
    payload = torch.zeros(LAYOUT.chunk_shape)
    for offset in range(BLOCK):
        position = block * BLOCK + offset
        payload[:, :, :, offset] = fake_kv(token_ids[position], position)
    return payload


def seed_l2(external, token_ids, num_blocks):
    """Put correct KV for the first blocks of a prompt into L2."""
    payloads = [expected_payload(token_ids, b) for b in range(num_blocks)]
    assert external.store(token_ids, 0, payloads) == num_blocks


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


def assert_l2_holds_correct_kv(external, token_ids, num_blocks):
    keys = external.block_keys(token_ids, num_blocks)
    for block, key in enumerate(keys):
        assert torch.equal(external.storage.get(key),
                           expected_payload(token_ids, block)), block


def attached(storage=None, **llm_kwargs):
    llm = FakeLLM(**llm_kwargs)
    # Not `storage or ...`: an empty cache has len() 0, which is falsy.
    external = NanoVLLMExternalCache(
        LAYOUT, storage if storage is not None else MemoryKVCache()
    )
    manager = attach(llm, external)
    return llm, external, manager


# ---- attach -----------------------------------------------------------


def test_attach_replaces_the_block_manager():
    llm, external, manager = attached()

    assert llm.scheduler.block_manager is manager
    assert isinstance(manager, BlockManager)
    assert manager.external is external


def test_attach_rejects_a_kv_cache_of_another_layout():
    llm = FakeLLM()
    other = KVBlockLayout(model="fake", revision="r", dtype="float32",
                          num_layers=2, num_kv_heads=4, head_dim=4,
                          block_size=BLOCK)

    with pytest.raises(ValueError, match="layout"):
        attach(llm, NanoVLLMExternalCache(other, MemoryKVCache()))


def test_attach_rejects_tensor_parallel_engines():
    llm = FakeLLM()
    llm.model_runner.world_size = 2

    with pytest.raises(NotImplementedError):
        attach(llm, NanoVLLMExternalCache(LAYOUT, MemoryKVCache()))


def test_attach_rejects_a_busy_engine():
    llm = FakeLLM()
    llm.scheduler.add(Sequence(prompt(1, extra=1)))

    with pytest.raises(RuntimeError):
        attach(llm, NanoVLLMExternalCache(LAYOUT, MemoryKVCache()))


def test_detach_gives_back_plain_nanovllm_with_an_empty_l1():
    token_ids = prompt(3, extra=10)
    reference = FakeLLM().generate([token_ids])
    llm, external, manager = attached()
    llm.generate([token_ids])

    plain = detach(llm)

    assert type(plain) is BlockManager
    assert llm.scheduler.block_manager is plain
    assert not plain.hash_to_block_id  # L1 starts empty
    llm.model_runner.computed_tokens = 0
    assert llm.generate([token_ids]) == reference
    # Nothing came from L2 and nothing new went into it.
    assert llm.model_runner.computed_tokens == len(token_ids) + 3
    assert manager.stats.l2_hit_blocks == 0
    assert len(external.storage) == 3


def test_detach_then_attach_again_uses_l2():
    token_ids = prompt(3, extra=10)
    llm, external, _ = attached()
    llm.generate([token_ids])
    detach(llm)

    manager = attach(llm, external)
    llm.generate([token_ids])

    assert manager.stats.l2_hit_blocks == 3


def test_detach_works_on_an_engine_that_was_never_attached():
    llm = FakeLLM()

    assert type(detach(llm)) is BlockManager
    assert llm.generate([prompt(1, extra=3)])


def test_detach_rejects_a_busy_engine():
    llm, _, _ = attached()
    llm.scheduler.add(Sequence(prompt(1, extra=1)))

    with pytest.raises(RuntimeError):
        detach(llm)


# ---- restoring --------------------------------------------------------


def test_without_l2_hits_the_engine_behaves_as_before():
    plain = FakeLLM()
    reference = plain.generate([prompt(3, extra=10)])

    llm, _, manager = attached()
    assert llm.generate([prompt(3, extra=10)]) == reference
    assert manager.stats.l2_hit_blocks == 0


def test_l2_hits_skip_prefill_and_give_the_same_output():
    token_ids = prompt(3, extra=10)
    reference = FakeLLM().generate([token_ids])

    llm, external, manager = attached()
    seed_l2(external, token_ids, 3)
    output = llm.generate([token_ids])

    assert output == reference
    assert manager.stats.l2_hit_blocks == 3
    # Only the 10-token tail of the prompt, plus decode steps, was computed.
    assert llm.model_runner.computed_tokens == 10 + 3


def test_restored_blocks_hold_exactly_the_stored_kv():
    token_ids = prompt(2, extra=5)
    llm, external, manager = attached()
    seed_l2(external, token_ids, 2)

    seq = Sequence(token_ids)
    n = manager.can_allocate(seq)
    manager.allocate(seq, n)
    llm.model_runner.run([], True)  # flushes pending loads, computes nothing

    for block in range(2):
        assert torch.equal(
            export_block(llm.model_runner.kv_cache, seq.block_table[block]),
            expected_payload(token_ids, block),
        )


def test_last_block_is_always_computed():
    token_ids = prompt(3)  # exactly 3 full blocks
    llm, external, manager = attached()
    seed_l2(external, token_ids, 3)

    llm.generate([token_ids])

    assert manager.stats.l2_hit_blocks == 2
    assert llm.model_runner.computed_tokens == BLOCK + 3


def test_l1_hits_come_first_and_only_the_rest_is_fetched():
    storage = RecordingStorage()
    llm, external, manager = attached(storage)
    shared = prompt(1)
    llm.generate([shared + prompt(1, extra=1, first=9000)])  # block 0 now in L1
    token_ids = shared + prompt(2, extra=3, first=5000)
    seed_l2(external, token_ids, 3)
    storage.gets.clear()

    llm.generate([token_ids])

    assert manager.stats.l1_hit_blocks == 1
    assert manager.stats.l2_hit_blocks == 2
    assert storage.gets == list(external.block_keys(token_ids)[1:3])


def test_restored_blocks_get_the_same_l1_hashes_nanovllm_computes():
    token_ids = prompt(3, extra=1)
    llm, external, manager = attached()
    seed_l2(external, token_ids, 3)
    seq = Sequence(token_ids)
    manager.allocate(seq, manager.can_allocate(seq))

    reference = BlockManager(4, BLOCK)
    ref_seq = Sequence(token_ids)
    reference.allocate(ref_seq, 0)
    ref_seq.num_scheduled_tokens = len(token_ids)
    reference.hash_blocks(ref_seq)

    for block in range(3):
        assert (manager.blocks[seq.block_table[block]].hash
                == reference.blocks[ref_seq.block_table[block]].hash)


def test_after_a_restore_later_requests_hit_l1():
    storage = RecordingStorage()
    llm, external, manager = attached(storage)
    token_ids = prompt(3, extra=4)
    seed_l2(external, token_ids, 3)
    llm.generate([token_ids])
    storage.gets.clear()

    llm.generate([token_ids])

    assert manager.stats.l1_hit_blocks == 3
    assert storage.gets == []


def test_requests_in_one_batch_share_a_single_restore():
    storage = RecordingStorage()
    llm, external, manager = attached(storage)
    shared = prompt(2)
    first = shared + prompt(1, extra=1, first=7000)
    second = shared + prompt(1, extra=2, first=8000)
    seed_l2(external, first, 2)
    reference = FakeLLM().generate([first, second])
    storage.gets.clear()

    assert llm.generate([first, second]) == reference
    # The second request hits the first one's restored blocks in L1.
    assert len(storage.gets) == 2


def test_chunked_prefill_after_a_restore():
    token_ids = prompt(4, extra=100)
    reference = FakeLLM().generate([token_ids])

    llm, external, manager = attached(max_num_batched_tokens=300)
    seed_l2(external, token_ids, 2)

    assert llm.generate([token_ids]) == reference
    assert manager.stats.l2_hit_blocks == 2


def test_no_free_gpu_blocks_means_no_restore():
    # An L2 hit still needs a free GPU block to load into.
    llm, external, manager = attached(num_blocks=3)
    token_ids = prompt(3, extra=1)  # needs 4 GPU blocks
    seed_l2(external, token_ids, 3)
    seq = Sequence(token_ids)

    assert manager.can_allocate(seq) == -1


def test_a_hit_that_disappears_before_allocate_is_computed_instead():
    storage = MemoryKVCache(max_entries=4)
    token_ids = prompt(3, extra=7)
    reference = FakeLLM().generate([token_ids])
    llm, external, manager = attached(storage)
    seed_l2(external, token_ids, 3)

    seq = Sequence(token_ids, SamplingParams(max_tokens=4, ignore_eos=True))
    promised = manager.can_allocate(seq)
    for i in range(4):  # unrelated traffic evicts the chain's tail
        storage.put(f"other-{i}", i)
    manager.allocate(seq, promised)

    assert promised == 3
    assert seq.num_cached_tokens < 3 * BLOCK
    restored = seq.num_cached_tokens // BLOCK
    # Blocks that never received their KV must not be findable in L1.
    for block in range(restored, 3):
        assert manager.blocks[seq.block_table[block]].hash == -1
    manager.deallocate(seq)

    assert llm.generate([token_ids]) == reference


def test_preempted_requests_are_restored_again():
    token_ids = prompt(3, extra=2)
    reference = FakeLLM().generate([token_ids])
    llm, external, manager = attached()
    seed_l2(external, token_ids, 3)
    seq = Sequence(token_ids)
    manager.allocate(seq, manager.can_allocate(seq))
    llm.model_runner.run([], True)
    manager.deallocate(seq)  # what Scheduler.preempt does

    assert llm.generate([token_ids]) == reference


# ---- storing (N3) -----------------------------------------------------


def test_a_cold_request_stores_its_complete_blocks():
    token_ids = prompt(3, extra=10)
    llm, external, manager = attached()

    llm.generate([token_ids])

    assert manager.stats.stored_blocks == 3
    assert_l2_holds_correct_kv(external, token_ids, 3)


def test_stored_blocks_are_restored_by_a_fresh_block_manager():
    """The whole loop: store, look up, restore, reuse, with no manual help."""
    token_ids = prompt(3, extra=10)
    reference = FakeLLM().generate([token_ids])
    llm, external, _ = attached()
    llm.generate([token_ids])

    fresh = attach(llm, external)  # empty L1, same L2
    llm.model_runner.computed_tokens = 0

    assert llm.generate([token_ids]) == reference
    assert fresh.stats.l2_hit_blocks == 3
    assert llm.model_runner.computed_tokens == 10 + 3


def test_the_incomplete_last_block_is_not_stored():
    llm, external, _ = attached()

    llm.generate([prompt(2, extra=100)], max_tokens=4)

    assert len(external.storage) == 2


def test_blocks_completed_during_decode_are_stored():
    token_ids = prompt(1, extra=250)  # 6 tokens short of two full blocks
    llm, external, manager = attached()

    (completion,) = llm.generate([token_ids], max_tokens=10)

    full = token_ids + completion
    assert manager.stats.stored_blocks == 2
    assert_l2_holds_correct_kv(external, full, 2)


def test_each_block_is_stored_once_through_chunked_prefill():
    storage = RecordingStorage()
    token_ids = prompt(4, extra=100)
    llm, external, _ = attached(storage, max_num_batched_tokens=300)

    llm.generate([token_ids])

    assert sorted(storage.puts) == sorted(external.block_keys(token_ids))
    assert_l2_holds_correct_kv(external, token_ids, 4)


def test_blocks_already_in_l2_are_not_stored_again():
    storage = RecordingStorage()
    token_ids = prompt(3, extra=10)
    llm, external, _ = attached(storage)
    llm.generate([token_ids])
    attach(llm, external)
    storage.puts.clear()

    llm.generate([token_ids])  # restored from L2
    llm.generate([token_ids])  # hits L1

    assert storage.puts == []


def test_l2_is_repaired_from_l1_after_losing_entries():
    storage = MemoryKVCache()
    token_ids = prompt(3, extra=10)
    llm, external, _ = attached(storage)
    llm.generate([token_ids])
    longer = token_ids[: 3 * BLOCK] + prompt(1, extra=5, first=60000)
    # L2 loses everything, but L1 still holds the first three blocks.
    attach_storage = MemoryKVCache()
    llm.scheduler.block_manager.external = NanoVLLMExternalCache(LAYOUT, attach_storage)

    llm.generate([longer])

    assert_l2_holds_correct_kv(llm.scheduler.block_manager.external, longer, 4)


def test_finished_requests_are_exported_before_their_blocks_are_reused():
    # Only 4 GPU blocks: the second request reuses the first one's blocks.
    llm, external, _ = attached(num_blocks=4)
    first = prompt(3, extra=10, first=1000)
    second = prompt(3, extra=10, first=30000)

    llm.generate([first])
    llm.generate([second])

    assert_l2_holds_correct_kv(external, first, 3)
    assert_l2_holds_correct_kv(external, second, 3)


def test_capacity_limited_l2_counts_what_it_kept():
    llm, _, manager = attached(MemoryKVCache(max_entries=2))

    llm.generate([prompt(3, extra=10)])

    assert manager.stats.stored_blocks == 2


def test_steps_that_complete_no_block_do_no_work(monkeypatch):
    llm, external, _ = attached()
    calls = []
    original = external.missing_blocks
    monkeypatch.setattr(
        external, "missing_blocks", lambda *a: calls.append(a) or original(*a)
    )

    # Prefill completes block 0; the next 40 decode steps complete nothing.
    llm.generate([prompt(1, extra=10)], max_tokens=40)

    assert len(calls) == 1


# ---- the real engine --------------------------------------------------


MODEL = os.environ.get("NANO_LMCACHE_TEST_MODEL", "")


@pytest.fixture(scope="module")
def llm():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not os.path.isdir(MODEL):
        pytest.skip("set NANO_LMCACHE_TEST_MODEL to a local model directory")
    from nanovllm import LLM

    engine = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
                 gpu_memory_utilization=0.5, max_model_len=4096)
    yield engine
    atexit.unregister(engine.exit)
    engine.exit()


def empty_l1(engine, external):
    """Attach a fresh block manager and scrub the GPU KV cache.

    In one process, a fresh block manager hands out the same physical
    blocks again, and they still hold the KV an earlier request computed.
    A restore that copied nothing would then pass by accident. Filling the
    whole KV cache with noise first means only blocks that were really
    restored (or recomputed) hold correct KV.
    """
    manager = attach(engine, external)
    engine.model_runner.kv_cache.normal_()
    return manager


def real_layout(engine):
    return KVBlockLayout.from_hf_config(
        engine.model_runner.config.hf_config, model="test-model", revision="local"
    )


def long_prompt(engine):
    text = ("The city council met on Tuesday to discuss the new transit plan, "
            "which adds bus lines and extends the tram to the airport. ") * 40
    token_ids = engine.tokenizer.encode(text)
    return token_ids[: 3 * BLOCK + 37]


@pytest.mark.gpu
def test_real_engine_stores_restores_and_generates_the_same_tokens(llm):
    params = SamplingParams(temperature=1e-6, max_tokens=16)
    token_ids = long_prompt(llm)
    external = NanoVLLMExternalCache(real_layout(llm), MemoryKVCache())

    first = attach(llm, external)
    torch.manual_seed(0)
    (cold,) = llm.generate([token_ids], params, use_tqdm=False)
    # N3: the cold request stored its complete blocks on its own.
    assert first.stats.stored_blocks == 3

    # A fresh block manager: L1 is empty, L2 has the prompt's blocks.
    manager = empty_l1(llm, external)
    torch.manual_seed(0)
    (restored,) = llm.generate([token_ids], params, use_tqdm=False)

    assert manager.stats.l2_hit_blocks == 3
    assert manager.stats.prefill_tokens == len(token_ids) - 3 * BLOCK
    assert restored["token_ids"] == cold["token_ids"]

    # And the restored blocks now serve later requests from L1.
    torch.manual_seed(0)
    (warm,) = llm.generate([token_ids], params, use_tqdm=False)
    assert manager.stats.l1_hit_blocks == 3
    assert warm["token_ids"] == cold["token_ids"]


# ---- N4: correctness against a no-cache baseline -----------------------
#
# Every test compares against `detach(llm)`: plain nano-vLLM with an empty
# L1, so the whole prompt is computed. Generation uses a tiny temperature
# (nano-vLLM forbids 0) and a fixed seed, because nano-vLLM breaks ties
# between equally likely tokens with random noise.

N4_PARAMS = SamplingParams(temperature=1e-6, max_tokens=12)


def generate(engine, prompts):
    torch.manual_seed(0)
    return [out["token_ids"] for out in
            engine.generate(prompts, N4_PARAMS, use_tqdm=False)]


def no_cache(engine, prompts):
    detach(engine)
    return generate(engine, prompts)


def document(engine, num_blocks, salt):
    text = (f"Report {salt}: the committee reviewed budgets, schedules, and "
            "staffing for the new library, the bridge repairs, and the park. ") * 120
    return engine.tokenizer.encode(text)[: num_blocks * BLOCK]


def question(engine, text):
    return engine.tokenizer.encode(" Question: " + text + " Answer:")


@pytest.mark.gpu
def test_n4_a_batch_sharing_a_prefix_matches_no_cache(llm):
    doc = document(llm, 3, "batch")
    prompts = [doc + question(llm, "what was reviewed?"),
               doc + question(llm, "which repairs are planned?")]
    reference = no_cache(llm, prompts)
    external = NanoVLLMExternalCache(real_layout(llm), MemoryKVCache())
    attach(llm, external)
    generate(llm, [prompts[0]])  # stores the shared document

    manager = empty_l1(llm, external)  # empty L1: both requests restore
    assert generate(llm, prompts) == reference
    assert manager.stats.l2_hit_blocks == 3
    # The second request of the batch found the blocks the first one
    # restored, in L1 (or both restored them in the same step).
    assert manager.stats.l1_hit_blocks + manager.stats.l2_hit_blocks >= 3


@pytest.mark.gpu
def test_n4_chunked_prefill_after_a_restore_matches_no_cache(llm):
    token_ids = document(llm, 5, "chunked") + question(llm, "summarize")
    reference = no_cache(llm, [token_ids])
    external = NanoVLLMExternalCache(real_layout(llm), MemoryKVCache())
    attach(llm, external)
    generate(llm, [token_ids[: 2 * BLOCK + 5]])  # L2 gets the first 2 blocks

    manager = empty_l1(llm, external)
    budget = llm.scheduler.max_num_batched_tokens
    llm.scheduler.max_num_batched_tokens = 300  # tail prefilled in chunks
    try:
        assert generate(llm, [token_ids]) == reference
    finally:
        llm.scheduler.max_num_batched_tokens = budget
    assert manager.stats.l2_hit_blocks == 2
    assert manager.stats.prefill_tokens == len(token_ids) - 2 * BLOCK


@pytest.mark.gpu
def test_n4_a_capacity_limited_l2_restores_what_it_kept(llm):
    token_ids = document(llm, 4, "capacity") + question(llm, "why?")
    reference = no_cache(llm, [token_ids])
    external = NanoVLLMExternalCache(real_layout(llm), MemoryKVCache(max_entries=2))
    attach(llm, external)
    generate(llm, [token_ids])

    manager = empty_l1(llm, external)
    assert generate(llm, [token_ids]) == reference
    assert manager.stats.l2_hit_blocks == 2


@pytest.mark.gpu
def test_n4_disk_l2_survives_a_restart_and_matches_no_cache(llm, tmp_path):
    token_ids = document(llm, 3, "restart") + question(llm, "when?")
    reference = no_cache(llm, [token_ids])
    layout = real_layout(llm)
    attach(llm, NanoVLLMExternalCache(
        layout, DiskKVCache(tmp_path, serializer=TensorSerializer())))
    generate(llm, [token_ids])

    # A restart: new storage objects over the same directory, empty L1.
    reopened = DiskKVCache(tmp_path, serializer=TensorSerializer())
    assert len(reopened) == 3
    manager = empty_l1(llm, NanoVLLMExternalCache(layout, reopened))
    assert generate(llm, [token_ids]) == reference
    assert manager.stats.l2_hit_blocks == 3


@pytest.mark.gpu
def test_n4_l2_restore_and_l1_hit_give_identical_tokens(llm):
    """Restored blocks are bit-exact copies, so L2 and L1 must agree exactly."""
    token_ids = document(llm, 3, "exact") + question(llm, "who?")
    reference = no_cache(llm, [token_ids])
    external = NanoVLLMExternalCache(real_layout(llm), MemoryKVCache())
    attach(llm, external)
    generate(llm, [token_ids])

    empty_l1(llm, external)
    restored = generate(llm, [token_ids])
    from_l1 = generate(llm, [token_ids])  # same manager: L1 now holds them

    assert restored == from_l1 == reference
