import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.disaggregation import (
    ConfigMismatchError,
    CorruptHandoffError,
    DecodeWorker,
    DirectoryTransport,
    HandoffNotFoundError,
    InMemoryTransport,
    PrefillWorker,
    ToyLanguageModel,
    ToyModelConfig,
    generate_colocated,
)

CONFIG = ToyModelConfig(seed=3)
PROMPT = list(range(50, 61))  # 11 tokens: not a multiple of the block size


def workers(transport, **prefill_options):
    return (
        PrefillWorker(ToyLanguageModel(CONFIG), transport, **prefill_options),
        DecodeWorker(ToyLanguageModel(CONFIG), transport),
    )


@pytest.mark.parametrize("make_transport", ["memory", "directory"])
@pytest.mark.parametrize("max_new_tokens", [1, 2, 9])
def test_disaggregated_matches_colocated(tmp_path, make_transport, max_new_tokens):
    transport = (
        InMemoryTransport()
        if make_transport == "memory"
        else DirectoryTransport(tmp_path)
    )
    prefill, decode = workers(transport, block_size=4)

    sent = prefill.prefill("r", PROMPT)
    result = decode.decode("r", max_new_tokens=max_new_tokens)

    expected = generate_colocated(ToyLanguageModel(CONFIG), PROMPT, max_new_tokens)
    assert list(result.tokens) == expected
    assert sent.first_token == expected[0]
    assert sent.num_blocks == 3


def test_decode_worker_never_recomputes_the_prompt():
    prefill, decode = workers(InMemoryTransport())

    prefill.prefill("r", PROMPT)
    result = decode.decode("r", max_new_tokens=6)

    assert prefill.model.computed_tokens == len(PROMPT)
    assert result.imported_tokens == len(PROMPT)
    # n tokens need n - 1 decode steps; the first came from prefill.
    assert decode.model.computed_tokens == result.computed_tokens == 5


def test_toy_model_output_depends_on_every_kv_entry():
    model = ToyLanguageModel(CONFIG)
    kv, first = model.prefill(PROMPT)

    altered = list(kv)
    altered[0] = tuple(value + 1 for value in altered[0])

    assert model.next_token(altered) != first


def test_kv_depends_on_model_weights():
    a, _ = ToyLanguageModel(ToyModelConfig(seed=1)).prefill(PROMPT)
    b, _ = ToyLanguageModel(ToyModelConfig(seed=2)).prefill(PROMPT)

    assert a != b


def test_decode_rejects_kv_from_another_model():
    transport = InMemoryTransport()
    PrefillWorker(ToyLanguageModel(ToyModelConfig(seed=1)), transport).prefill(
        "r", PROMPT
    )
    decode = DecodeWorker(ToyLanguageModel(ToyModelConfig(seed=2)), transport)

    with pytest.raises(ConfigMismatchError):
        decode.decode("r", max_new_tokens=3)
    assert decode.model.computed_tokens == 0
    # The rejected handoff is gone: the request must be prefilled again.
    assert transport.pending_ids() == []


def test_decode_rejects_corrupt_transfer(tmp_path):
    transport = DirectoryTransport(tmp_path)
    prefill, decode = workers(transport)
    prefill.prefill("r", PROMPT)
    path = transport.path_for("r")
    data = bytearray(path.read_bytes())
    data[-4] ^= 0x01
    path.write_bytes(bytes(data))

    with pytest.raises(CorruptHandoffError):
        decode.decode("r", max_new_tokens=3)
    assert decode.model.computed_tokens == 0


def test_decode_after_cancel_finds_nothing():
    transport = InMemoryTransport()
    prefill, decode = workers(transport)
    prefill.prefill("r", PROMPT)

    assert transport.discard("r")
    with pytest.raises(HandoffNotFoundError):
        decode.decode("r", max_new_tokens=3)


def test_second_decode_worker_cannot_claim_same_request():
    transport = InMemoryTransport()
    prefill, decode = workers(transport)
    prefill.prefill("r", PROMPT)
    decode.decode("r", max_new_tokens=3)

    with pytest.raises(HandoffNotFoundError):
        DecodeWorker(ToyLanguageModel(CONFIG), transport).decode("r", max_new_tokens=3)


def test_prefill_reuses_prefix_cache_and_stays_exact():
    transport = InMemoryTransport()
    cache = ChunkedPrefixCache(chunk_size=4, namespace=CONFIG.fingerprint)
    prefill, decode = workers(transport, prefix_cache=cache)
    document = list(range(200, 212))  # three complete chunks

    first = prefill.prefill("a", document + [1, 2])
    decode.decode("a", max_new_tokens=2)
    second = prefill.prefill("b", document + [3, 4, 5])
    result = decode.decode("b", max_new_tokens=5)

    assert (first.reused_tokens, first.computed_tokens) == (0, 14)
    assert (second.reused_tokens, second.computed_tokens) == (12, 3)
    expected = generate_colocated(ToyLanguageModel(CONFIG), document + [3, 4, 5], 5)
    assert list(result.tokens) == expected


def test_invalid_arguments():
    prefill, decode = workers(InMemoryTransport())

    with pytest.raises(ValueError):
        prefill.prefill("r", [])
    with pytest.raises(ValueError):
        decode.decode("r", max_new_tokens=0)
    with pytest.raises(ValueError):
        PrefillWorker(ToyLanguageModel(CONFIG), InMemoryTransport(), block_size=0)
