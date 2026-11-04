from itertools import pairwise

import pytest

from nano_lmcache.disaggregation import (
    Request,
    ToyModel,
    decode,
    generate,
    prefill,
    timeline,
)

PROMPT = list(range(50, 61))


@pytest.mark.parametrize("max_new_tokens", [1, 2, 9])
def test_disaggregated_matches_colocated(max_new_tokens):
    channel: dict[str, bytes] = {}
    prefill(ToyModel(), "r", PROMPT, channel)

    tokens = decode(ToyModel(), "r", max_new_tokens, channel)

    assert tokens == generate(ToyModel(), PROMPT, max_new_tokens)


def test_decode_never_recomputes_the_prompt():
    channel: dict[str, bytes] = {}
    prefill(ToyModel(), "r", PROMPT, channel)
    decoder = ToyModel()

    decode(decoder, "r", 6, channel)

    # 6 tokens need 5 decode steps; the first token came from prefill.
    assert decoder.computed == 5


def test_next_token_reads_every_kv_entry():
    model = ToyModel()
    kv: list[list[int]] = []
    model.extend(kv, PROMPT)
    altered = [[kv[0][0] + 1, kv[0][1]], *kv[1:]]

    assert model.next_token(altered) != model.next_token(kv)


def test_handoff_is_claimed_once():
    channel: dict[str, bytes] = {}
    prefill(ToyModel(), "r", PROMPT, channel)
    decode(ToyModel(), "r", 3, channel)

    with pytest.raises(KeyError):
        decode(ToyModel(), "r", 3, channel)


def test_corrupt_bytes_are_rejected():
    channel: dict[str, bytes] = {}
    prefill(ToyModel(), "r", PROMPT, channel)
    channel["r"] = channel["r"][:-1] + b"?"

    with pytest.raises(ValueError, match="checksum"):
        decode(ToyModel(), "r", 3, channel)
    assert channel == {}


def test_kv_from_another_model_is_rejected():
    channel: dict[str, bytes] = {}
    prefill(ToyModel(seed=1), "r", PROMPT, channel)

    with pytest.raises(ValueError, match="different model"):
        decode(ToyModel(seed=2), "r", 3, channel)


CHAT = Request("chat", arrival_ms=0, prompt_tokens=200, output_tokens=20)
LONG = Request("long", arrival_ms=150, prompt_tokens=4000, output_tokens=5)


def max_gap(times):
    return max((b - a for a, b in pairwise(times)), default=0.0)


def test_long_prefill_stalls_colocated_decode():
    times = timeline([CHAT, LONG], disaggregated=False)

    assert max_gap(times["chat"]) == 2020  # waits out the 2000 ms prefill


def test_disaggregation_removes_stall_but_adds_transfer():
    colocated = timeline([CHAT, LONG], disaggregated=False)
    split = timeline([CHAT, LONG], disaggregated=True)

    assert max_gap(split["chat"]) == 20
    assert split["long"][0] > colocated["long"][0]  # TTFT includes KV transfer


def test_every_request_gets_all_tokens_in_order():
    requests = [Request(f"r{i}", i * 7.0, 50 + 30 * i, 1 + i % 4) for i in range(12)]
    for disaggregated in (False, True):
        times = timeline(requests, disaggregated=disaggregated)
        for r in requests:
            assert len(times[r.id]) == r.output_tokens
            assert times[r.id] == sorted(times[r.id])
            assert times[r.id][0] > r.arrival_ms
