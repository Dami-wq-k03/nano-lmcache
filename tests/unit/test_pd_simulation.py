import pytest

from nano_lmcache.disaggregation import (
    CostModel,
    SimRequest,
    simulate_colocated,
    simulate_disaggregated,
)

COST = CostModel(
    prefill_ms_per_token=0.5, decode_ms_per_step=20, transfer_ms_per_token=0.05
)
CHAT = SimRequest("chat", arrival_ms=0, prompt_tokens=200, output_tokens=20)
LONG = SimRequest("long", arrival_ms=150, prompt_tokens=4000, output_tokens=5)


def test_single_request_colocated_timeline():
    timing = simulate_colocated([SimRequest("a", 10, 100, 3)], COST)["a"]

    assert timing.token_times_ms == (60, 80, 100)
    assert timing.ttft_ms == 50
    assert timing.max_itl_ms == 20


def test_single_request_disaggregated_pays_transfer():
    timing = simulate_disaggregated([SimRequest("a", 10, 100, 3)], COST)["a"]

    assert timing.token_times_ms == (65, 85, 105)
    assert timing.ttft_ms == 55


def test_long_prefill_stalls_colocated_decode():
    colocated = simulate_colocated([CHAT, LONG], COST)

    # The chat request's next token waits for the whole 2000 ms prefill.
    assert colocated["chat"].max_itl_ms == pytest.approx(2020)


def test_disaggregation_removes_the_stall_but_adds_transfer():
    colocated = simulate_colocated([CHAT, LONG], COST)
    disaggregated = simulate_disaggregated([CHAT, LONG], COST)

    assert disaggregated["chat"].max_itl_ms == COST.decode_ms_per_step
    assert disaggregated["chat"].finish_ms < colocated["chat"].finish_ms
    assert disaggregated["long"].ttft_ms > colocated["long"].ttft_ms


def test_every_request_gets_all_tokens_in_order():
    requests = [
        SimRequest(
            f"r{i}",
            arrival_ms=i * 7.0,
            prompt_tokens=50 + 30 * i,
            output_tokens=1 + i % 4,
        )
        for i in range(12)
    ]
    for simulate in (simulate_colocated, simulate_disaggregated):
        timings = simulate(requests, COST)
        for request in requests:
            times = timings[request.request_id].token_times_ms
            assert len(times) == request.output_tokens
            assert list(times) == sorted(times)
            assert times[0] > request.arrival_ms


def test_one_token_request_has_zero_itl():
    timing = simulate_colocated([SimRequest("a", 0, 10, 1)], COST)["a"]

    assert timing.max_itl_ms == 0


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        simulate_colocated([CHAT, CHAT], COST)


def test_invalid_request():
    with pytest.raises(ValueError):
        SimRequest("a", 0, 0, 1)
