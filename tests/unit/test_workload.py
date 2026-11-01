import json

import pytest

from nano_lmcache.datasets import WorkloadRequest, generate_synthetic
from nano_lmcache.datasets.adapters.freeinference import (
    FreeInferenceAdapter,
    iter_freeinference_jsonl,
)


def test_workload_request_keeps_token_ids_separate_from_prefix_blocks():
    request = WorkloadRequest(
        request_id="r-0",
        session_id="s-0",
        arrival_time_ms=12,
        model_id="model-a",
        token_ids=(101, 205),
        prefix_blocks=(77,),
        prompt_tokens=2,
    )

    assert request.prompt_tokens == 2
    assert request.token_ids == (101, 205)
    assert request.prefix_blocks == (77,)
    assert request.to_dict()["token_ids"] == [101, 205]


def test_workload_request_rejects_token_count_mismatch():
    with pytest.raises(ValueError, match="prompt_tokens"):
        WorkloadRequest(
            request_id="r",
            session_id="s",
            arrival_time_ms=0,
            model_id="m",
            prompt_tokens=3,
            token_ids=(1, 2),
        )


@pytest.mark.parametrize(
    "name, expected_count",
    [
        ("cold_cache", 1),
        ("full_prefix_hit", 2),
        ("partial_prefix_hit", 2),
        ("prefix_branching", 3),
        ("lru_eviction", 4),
        ("incomplete_chunk", 1),
    ],
)
def test_synthetic_scenarios_are_deterministic(name, expected_count):
    first = generate_synthetic(name)
    second = generate_synthetic(name)

    assert len(first) == expected_count
    assert first == second
    assert all(item.token_ids is not None for item in first)


def test_freeinference_adapter_preserves_block_and_observed_cache_semantics():
    session = {
        "session_id": "session-a",
        "harness": "demo",
        "requests": [
            {
                "step": 0,
                "turn_index": 0,
                "start_ms": 100,
                "model": "model-a",
                "provider": "B",
                "provider_tokens": {
                    "prompt": 32,
                    "completion": 4,
                    "cached": 16,
                },
                "block_ids": [10, 11],
            }
        ],
    }

    request = FreeInferenceAdapter().adapt_session(session)[0]

    assert request.request_id == "session-a:0"
    assert request.prompt_tokens == 32
    assert request.token_ids is None
    assert request.prefix_blocks == (10, 11)
    assert request.observed_cached_tokens == 16
    assert request.metadata["block_tokens"] == 16


def test_freeinference_adapter_includes_nested_sessions_and_can_stream_jsonl(tmp_path):
    session = {
        "session_id": "parent",
        "requests": [
            {
                "step": 0,
                "start_ms": 100,
                "model": "model-a",
                "provider_tokens": {"prompt": 2, "completion": 1},
                "tool_calls": [
                    {
                        "subagent": {
                            "session_id": "child",
                            "requests": [
                                {
                                    "step": 0,
                                    "start_ms": 110,
                                    "model": "model-a",
                                    "provider_tokens": {
                                        "prompt": 4,
                                        "completion": 1,
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(session) + "\n", encoding="utf-8")

    requests = list(iter_freeinference_jsonl(path))
    assert [item.request_id for item in requests] == ["parent:0", "child:0"]
    assert list(iter_freeinference_jsonl(path, limit=1))[0].request_id == "parent:0"
    assert len(list(iter_freeinference_jsonl(path, include_nested_sessions=False))) == 1
