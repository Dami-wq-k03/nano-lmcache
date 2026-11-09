"""Validate real vLLM generation with scheduler lookup and empty external storage.

Requires vLLM 0.33.0, a CUDA GPU and cached Qwen/Qwen3-0.6B weights:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -s \
        tests/integration/vllm/test_vllm_with_nano_lmcache_connector.py

KV reuse here means vLLM's automatic prefix cache (APC). The connector queries
external storage, which stays empty because worker saves are not implemented.
Populated-storage accounting is covered by the direct connector unit tests;
generation with positive external hits requires the worker load path.
"""

import json
from dataclasses import asdict, dataclass

import pytest

pytestmark = pytest.mark.vllm

MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16
PROMPT_TOKENS = 6 * BLOCK_SIZE + 1
MAX_TOKENS = 8


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    cached_tokens: int
    finish_reason: str | None


def _generate_cold_warm_reset(
    *, use_connector: bool, enable_prefix_caching: bool
) -> tuple[list[int], list[Generation]]:
    """Run identical requests in a fresh engine, then release its GPU resources."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    transfer_config = (
        KVTransferConfig(
            kv_connector="NanoLMCacheConnector",
            kv_connector_module_path=(
                "nano_lmcache.integrations.vllm.nano_lmcache_connector"
            ),
            kv_role="kv_both",
        )
        if use_connector
        else None
    )
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=0,
        max_model_len=512,
        max_num_seqs=1,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=enable_prefix_caching,
        kv_transfer_config=transfer_config,
        generation_config="vllm",
    )
    try:
        actual_config = llm.llm_engine.vllm_config.kv_transfer_config
        if use_connector:
            assert actual_config is not None
            assert actual_config.kv_connector == "NanoLMCacheConnector"
            assert (
                actual_config.kv_connector_module_path
                == transfer_config.kv_connector_module_path
            )
        else:
            assert actual_config is None

        # More than one block is needed for meaningful APC hits. Leave a tail
        # token so the warm request can reuse exactly six complete blocks.
        prompt_ids = llm.get_tokenizer().encode(
            "The city council discussed new bus routes and the library budget. " * 40
        )[:PROMPT_TOKENS]
        assert len(prompt_ids) == PROMPT_TOKENS
        results = []
        for phase in ("cold", "warm", "reset"):
            if phase == "reset":
                assert llm.reset_prefix_cache()
            (output,) = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(
                    temperature=0.0, max_tokens=MAX_TOKENS, seed=0, ignore_eos=True
                ),
                use_tqdm=False,
            )
            assert output.finished
            assert output.prompt_token_ids == prompt_ids
            (completion,) = output.outputs
            assert len(completion.token_ids) == MAX_TOKENS
            assert output.num_cached_tokens is not None
            results.append(
                Generation(
                    tuple(completion.token_ids),
                    output.num_cached_tokens,
                    completion.finish_reason,
                )
            )
        return prompt_ids, results
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize("enable_prefix_caching", [False, True], ids=["no-apc", "apc"])
def test_vllm_generates_with_nano_lmcache_connector(monkeypatch, enable_prefix_caching):
    """Empty external storage preserves generation and does not double-count APC."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pytest.importorskip("vllm.config")

    baseline_prompt, baseline = _generate_cold_warm_reset(
        use_connector=False, enable_prefix_caching=enable_prefix_caching
    )
    connector_prompt, connected = _generate_cold_warm_reset(
        use_connector=True, enable_prefix_caching=enable_prefix_caching
    )

    assert connector_prompt == baseline_prompt
    # Compare exact token IDs, cache hit counts and finish reasons in each phase.
    assert connected == baseline
    expected_local_hits = [0, 6 * BLOCK_SIZE if enable_prefix_caching else 0, 0]
    for results in (baseline, connected):
        assert [result.cached_tokens for result in results] == expected_local_hits
        assert results[0].token_ids == results[1].token_ids == results[2].token_ids
        assert all(result.finish_reason == "length" for result in results)
    # Keep the actual IDs and hit counts alongside lifecycle logs for the
    # documented real-request validation (visible with pytest -s).
    print(
        "VLLM_CONNECTOR_COMPARISON "
        + json.dumps(
            {
                "model": MODEL,
                "enable_prefix_caching": enable_prefix_caching,
                "external_cache_state": "empty",
                "prompt_token_ids": baseline_prompt,
                "phases": ["cold", "warm", "reset"],
                "baseline": [asdict(result) for result in baseline],
                "connected": [asdict(result) for result in connected],
            }
        )
    )
