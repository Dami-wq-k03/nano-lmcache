"""Validate vLLM generation and worker-saved KV against Transformers.

Requires vLLM 0.33.0, a CUDA GPU and cached Qwen/Qwen3-0.6B weights:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -s \
        tests/integration/vllm/test_vllm_with_nano_lmcache_connector.py

Generation tests cover vLLM's automatic prefix cache (APC); scheduler-side
external storage remains empty while workers save into separate CPU caches.
KV tests export those actual stored chunks and compare them with Transformers
using the same pinned weights, dtype, token IDs and positions. Set
NANO_LMCACHE_V3_REPORT_DIR to retain JSON comparison reports. Generation with
positive external hits still requires the worker load path.
"""

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.vllm

MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16
PROMPT_TOKENS = 6 * BLOCK_SIZE + 1
MAX_TOKENS = 8
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ATOL = 1e-3
RTOL = 1e-3
MAX_RELATIVE_RMS = 1e-3


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
        revision=REVISION,
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
    """Worker saves preserve generation; scheduler lookup does not count APC twice."""
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
    assert connected == baseline
    expected_local_hits = [0, 6 * BLOCK_SIZE if enable_prefix_caching else 0, 0]
    for results in (baseline, connected):
        assert [result.cached_tokens for result in results] == expected_local_hits
        assert results[0].token_ids == results[1].token_ids == results[2].token_ids
        assert all(result.finish_reason == "length" for result in results)
    print(
        "VLLM_CONNECTOR_COMPARISON "
        + json.dumps(
            {
                "model": MODEL,
                "enable_prefix_caching": enable_prefix_caching,
                "external_cache_state": "scheduler_empty_worker_populated",
                "prompt_token_ids": baseline_prompt,
                "phases": ["cold", "warm", "reset"],
                "baseline": [asdict(result) for result in baseline],
                "connected": [asdict(result) for result in connected],
            }
        )
    )


@pytest.fixture(scope="module")
def hf_kv_reference():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nano_lmcache.tensor.hf import cache_to_layers
    from nano_lmcache.tensor.kv_chunks import split_kv

    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    token_ids = tokenizer.encode(
        "The city council discussed new bus routes and the library budget. " * 40
    )[:PROMPT_TOKENS]
    assert len(token_ids) == PROMPT_TOKENS
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    model = (
        AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .to("cuda")
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([token_ids], device="cuda"),
                position_ids=torch.arange(PROMPT_TOKENS, device="cuda").unsqueeze(0),
                use_cache=True,
            )
        chunks = [
            chunk.cpu()
            for chunk in split_kv(cache_to_layers(output.past_key_values), BLOCK_SIZE)
        ]
        expected_layers = model.config.num_hidden_layers
        del output
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision(previous_precision)
    return snapshot, token_ids, chunks, expected_layers


@pytest.mark.parametrize(
    "chunked", [False, True], ids=["full-prefill", "chunked-prefill"]
)
def test_saved_kv_matches_transformers(hf_kv_reference, monkeypatch, tmp_path, chunked):
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    snapshot, token_ids, expected_chunks, num_layers = hf_kv_reference
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")

    llm = LLM(
        model=snapshot,
        tensor_parallel_size=1,
        dtype="float32",
        attention_config={"backend": "TRITON_ATTN"},
        seed=0,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=32 if chunked else 512,
        enable_chunked_prefill=True,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="NanoLMCacheConnector",
            kv_connector_module_path=(
                "nano_lmcache.integrations.vllm.nano_lmcache_connector"
            ),
            kv_role="kv_both",
        ),
        generation_config="vllm",
    )


    def observe_plans(worker):
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        worker.nano_save_plans = []
        original = connector.bind_connector_metadata

        def bind(metadata):
            original(metadata)
            for request in metadata.requests:
                worker.nano_save_plans.append(
                    {
                        "step_id": metadata.step_id,
                        "request_id": request.request_id,
                        "tokens": len(request.token_ids),
                        "slots": list(request.slot_mapping),
                    }
                )

        connector.bind_connector_metadata = bind

    def export_saved_chunks(worker, prompt, path):
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        result = connector._nano_lmcache_engine.lookup(prompt)
        torch.save(
            {
                "chunks": list(result.values),
                "keys": list(result.keys),
                "matched_tokens": result.matched_tokens,
                "layer_names": connector._layer_names,
                "plans": worker.nano_save_plans,
                "namespace": connector._nano_lmcache_engine._namespace,
            },
            path,
        )
        return result.matched_tokens

    export_path = tmp_path / "saved-kv.pt"
    try:
        llm.collective_rpc(observe_plans)
        (generated,) = llm.generate(
            [{"prompt_token_ids": token_ids}],
            SamplingParams(temperature=0, max_tokens=3, seed=0, ignore_eos=True),
            use_tqdm=False,
        )
        assert generated.finished
        assert generated.prompt_token_ids == token_ids
        assert generated.num_cached_tokens == 0
        assert llm.collective_rpc(
            export_saved_chunks, args=(token_ids, str(export_path))
        ) == [96]
    finally:
        llm.llm_engine.engine_core.shutdown()

    saved = torch.load(export_path, map_location="cpu", weights_only=True)
    assert len(saved["chunks"]) == len(expected_chunks) == 6
    assert len(saved["layer_names"]) == num_layers
    assert len(set(saved["keys"])) == 6
    planned_lengths = [plan["tokens"] for plan in saved["plans"]]
    assert planned_lengths == ([32, 64, 96, 96] if chunked else [96])
    assert saved["namespace"].startswith("vllm-kv/v1/")

    first, expected_first = saved["chunks"][0], expected_chunks[0]
    assert not torch.allclose(
        first.roll(1, dims=3), expected_first, atol=ATOL, rtol=RTOL
    )
    assert not torch.allclose(first.flip(1), expected_first, atol=ATOL, rtol=RTOL)

    errors = []
    failures = []
    for chunk_index, (actual, expected) in enumerate(
        zip(saved["chunks"], expected_chunks, strict=True)
    ):
        assert actual.dtype == expected.dtype == torch.float32
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        for layer in range(num_layers):
            for kv_index, kind in enumerate(("K", "V")):
                a = actual[layer, kv_index].float()
                b = expected[layer, kv_index].float()
                diff = (a - b).abs()
                relative_rms = (
                    diff.square().mean().sqrt()
                    / b.square().mean().sqrt().clamp_min(1e-12)
                ).item()
                close = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
                entry = {
                    "chunk": chunk_index,
                    "layer": layer,
                    "kind": kind,
                    "max_abs_error": diff.max().item(),
                    "mean_abs_error": diff.mean().item(),
                    "relative_rms_error": relative_rms,
                    "allclose": close,
                }
                errors.append(entry)
                if not close or relative_rms > MAX_RELATIVE_RMS:
                    failures.append(entry)

    report = {
        "model": MODEL,
        "revision": REVISION,
        "snapshot": snapshot,
        "dtype": "float32",
        "attention_backend": "TRITON_ATTN",
        "triton_f32_default": "ieee",
        "namespace": saved["namespace"],
        "negative_controls": {
            "shifted_tokens_rejected": True,
            "swapped_kv_rejected": True,
        },
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_token_ids": token_ids,
        "position_ids": list(range(PROMPT_TOKENS)),
        "chunk_size": BLOCK_SIZE,
        "chunked_prefill": chunked,
        "saved_tokens": saved["matched_tokens"],
        "saved_chunk_shapes": [list(chunk.shape) for chunk in saved["chunks"]],
        "layer_names": saved["layer_names"],
        "plans": saved["plans"],
        "generated_token_ids": list(generated.outputs[0].token_ids),
        "atol": ATOL,
        "rtol": RTOL,
        "max_relative_rms": MAX_RELATIVE_RMS,
        "max_abs_error": max(entry["max_abs_error"] for entry in errors),
        "max_relative_rms_error": max(entry["relative_rms_error"] for entry in errors),
        "comparisons": errors,
        "failures": failures,
        "passed": not failures,
    }
    report_dir = Path(os.environ.get("NANO_LMCACHE_V3_REPORT_DIR", str(tmp_path)))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"vllm-v3-{'chunked' if chunked else 'full'}-kv.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"VLLM_V3_KV_REPORT {report_path}")
    assert not failures, (
        f"{len(failures)} layer/KV comparisons failed; "
        f"max abs={report['max_abs_error']}, "
        f"max relative RMS={report['max_relative_rms_error']}; "
        f"details: {report_path}"
    )
