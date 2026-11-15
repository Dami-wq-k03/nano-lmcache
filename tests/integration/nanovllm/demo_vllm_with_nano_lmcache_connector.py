"""Compare vLLM TTFT with no cache, APC, and external KV after restart.

Run in the pinned vLLM 0.33.0 CUDA environment with cached model weights:

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
        uv run python examples/demo_vllm_with_nano_lmcache_connector.py

The demo runs four engines sequentially:

1. no cache: every request recomputes the document;
2. APC only: vLLM's GPU prefix cache is warm within one engine;
3. external before restart: the connector stores the document in a
   disk-backed RemoteCacheServer;
4. external after restart: a new server process view and a new engine read
   the same disk directory, so the document KV survives the restart.

Each setup prints TTFT, wall time, and cached tokens; slowdowns appear in
the same table as speedups.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from nano_lmcache.storage.coalesced import CoalescedDiskKVCache
from nano_lmcache.storage.remote import RemoteCacheServer

# Repeated 44 times plus the closing sentence, this text encodes to exactly
# DOCUMENT_TOKENS with the pinned Qwen3 tokenizer.
DOCUMENT = (
    "The city council reviewed the transit budget, the library expansion, "
    "the riverfront plan, and the school maintenance schedule. "
) * 44 + "The council also approved the new parking plan in the spring."
QUESTIONS = (
    "What did the council decide about bus routes?",
    "How much did the library receive?",
    "When is the next council meeting?",
)
DOCUMENT_TOKENS = 1024
CACHE_BYTES = 2 << 30
MAX_MODEL_LEN = 2048


class _CacheServer:
    """Disk-backed remote cache whose contents survive a server restart."""

    def __init__(self, directory: Path) -> None:
        self.disk = CoalescedDiskKVCache(directory, max_bytes=CACHE_BYTES)
        self.server = RemoteCacheServer(storage=self.disk)
        self.server.start()

    @property
    def address(self) -> tuple[str, int]:
        return self.server.address

    @property
    def reads(self) -> int:
        return self.disk.read_ops

    def close(self) -> None:
        self.server.close()
        self.disk.close()


def _warmup(llm: Any, tokenizer: Any) -> None:
    from vllm import SamplingParams

    prompt = tokenizer.encode("A different warmup request with a unique prefix.")
    llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )


def _run_requests(
    llm: Any, prompts: dict[str, list[int]], params: Any
) -> list[dict[str, Any]]:
    rows = []
    for question, token_ids in prompts.items():
        begin = time.perf_counter()
        (output,) = llm.generate(
            [{"prompt_token_ids": token_ids}], params, use_tqdm=False
        )
        wall_ms = (time.perf_counter() - begin) * 1000
        metrics = output.metrics
        rows.append(
            {
                "question": question,
                "prompt_tokens": len(token_ids),
                "cached_tokens": output.num_cached_tokens,
                "reused_tokens": output.num_cached_tokens,
                "ttft_ms": (
                    metrics.first_token_latency * 1000
                    if metrics is not None
                    else None
                ),
                "wall_ms": wall_ms,
                "finish_reason": output.outputs[0].finish_reason,
                "token_ids": list(output.outputs[0].token_ids),
            }
        )
    return rows


def run_vllm_with_no_cache(
    args: argparse.Namespace,
    prompts: dict[str, list[int]],
    tokenizer: Any,
    params: Any,
) -> None:
    """Run every prompt in a fresh vLLM engine with no prefix cache."""
    from vllm import LLM

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        generation_config="vllm",
    )
    try:
        _warmup(llm, tokenizer)
        _print_rows("no_cache", _run_requests(llm, prompts, params))
    finally:
        llm.llm_engine.engine_core.shutdown()


def run_vllm_with_apc(
    args: argparse.Namespace,
    prompts: dict[str, list[int]],
    tokenizer: Any,
    params: Any,
) -> None:
    """Run every prompt with vLLM's GPU prefix cache enabled."""
    from vllm import LLM

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=True,
        generation_config="vllm",
    )
    try:
        _warmup(llm, tokenizer)
        _print_rows("apc", _run_requests(llm, prompts, params))
    finally:
        llm.llm_engine.engine_core.shutdown()


def run_vllm_with_nano_lmcache_connector(
    args: argparse.Namespace,
    prompts: dict[str, list[int]],
    tokenizer: Any,
    params: Any,
    server: _CacheServer,
    setup: str,
) -> None:
    """Run every prompt with nano-lmcache as the external KV cache."""
    from vllm import LLM
    from vllm.config import KVTransferConfig

    host, port = server.address
    transfer = KVTransferConfig(
        kv_connector="NanoLMCacheConnector",
        kv_connector_module_path=(
            "nano_lmcache.integrations.vllm.nano_lmcache_connector"
        ),
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "remote_cache": {"host": host, "port": port},
        },
    )
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
        kv_transfer_config=transfer,
        generation_config="vllm",
    )
    try:
        _warmup(llm, tokenizer)
        _print_rows(setup, _run_requests(llm, prompts, params))
    finally:
        llm.llm_engine.engine_core.shutdown()


def _print_rows(setup: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        ttft = row["ttft_ms"]
        ttft_text = f"{ttft:.2f}" if ttft is not None else "-"
        print(
            f"{setup:<22} {row['question']:<3} "
            f"cached_tokens={row['cached_tokens']:<5} "
            f"reused_tokens={row['reused_tokens']:<5} "
            f"ttft_ms={ttft_text:<8} "
            f"wall_ms={row['wall_ms']:.2f}"
        )


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    shutil.rmtree(args.cache_dir, ignore_errors=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    document = tokenizer.encode(DOCUMENT)
    prompts = {
        f"Q{index}": document + tokenizer.encode(question)
        for index, question in enumerate(QUESTIONS, start=1)
    }

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
    )

    run_vllm_with_no_cache(args, prompts, tokenizer, params)
    run_vllm_with_apc(args, prompts, tokenizer, params)

    server = _CacheServer(cache_dir)
    try:
        run_vllm_with_nano_lmcache_connector(
            args, prompts, tokenizer, params, server, "external_pre_restart"
        )
    finally:
        server.close()

    server = _CacheServer(cache_dir)
    try:
        run_vllm_with_nano_lmcache_connector(
            args, prompts, tokenizer, params, server, "external_post_restart"
        )
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--revision",
        default="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir())
        / "demo-vllm-with-nano-lmcache-connector",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
