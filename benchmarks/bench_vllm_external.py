"""Real vLLM external-prefix reuse with local APC disabled.

Run in the pinned vLLM 0.33.0 CUDA environment. Model weights must already
exist. This measures request wall time; optional engine metrics supply TTFT.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from nano_lmcache.storage.coalesced import CoalescedDiskKVCache
from nano_lmcache.storage.remote import RemoteCacheServer


class ObservedServer(RemoteCacheServer):
    """Retain successful payload reads as evidence beyond scheduler lookups."""

    def __init__(self, directory: Path | None = None) -> None:
        self.disk = (
            CoalescedDiskKVCache(directory, max_bytes=512 << 20) if directory else None
        )
        super().__init__(storage=self.disk, max_bytes=512 << 20)
        self.reads: list[str] = []
        self.writes: list[str] = []

    def drop_disk_cache(self) -> None:
        if self.disk is None:
            raise ValueError("cold disk reads require a disk backend")
        for path in self.disk.directory.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self.disk is not None:
                self.disk.close()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        result = super().handle(request)
        if request.get("op") == "get" and result.get("payload") is not None:
            self.reads.append(request["key"])
        if request.get("op") == "put" and result.get("admitted"):
            self.writes.append(request["key"])
        return result


def physical_read_bytes() -> int | None:
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def generate_case(
    args: argparse.Namespace, server: ObservedServer | None, *, local_apc: bool = False
) -> list:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    transfer = None
    if server is not None:
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
                "async_store": args.async_store,
            },
        )
    engine = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=0,
        max_model_len=args.prompt_tokens + args.output_tokens + 256,
        max_num_seqs=1,
        block_size=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=local_apc,
        kv_transfer_config=transfer,
        generation_config="vllm",
    )
    try:
        params = SamplingParams(
            temperature=0.0, max_tokens=args.output_tokens, seed=0, ignore_eos=True
        )
        tokens = engine.get_tokenizer().encode(
            "The council discussed bus routes and library funding. "
            * (args.prompt_tokens + 1)
        )[: args.prompt_tokens]
        if len(tokens) != args.prompt_tokens:
            raise RuntimeError("not enough prompt tokens")
        # Compile/warm engine paths with a different prefix; measured cold prompt
        # must still be absent from external storage.
        warm = list(reversed(tokens))
        for index in range(args.warmup):
            writes_before = len(server.writes) if server is not None else 0
            engine.generate([{"prompt_token_ids": warm}], params, use_tqdm=False)
            if server is not None and index == 0:
                # Drain the first warmup before another request can observe a
                # partial prefix. Otherwise its late PUTs can falsely satisfy
                # the measured cold request's publication counter.
                deadline = time.monotonic() + 30
                while len(server.writes) < writes_before + args.prompt_tokens // 16:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("warmup writes did not publish")
                    time.sleep(0.01)
        cases = [("cold", tokens)] + [("repeat", tokens)] * args.repeats
        for index in range(getattr(args, "store_repeats", args.repeats)):
            novel = tokens.copy()
            novel[0] = index + 1 if tokens[0] != index + 1 else 10000 + index
            cases.append(("store", novel))
        rows = []
        for phase, prompt in cases:
            if server is not None and getattr(args, "cold_disk", False):
                server.drop_disk_cache()
            physical_before = physical_read_bytes()
            before = len(server.reads) if server is not None else 0
            writes_before = len(server.writes) if server is not None else 0
            begin = time.perf_counter()
            (output,) = engine.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )
            wall_ms = (time.perf_counter() - begin) * 1000
            physical_after = physical_read_bytes()
            (completion,) = output.outputs
            reads = len(server.reads) - before if server is not None else 0
            if len(completion.token_ids) != args.output_tokens:
                raise RuntimeError("generation stopped early")
            expected_hits = (args.prompt_tokens - 1) // 16 * 16
            if phase == "repeat" and (server is not None or local_apc):
                expected_reads = expected_hits // 16 if server is not None else 0
                if reads != expected_reads:
                    raise RuntimeError(
                        f"phase={phase}: expected {expected_reads} payload reads, "
                        f"got {reads}; cached_tokens={output.num_cached_tokens}; "
                        f"writes={writes_before}"
                    )
                if output.num_cached_tokens != expected_hits:
                    raise RuntimeError(
                        f"wrong cached tokens: {output.num_cached_tokens}"
                    )
            elif reads or output.num_cached_tokens:
                raise RuntimeError("cold/no-cache request unexpectedly reused KV")
            metrics = output.metrics
            ttft = metrics.first_token_latency * 1000 if metrics is not None else None
            tpot = (
                (metrics.last_token_ts - metrics.first_token_ts)
                * 1000
                / (len(completion.token_ids) - 1)
                if metrics is not None and len(completion.token_ids) > 1
                else None
            )
            publication_start = time.perf_counter()
            if server is not None:
                # Drain remote publication outside request timing. A pending store
                # is allowed to miss; do not mislabel that race as a load failure.
                deadline = time.monotonic() + 30
                expected_writes = writes_before + (
                    args.prompt_tokens // 16 if phase != "repeat" else 0
                )
                while len(server.writes) < expected_writes:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "async writes did not publish before deadline"
                        )
                    time.sleep(0.01)
            rows.append(
                {
                    "phase": phase,
                    "wall_ms": wall_ms,
                    "server_physical_read_bytes": (
                        physical_after - physical_before
                        if physical_after is not None and physical_before is not None
                        else None
                    ),
                    "publication_wait_ms": (time.perf_counter() - publication_start)
                    * 1000,
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "cached_tokens": output.num_cached_tokens,
                    "external_payload_reads": reads,
                    "token_ids": list(completion.token_ids),
                }
            )
        return rows
    finally:
        engine.llm_engine.engine_core.shutdown()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import vllm

    if vllm.__version__ != "0.33.0":
        raise RuntimeError("this benchmark targets vLLM 0.33.0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    baseline = generate_case(args, None)
    apc = (
        generate_case(args, None, local_apc=True)
        if getattr(args, "with_apc_control", False)
        else []
    )
    with tempfile.TemporaryDirectory(dir=getattr(args, "disk_directory", None)) as temp:
        disk_directory = Path(temp) if getattr(args, "disk_directory", None) else None
        server = ObservedServer(disk_directory)
        server.start()
        try:
            connected = generate_case(args, server)
        finally:
            server.close()
    if len(baseline) != len(connected) or any(
        reference["token_ids"] != cached["token_ids"]
        for reference, cached in zip(baseline, connected, strict=True)
    ):
        raise RuntimeError("external-cache generation differs from the baseline")
    if apc and any(
        a["token_ids"] != b["token_ids"] for a, b in zip(baseline, apc, strict=True)
    ):
        raise RuntimeError("local APC generation differs from the baseline")
    return {
        "environment": {
            "git_sha": os.environ.get("NANO_LMCACHE_GIT_SHA"),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": vars(args)
        | {
            "json": str(args.json) if args.json else None,
            "disk_directory": str(args.disk_directory)
            if getattr(args, "disk_directory", None)
            else None,
        },
        "scope": (
            "Sequential external reuse; APC disabled in baseline/external; "
            "optional separate local APC control"
        ),
        "baseline": baseline,
        "local_apc": apc,
        "external": connected,
        "baseline_repeat_wall_p50_ms": statistics.median(
            row["wall_ms"] for row in baseline if row["phase"] == "repeat"
        ),
        "external_repeat_wall_p50_ms": statistics.median(
            row["wall_ms"] for row in connected if row["phase"] == "repeat"
        ),
        "outputs_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--async-store", action="store_true")
    parser.add_argument("--with-apc-control", action="store_true")
    parser.add_argument(
        "--disk-directory",
        type=Path,
        help="parent NVMe directory for isolated server storage",
    )
    parser.add_argument("--cold-disk", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt-tokens", type=int, default=257)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--store-repeats", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if (
        min(args.prompt_tokens, args.output_tokens, args.repeats, args.store_repeats)
        <= 0
        or args.warmup < 0
    ):
        parser.error("token counts/repeats must be positive; warmup nonnegative")
    if args.prompt_tokens <= 16 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("need >16 prompt tokens and memory utilization in (0,1)")
    if args.cold_disk and (
        args.disk_directory is None or not hasattr(os, "posix_fadvise")
    ):
        parser.error("--cold-disk requires --disk-directory and Linux fadvise")
    result = run(args)
    encoded = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
