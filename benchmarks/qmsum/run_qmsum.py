"""Replay frozen QMSum token IDs with empty L1 and a per-cohort L2 lifetime."""

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Only our trusted source directories are added, never the dataset directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def read_workload(path, max_model_len, max_output):
    groups = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if len(row["input_ids"]) != row["prompt_tokens"]:
                raise ValueError("prompt length differs from frozen token IDs")
            if row["prompt_tokens"] + max_output > max_model_len:
                raise ValueError("full prompt + output exceeds context; no truncation")
            groups[row["meeting_id"]].append(row)
    if not groups:
        raise ValueError("empty workload")
    for rows in groups.values():
        if [row["query_index"] for row in rows] != list(range(len(rows))):
            raise ValueError("meeting queries must be unique, contiguous, and ordered")
    return list(groups.values())


def environment(args, llm):
    import torch

    root = Path(__file__).resolve().parents[2]
    code = hashlib.sha256()
    for path in sorted(
        [*root.glob("src/**/*.py"), *root.glob("benchmarks/qmsum/*.py")]
    ):
        code.update(str(path.relative_to(root)).encode())
        code.update(path.read_bytes())
    with args.workload.open() as stream:
        workload_requests = sum(1 for _ in stream)
    info = {
        "expected_records": (
            workload_requests
            * args.repeats
            * len(args.modes)
            * len(args.concurrency)
            * len(args.max_tokens)
        ),
        "source_revision": args.source_revision,
        "code_sha256": code.hexdigest(),
        "nano_vllm_revision": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "model": args.model.name,
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": (torch.cuda.get_device_properties(0).total_memory),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "flash-attn", "triton")
        },
        "settings": {
            k: v
            for k, v in vars(args).items()
            if k not in ("model", "workload", "output")
        },
        "runtime_environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "kv_cache_shape": list(llm.model_runner.kv_cache.shape),
        "dtype": str(llm.model_runner.kv_cache.dtype),
        "ttft_definition": "arrival to first observable token after step/postprocess",
        "async_implementation": "AsyncWriteBehindStorage; blocking pinned-host prepare",
    }
    conversion = args.model / "conversion.json"
    if conversion.exists():
        info["model_conversion"] = json.loads(conversion.read_text())
    info["nvidia_smi"] = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    return info


def run(args):
    import torch
    from measurement import new_storage, replay_wave
    from nanovllm import LLM, SamplingParams

    from nano_lmcache.integrations.nanovllm import KVBlockLayout, NanoVLLMExternalCache

    groups = read_workload(args.workload, args.max_model_len, max(args.max_tokens))
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("use a new, empty output directory for every invocation")
    # The pinned engine hard-codes port 2333, even at TP=1. Redirect only
    # its initialization call so independent single-GPU jobs can coexist.
    original_init = torch.distributed.init_process_group

    def init_group(backend, init_method, **kwargs):
        if init_method != "tcp://localhost:2333":
            raise ValueError("unexpected nano-vLLM rendezvous method")
        return original_init(backend, f"tcp://localhost:{args.master_port}", **kwargs)

    torch.distributed.init_process_group = init_group
    try:
        llm = LLM(
            str(args.model),
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=max(args.concurrency),
            max_num_batched_tokens=args.max_batched_tokens,
        )
    finally:
        torch.distributed.init_process_group = original_init
    layout = KVBlockLayout.from_hf_config(
        llm.model_runner.config.hf_config,
        model=args.model.name,
        revision=args.model_revision,
    )
    (args.output / "environment.json").write_text(
        json.dumps(environment(args, llm), indent=2)
    )

    def cohort(meetings, mode, output_tokens, repeat, concurrency, cohort_id):
        capacity = (
            sum(row["prompt_tokens"] // 256 for rs in meetings for row in rs) + 64
        )
        storage = None if mode == "off" else new_storage(mode, capacity)
        records, waves = [], []
        params = SamplingParams(
            temperature=1e-6, max_tokens=output_tokens, ignore_eos=True
        )
        common = dict(
            mode=mode,
            max_tokens=output_tokens,
            repeat=repeat,
            concurrency=concurrency,
            cohort=cohort_id,
        )
        start = time.perf_counter()
        try:
            for index in range(max(map(len, meetings))):
                requests = [rows[index] for rows in meetings if index < len(rows)]
                # Fresh manager for every wave, same L2 storage for the cohort.
                external = (
                    NanoVLLMExternalCache(layout, storage)
                    if storage is not None
                    else None
                )
                torch.manual_seed(0)
                rows, wave = replay_wave(
                    llm, requests, params, external, storage=storage
                )
                wave["inter_wave_gap_ms"] = (
                    (wave["arrival_monotonic"] - waves[-1]["finished_monotonic"]) * 1000
                    if waves
                    else None
                )
                for row in rows:
                    expected = (
                        row["num_cacheable_blocks"] if index and mode != "off" else 0
                    )
                    if row["l2_hit_blocks"] < expected:
                        raise RuntimeError(f"restore shortfall: {row['meeting_id']}")
                    if index == 0 and row["l2_hit_blocks"]:
                        raise RuntimeError("first query found historical L2 data")
                    row.update(
                        common,
                        wave=index,
                        cold=index == 0,
                        reused_tokens=row["l2_hit_blocks"] * 256,
                    )
                    records.append(row)
                wave.update(common, wave=index, cold=index == 0)
                before = time.perf_counter()
                if mode == "async-flush":
                    storage.flush()  # deliberately OUTSIDE request latency
                wave["outside_flush_ms"] = (time.perf_counter() - before) * 1000
                waves.append(wave)
        finally:
            if mode.startswith("async"):
                storage.close()  # final drain IS part of sustained cohort wall time
        wall_ms = (time.perf_counter() - start) * 1000
        return records, waves, {**common, "wall_ms": wall_ms, "requests": len(records)}

    try:
        with (
            (args.output / "raw_results.jsonl").open("x") as raw,
            (args.output / "waves.jsonl").open("x") as waves_file,
            (args.output / "cohorts.jsonl").open("x") as cohorts_file,
        ):
            for concurrency in args.concurrency:
                # Warm both the compute and prefix-attention paths at every
                # tested batch size. These observations never enter the results.
                for warm in range(args.warmup):
                    for mode in args.modes:
                        cohort(
                            groups[:concurrency],
                            mode,
                            max(args.max_tokens),
                            -1,
                            concurrency,
                            f"warmup-{warm}",
                        )
                for output_tokens in args.max_tokens:
                    for repeat in range(args.repeats):
                        modes = (
                            args.modes[repeat % len(args.modes) :]
                            + args.modes[: repeat % len(args.modes)]
                        )
                        for start in range(0, len(groups), concurrency):
                            for mode in modes:
                                records, waves, summary = cohort(
                                    groups[start : start + concurrency],
                                    mode,
                                    output_tokens,
                                    repeat,
                                    concurrency,
                                    start // concurrency,
                                )
                                for row in records:
                                    raw.write(
                                        json.dumps(row, ensure_ascii=False) + "\n"
                                    )
                                for wave in waves:
                                    waves_file.write(json.dumps(wave) + "\n")
                                cohorts_file.write(json.dumps(summary) + "\n")
                                raw.flush()
                                waves_file.flush()
                                cohorts_file.flush()
                                print(json.dumps(summary), flush=True)
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off", "sync", "async-flush", "async-continuous"],
        choices=["off", "sync", "async-flush", "async-continuous"],
    )
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument(
        "--co-run-label",
        default="none",
        help="record any simultaneous experiment on the other GPU",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if (
        min(
            *args.max_tokens,
            *args.concurrency,
            args.repeats,
            args.max_model_len,
            args.max_batched_tokens,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("counts must be positive (warmup may be zero)")
    run(args)


if __name__ == "__main__":
    main()
