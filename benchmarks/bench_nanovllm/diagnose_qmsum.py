"""Separate full-prefill/prefix numerical differences from L2 copy errors.

Diagnostic only: these instrumented generations never enter performance results.
"""

import argparse
import atexit
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def bitwise_equal(left, right):
    """Compare representations too: torch.equal alone treats +0 and -0 as equal."""
    import torch

    return (
        isinstance(right, torch.Tensor)
        and left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


def diagnose(args):
    import torch
    from measurement import new_storage
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence

    from nano_lmcache.integrations.nanovllm import (
        KVBlockLayout,
        NanoVLLMExternalCache,
        engine,
    )
    from nano_lmcache.integrations.nanovllm.kv_blocks import export_blocks

    with args.workload.open() as stream:
        rows = [json.loads(line) for line in stream]
    meeting = [r for r in rows if r["meeting_id"] == args.meeting]
    parent = next(r for r in meeting if r["query_index"] == 0)
    target = next(r for r in meeting if r["query_id"] == args.query)
    if target["query_index"] == 0:
        raise ValueError("select a subsequent query, not the priming query")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("use an empty diagnostic output directory")
    original_init = torch.distributed.init_process_group

    def init_group(backend, init_method, **kwargs):
        if init_method != "tcp://localhost:2333":
            raise ValueError("unexpected pinned-engine rendezvous method")
        return original_init(backend, f"tcp://localhost:{args.master_port}", **kwargs)

    torch.distributed.init_process_group = init_group
    try:
        llm = LLM(
            str(args.model),
            enforce_eager=True,
            tensor_parallel_size=1,
            max_num_seqs=1,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    finally:
        torch.distributed.init_process_group = original_init
    layout = KVBlockLayout.from_hf_config(
        llm.model_runner.config.hf_config,
        model=args.model.name,
        revision=args.model_revision,
    )
    params = SamplingParams(temperature=1e-6, max_tokens=1, ignore_eos=True)
    original_logits = llm.model_runner.model.compute_logits
    original_import = engine.import_blocks
    latest, checks = [], []
    captured, result = {}, {}

    def capture(hidden):
        logits = original_logits(hidden)
        # max_tokens=1: the last captured logits produce the first output token.
        latest[:] = [logits[-1].detach().float().cpu()]
        return logits

    def checked_import(cache, block_ids, payloads):
        original_import(cache, block_ids, payloads)
        actual = export_blocks(cache, block_ids)
        equal = [
            bitwise_equal(a, b.to(a.device))
            for a, b in zip(actual, payloads, strict=True)
        ]
        checks.extend(equal)
        if not all(equal):
            raise RuntimeError("imported KV differs from fetched payload")

    def generate(record):
        torch.manual_seed(0)
        output = llm.generate([record["input_ids"]], params, use_tqdm=False)[0]
        torch.cuda.synchronize()
        return output

    def save(name, output, extra):
        logits = latest[0].clone()
        captured[name] = logits
        values, ids = logits.topk(5)
        result[name] = {
            "output_ids": output["token_ids"],
            "output_text": output["text"],
            "top5": [
                {"id": i, "token": llm.tokenizer.decode([i]), "logit": v}
                for i, v in zip(ids.tolist(), values.tolist(), strict=True)
            ],
            **extra,
        }

    llm.model_runner.model.compute_logits = capture
    engine.import_blocks = checked_import
    try:
        engine.detach(llm)
        save("full-prefill", generate(target), {})
        manager = engine.detach(llm)
        generate(parent)
        hits = manager.can_allocate(Sequence(target["input_ids"], params))
        save("native-l1", generate(target), {"l1_hit_blocks": hits})
        for mode in ("sync", "async-continuous"):
            checks.clear()
            storage = new_storage(mode, 512)
            expected_cpu = {}
            original_put = storage.put

            def checked_put(
                key, value, *, parent_key=None, snapshots=expected_cpu, put=original_put
            ):
                snapshots[key] = value.detach().cpu().clone()
                put(key, value, parent_key=parent_key)

            storage.put = checked_put
            external = NanoVLLMExternalCache(layout, storage)
            try:
                engine.attach(llm, external)
                generate(parent)
                # Diagnostic-only drain: inspect completed CPU payloads, not timing.
                if mode.startswith("async"):
                    storage.flush()
                backend = storage.backend if mode.startswith("async") else storage
                cpu_equal = [
                    bitwise_equal(value, backend.backend.peek(key))
                    for key, value in expected_cpu.items()
                ]
                if not all(cpu_equal):
                    raise RuntimeError("CPU cache differs from exported GPU payload")
                manager = engine.attach(llm, external)
                output = generate(target)
                save(
                    mode,
                    output,
                    {
                        "l2_hit_blocks": manager.stats.l2_hit_blocks,
                        "kv_blocks_checked": len(checks),
                        "cpu_blocks_checked": len(cpu_equal),
                        "all_stored_cpu_payloads_bitwise_equal": bool(cpu_equal)
                        and all(cpu_equal),
                        "all_imported_kv_bitwise_equal": bool(checks) and all(checks),
                    },
                )
            finally:
                del storage.put
                if mode.startswith("async"):
                    storage.close()
        for name, logits in captured.items():
            for reference in ("full-prefill", "native-l1"):
                result[name][f"vs_{reference}_max_abs_logit_delta"] = (
                    (logits - captured[reference]).abs().max().item()
                )
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
        torch.save(captured, args.output / "logits.pt")
        (args.output / "environment.json").write_text(
            json.dumps(
                {
                    "model": args.model.name,
                    "model_revision": args.model_revision,
                    "meeting_id": args.meeting,
                    "query_id": args.query,
                    "workload_sha256": hashlib.sha256(
                        args.workload.read_bytes()
                    ).hexdigest(),
                    "script_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "seed": 0,
                    "temperature": 1e-6,
                    "note": "Near-greedy is not deterministic argmax; no timing claim",
                },
                indent=2,
            )
        )
        print(json.dumps(result, indent=2))
    finally:
        engine.import_blocks = original_import
        llm.model_runner.model.compute_logits = original_logits
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.74)
    parser.add_argument("--master-port", type=int, default=23412)
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
