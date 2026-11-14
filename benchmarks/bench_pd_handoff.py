"""Two-GPU real-model KV handoff, with colocated greedy generation as control.

This is a single-process Hugging Face prototype with two model instances,
not a disaggregated serving scheduler or an engine-to-engine RPC protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from nano_lmcache.tensor.peer import PeerTransferEngine

    if torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    for device in (args.prefill_device, args.decode_device):
        free, _ = torch.cuda.mem_get_info(device)
        if free < args.min_free_gib * 1024**3:
            raise RuntimeError(
                f"device {device} lacks the requested free-memory margin"
            )
        # Cap allocator use; retain a margin for other processes and CUDA context.
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(0.5, (free - 512 * 1024**2) / total), device
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(
        "The council discussed library funding. " * args.prompt_tokens
    )
    tokens = tokens[: args.prompt_tokens]
    models = [
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        )
        .eval()
        .to(f"cuda:{device}")
        for device in (args.prefill_device, args.decode_device)
    ]
    source_model, target_model = models

    def prefill(model):
        output = model(
            input_ids=torch.tensor([tokens], device=model.device),
            use_cache=True,
            logits_to_keep=1,
        )
        return output.past_key_values, int(output.logits[0, -1].argmax())

    def finish(model, cache, first):
        generated = [first]
        samples = []
        while len(generated) < args.output_tokens:
            torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            output = model(
                input_ids=torch.tensor([[generated[-1]]], device=model.device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            generated.append(int(output.logits[0, -1].argmax()))
            cache = output.past_key_values
            torch.cuda.synchronize(model.device)
            samples.append((time.perf_counter() - start) * 1000)
        return generated, samples

    rows = []
    with (
        torch.inference_mode(),
        PeerTransferEngine(args.prefill_device, args.decode_device) as transport,
    ):
        for iteration in range(args.warmup + args.repeats):
            control_cache, control_first = prefill(target_model)
            control, _ = finish(target_model, control_cache, control_first)
            del control_cache
            torch.cuda.synchronize(source_model.device)
            start = time.perf_counter()
            cache, first = prefill(source_model)
            torch.cuda.synchronize(source_model.device)
            prefill_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            packed = torch.stack(
                [torch.stack((layer.keys, layer.values)) for layer in cache.layers]
            )
            torch.cuda.synchronize(source_model.device)
            pack_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            transfer = transport.copy(packed, mode=args.transport)
            restored = transfer.wait()
            transfer_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            target_cache = DynamicCache(config=target_model.config)
            for index, layer in enumerate(restored):
                target_cache.update(layer[0], layer[1], index)
            restore_ms = (time.perf_counter() - start) * 1000
            actual, decode_ms = finish(target_model, target_cache, first)
            if control != actual:
                raise RuntimeError("handoff generation differs from colocated control")
            if iteration >= args.warmup:
                rows.append(
                    {
                        "queue_ms": 0,
                        "prefill_ms": prefill_ms,
                        "pack_ms": pack_ms,
                        "transfer_ms": transfer_ms,
                        "restore_ms": restore_ms,
                        "first_decode_ms": decode_ms[0] if decode_ms else None,
                        "decode_step_ms": decode_ms,
                        "kv_bytes": packed.nbytes,
                        "transport": transfer.transport,
                        "outputs_match": True,
                        "token_ids": actual,
                    }
                )
            del cache, packed, restored, target_cache, transfer
    return {
        "scope": "two model instances, one process; no scheduler or network",
        "config": vars(args) | {"json": str(args.json) if args.json else None},
        "environment": {
            "git_sha": os.environ.get("NANO_LMCACHE_GIT_SHA"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-device", type=int, default=0)
    parser.add_argument("--decode-device", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=2.5)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--transport", choices=("auto", "peer", "host"), default="auto")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.min_free_gib <= 0 or min(args.prefill_device, args.decode_device) < 0:
        parser.error("memory margin must be positive and devices nonnegative")
    if args.prefill_device == args.decode_device:
        parser.error("prefill/decode devices must differ")
    if (
        min(args.prompt_tokens, args.output_tokens, args.repeats) <= 0
        or args.warmup < 0
    ):
        parser.error("counts must be positive and warmup nonnegative")
    encoded = json.dumps(run(args), indent=2)
    if args.json:
        args.json.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
