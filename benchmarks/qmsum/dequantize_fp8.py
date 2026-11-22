"""Offline block-FP8 -> BF16 for nano-vLLM's unquantized Qwen3 loader.

This reconstructs quantized weights, NOT the original pre-quantization model.
No activation quantization remains. Output goes in a new, separate directory.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def dequantize(weight, scales, block_size):
    """Qwen fine-grained FP8: real weight = FP8 weight * weight_scale_inv."""
    rows, cols = weight.shape
    br, bc = block_size
    if rows % br or cols % bc:
        raise ValueError("this small converter requires block-aligned weights")
    if tuple(scales.shape) != (rows // br, cols // bc):
        raise ValueError("scale shape does not match the FP8 blocks")
    expanded = weight.float().reshape(rows // br, br, cols // bc, bc)
    return (expanded * scales.float()[:, None, :, None]).reshape(rows, cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((args.source / "config.json").read_text())
    quant = config["quantization_config"]
    if config["model_type"] != "qwen3" or quant["quant_method"] != "fp8":
        raise ValueError("only Qwen3 fine-grained FP8 checkpoints are supported")
    if quant.get("fmt") != "e4m3" or quant.get("activation_scheme") != "dynamic":
        raise ValueError("unexpected FP8 encoding")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    weight_map, hashes, total = {}, {}, 0
    for path in sorted(args.source.glob("*.safetensors")):
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.endswith(".weight_scale_inv"):
                    continue
                tensor = source.get_tensor(name)
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_name = name + "_scale_inv"
                    tensor = dequantize(
                        tensor,
                        source.get_tensor(scale_name),
                        quant["weight_block_size"],
                    ).to(torch.bfloat16)
                elif not tensor.is_floating_point():
                    raise ValueError(f"unexpected tensor dtype: {name} {tensor.dtype}")
                tensors[name] = tensor.contiguous()
                weight_map[name] = path.name
                total += tensor.nbytes
        save_file(tensors, args.output / path.name, metadata={"format": "pt"})
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"Converted {path.name}", flush=True)
        del tensors
    if not weight_map:
        raise ValueError("checkpoint contains no weights")
    del config["quantization_config"]
    config["torch_dtype"] = "bfloat16"
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
    )
    for pattern in ("tokenizer*", "vocab.json", "merges.txt", "generation_config.json"):
        for path in args.source.glob(pattern):
            if path.is_file():
                shutil.copyfile(path, args.output / path.name)
    (args.output / "conversion.json").write_text(
        json.dumps(
            {
                "source_revision": args.source.name,
                "source_shard_sha256": hashes,
                "method": "float32(weight_fp8) * block_scale_inv -> bfloat16",
                "source_quantization": quant,
                "output_bytes": total,
                "note": "Reconstructed FP8 weights; not original BF16 checkpoint",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
