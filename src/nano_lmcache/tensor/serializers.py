"""Serialize KV tensors for `DiskKVCache`.

Uses `torch.save` / `torch.load(weights_only=True)`. `torch.save` writes a
pickle, which is why `nano_lmcache.storage.serializers` warns against
pickle in general. `weights_only=True` swaps in a restricted unpickler that
only rebuilds tensors and plain containers and refuses every other class,
so a malicious file in the cache directory cannot run code on load.
"""

import io

import torch

from nano_lmcache.tensor.quantization import QuantizedKV, dequantize_kv, quantize_kv


class TensorSerializer:
    """Encodes one tensor; always decodes onto the CPU.

    Decoding to the CPU keeps a disk tier independent of which GPU wrote
    the file. The caller (for example `run_with_prefix_cache`, or a
    `CPUOffloadKVCache` fast tier) moves it to the right device.
    """

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"TensorSerializer stores tensors, got {type(value).__name__}"
            )

        buffer = io.BytesIO()
        # `torch.save` writes a view's whole underlying storage, so a small
        # slice of a big tensor would save the big tensor. Copy just the view.
        torch.save(value.detach().cpu().contiguous().clone(), buffer)
        return buffer.getvalue()

    def loads(self, data: bytes) -> object:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class QuantizedTensorSerializer:
    """Opt-in symmetric INT8 storage; decodes to the original dtype on CPU.

    FP32 scales use absolute maxima over the token axis (-2). For KV chunks this
    isolates each layer, K/V, head and channel, so large key channels do
    not coarsen smaller ones. The INT8 tensor retains the original shape.
    """

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, torch.Tensor) or value.dtype not in (
            torch.float16, torch.bfloat16, torch.float32,
        ):
            raise TypeError("QuantizedTensorSerializer needs an FP16/BF16/FP32 tensor")

        quantized = quantize_kv(value)
        buffer = io.BytesIO()
        torch.save(
            {
                "values": quantized.values.cpu(),
                "scales": quantized.scales.cpu(),
                "dtype": quantized.dtype,
                "token_axis": quantized.token_axis,
            },
            buffer,
        )
        return buffer.getvalue()

    def loads(self, data: bytes) -> torch.Tensor:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        quantized = QuantizedKV(
            values=payload["values"],
            scales=payload["scales"],
            dtype=payload["dtype"],
            token_axis=payload.get("token_axis", -2),
        )
        return dequantize_kv(quantized)
