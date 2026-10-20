"""Serialize KV tensors for `DiskKVCache`.

Uses `torch.save` / `torch.load(weights_only=True)`. `torch.save` writes a
pickle, which is why `nano_lmcache.storage.serializers` warns against
pickle in general. `weights_only=True` swaps in a restricted unpickler that
only rebuilds tensors and plain containers and refuses every other class,
so a malicious file in the cache directory cannot run code on load.
"""

import io

import torch


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
