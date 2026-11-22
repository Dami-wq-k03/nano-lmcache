import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

BLOCK_FORMAT_VERSION = 1


def _dtype_name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def model_config_fingerprint(hf_config: Any) -> str:
    fields = (
        hf_config.to_dict()
        if hasattr(hf_config, "to_dict")
        else dict(hf_config)
    )
    encoded = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class KVBlockLayout:
    model: str
    revision: str
    dtype: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 256
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    config_fingerprint: str = ""

    def namespace(self) -> str:
        fields = {"format": BLOCK_FORMAT_VERSION, **asdict(self)}
        encoded = json.dumps(fields, sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"nanovllm-kv/v{BLOCK_FORMAT_VERSION}/{digest[:32]}"

    @classmethod
    def from_kv_cache(
        cls,
        kv_cache: torch.Tensor,
        *,
        model: str,
        revision: str,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        config_fingerprint: str = "",
    ) -> "KVBlockLayout":
        _, num_layers, _, block_size, num_kv_heads, head_dim = kv_cache.shape
        return cls(
            model=model,
            revision=revision,
            dtype=_dtype_name(kv_cache.dtype),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            config_fingerprint=config_fingerprint,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        *,
        model: str,
        revision: str,
        block_size: int = 256,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
    ) -> "KVBlockLayout":
        total_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        return cls(
            model=model,
            revision=revision,
            dtype=_dtype_name(hf_config.dtype),
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=total_kv_heads // tensor_parallel_size,
            head_dim=head_dim,
            block_size=block_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            config_fingerprint=model_config_fingerprint(hf_config),
        )

    @property
    def chunk_shape(self) -> tuple[int, int, int, int, int]:
        return (self.num_layers, 2, self.num_kv_heads, self.block_size,
                self.head_dim)
