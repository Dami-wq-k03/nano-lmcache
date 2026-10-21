"""What a nano-vLLM KV block's bytes depend on, and the cache namespace.

nano-lmcache's L2 outlives the process that wrote it, so a block can be
read by an engine configured differently from the one that computed it.
A block of the same shape can still hold the wrong numbers: other weights,
another dtype, a different split of KV heads across GPUs, other RoPE
settings. Every chunk key hashes a namespace, so putting everything that
changes a block's bytes into the namespace makes incompatible blocks
simply never hit.

The two possible mistakes are not equally bad. Leaving out a field that
does not matter only causes a miss: the block is recomputed. Leaving out a
field that does matter causes a wrong hit: the model silently reads KV
that does not belong to it. So the model config is hashed as a whole,
minus a short list of fields that provably cannot change KV, instead of
picking the fields that seem to matter.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from nano_lmcache.integrations.nanovllm.kv_blocks import chunk_shape

BLOCK_FORMAT_VERSION = 1

# Config fields that cannot change any KV value. Token IDs only choose
# which tokens are fed in, `use_cache` toggles caching itself, and the
# rest describe where and how the config was loaded.
_FIELDS_THAT_CANNOT_CHANGE_KV = frozenset(
    {
        "_name_or_path",
        "transformers_version",
        "architectures",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "use_cache",
        "return_dict",
        "output_attentions",
        "output_hidden_states",
    }
)


def model_config_fingerprint(hf_config: Any) -> str:
    """Hash of a Hugging Face model config, ignoring fields that cannot change KV.

    Covers everything that shapes the numbers in a KV block for given
    weights, for example: RoPE parameters (nano-vLLM caches keys after
    rotary embedding), the norm epsilon (Qwen3 normalizes keys before
    caching them), and the MLP and activation settings (they shape the
    hidden states every later layer's KV is computed from).
    """
    fields = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    relevant = {
        name: value
        for name, value in fields.items()
        if name not in _FIELDS_THAT_CANNOT_CHANGE_KV
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _dtype_name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class KVBlockLayout:
    """Everything that determines the bytes of one nano-vLLM KV block."""

    # Model identity, for example "Qwen/Qwen3-0.6B".
    model: str
    # Which weights: a repository commit, a release tag, or a checksum.
    # Different weights give different KV for the same tokens.
    revision: str
    # torch dtype name of the KV cache, for example "bfloat16".
    dtype: str
    num_layers: int
    # KV heads held by this rank; nano-vLLM splits them across GPUs.
    num_kv_heads: int
    head_dim: int
    block_size: int = 256
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    # `model_config_fingerprint` of the model config, or "" if unknown.
    config_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.revision:
            raise ValueError(
                "revision must not be empty; use a commit, tag, or checksum "
                "of the weights"
            )
        for name in ("num_layers", "num_kv_heads", "head_dim", "block_size",
                     "tensor_parallel_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.tensor_parallel_rank < self.tensor_parallel_size:
            raise ValueError(
                "tensor_parallel_rank must be in 0..tensor_parallel_size-1"
            )
        if not isinstance(getattr(torch, self.dtype, None), torch.dtype):
            raise ValueError(f"unknown torch dtype {self.dtype!r}")

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
        """Read the layout from nano-vLLM's `ModelRunner.kv_cache`."""
        num_layers, _, num_kv_heads, block_size, head_dim = chunk_shape(kv_cache)
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
        """Size a layout the way nano-vLLM sizes its KV cache.

        Mirrors `ModelRunner.allocate_kv_cache` at nano-vLLM `bb823b3`: KV
        heads are divided evenly across ranks, `head_dim` falls back to
        `hidden_size // num_attention_heads`, and the cache uses the model
        config's dtype.
        """
        total_kv_heads = hf_config.num_key_value_heads
        if total_kv_heads % tensor_parallel_size:
            raise ValueError(
                f"{total_kv_heads} KV heads are not divisible by "
                f"tensor_parallel_size={tensor_parallel_size}"
            )
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
    def torch_dtype(self) -> torch.dtype:
        dtype: torch.dtype = getattr(torch, self.dtype)
        return dtype

    @property
    def chunk_shape(self) -> tuple[int, int, int, int, int]:
        """Shape of one stored block: [layers, 2, kv_heads, block_size, head_dim]."""
        return (self.num_layers, 2, self.num_kv_heads, self.block_size,
                self.head_dim)

    @property
    def chunk_nbytes(self) -> int:
        count = 1
        for size in self.chunk_shape:
            count *= size
        return count * int(torch.empty((), dtype=self.torch_dtype).element_size())

    def namespace(self) -> str:
        """Differs whenever any field, or the block format version, differs."""
        fields = {"format": BLOCK_FORMAT_VERSION, **asdict(self)}
        digest = hashlib.sha256(
            json.dumps(fields, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"nanovllm-kv/v{BLOCK_FORMAT_VERSION}/{digest[:32]}"
