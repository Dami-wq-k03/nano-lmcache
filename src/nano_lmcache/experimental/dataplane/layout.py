"""Layout metadata for the experimental KV data plane.

The descriptor is deliberately independent of PyTorch. Tensor adapters can use
it to agree on shapes and byte sizes without making the cache's ABI depend on a
particular tensor library.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

AxisLayout = Literal["NHD", "HND"]
LayerLayout = Literal["layer-major", "page-major"]

_DTYPE_BITS = {
    "bfloat16": 16,
    "bf16": 16,
    "float16": 16,
    "fp16": 16,
    "float32": 32,
    "fp32": 32,
    "float64": 64,
    "fp64": 64,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "fp8_e4m3fn": 8,
    "fp8_e5m2": 8,
    "int8": 8,
    "uint8": 8,
    "int4": 4,
    "uint4": 4,
}


def _dtype_bits(dtype: str) -> int:
    name = dtype.removeprefix("torch.").lower().replace("-", "_")
    try:
        return _DTYPE_BITS[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {dtype!r}") from error


def _axis_shape(layout: AxisLayout, page_tokens: int, num_kv_heads: int,
                head_dim: int) -> tuple[int, int, int]:
    if layout == "NHD":
        return (page_tokens, num_kv_heads, head_dim)
    return (num_kv_heads, page_tokens, head_dim)


@dataclass(frozen=True, slots=True)
class KVLayout:
    """Describe the bytes and tensor axes of one logical KV page."""

    model_namespace: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_tokens: int
    dtype: str
    model_revision: str = ""
    tokenizer_fingerprint: str = ""
    key_layout: AxisLayout = "NHD"
    value_layout: AxisLayout = "NHD"
    layer_layout: LayerLayout = "layer-major"
    engine_layout_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.model_namespace:
            raise ValueError("model_namespace must not be empty")
        if not self.engine_layout_version:
            raise ValueError("engine_layout_version must not be empty")
        for name in ("num_layers", "num_kv_heads", "head_dim", "page_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.key_layout not in ("NHD", "HND"):
            raise ValueError(f"key_layout must be NHD or HND, got {self.key_layout!r}")
        if self.value_layout not in ("NHD", "HND"):
            raise ValueError(
                f"value_layout must be NHD or HND, got {self.value_layout!r}"
            )
        if self.layer_layout not in ("layer-major", "page-major"):
            raise ValueError(
                "layer_layout must be layer-major or page-major, "
                f"got {self.layer_layout!r}"
            )
        _dtype_bits(self.dtype)

    @property
    def dtype_bits(self) -> int:
        """Number of bits used by one stored scalar."""
        return _dtype_bits(self.dtype)

    @property
    def key_shape(self) -> tuple[int, int, int]:
        """Shape of one page's key tensor, excluding layers."""
        return _axis_shape(
            self.key_layout, self.page_tokens, self.num_kv_heads, self.head_dim
        )

    @property
    def value_shape(self) -> tuple[int, int, int]:
        """Shape of one page's value tensor, excluding layers."""
        return _axis_shape(
            self.value_layout, self.page_tokens, self.num_kv_heads, self.head_dim
        )

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        """Shape when keys and values use the same axis order."""
        if self.key_layout != self.value_layout:
            raise ValueError("tensor_shape is ambiguous for different K/V layouts")
        return self.key_shape

    @property
    def tensor_shapes(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Key and value shapes, in that order."""
        return self.key_shape, self.value_shape

    @property
    def storage_shape(self) -> tuple[int, ...]:
        """Shape of a page when K/V use the same axis order.

        ``layer-major`` stores layers before the K/V axis. ``page-major``
        stores the K/V records before the layer axis; the descriptor keeps this
        order explicit even though a single page has no outer page dimension.
        """
        if self.key_layout != self.value_layout:
            raise ValueError("storage_shape is ambiguous for different K/V layouts")
        prefix = (
            (self.num_layers, 2)
            if self.layer_layout == "layer-major"
            else (2, self.num_layers)
        )
        return (*prefix, *self.tensor_shape)

    @property
    def page_nbytes(self) -> int:
        """Bytes required by a full page, including every layer and K/V."""
        return self.nbytes_for_tokens(self.page_tokens)

    @property
    def fingerprint(self) -> str:
        """Stable identifier for layout-compatible physical pages."""
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible descriptor for persistence and traces."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "KVLayout":
        """Rebuild a validated layout from :meth:`to_dict` output."""
        return cls(**values)

    def nbytes_for_tokens(self, token_count: int) -> int:
        """Return the packed byte count for a page containing ``token_count``."""
        if not 0 < token_count <= self.page_tokens:
            raise ValueError(
                f"token_count must be in 1..{self.page_tokens}, got {token_count}"
            )
        scalar_count = (
            self.num_layers
            * 2
            * self.num_kv_heads
            * token_count
            * self.head_dim
        )
        bits = scalar_count * self.dtype_bits
        return (bits + 7) // 8
