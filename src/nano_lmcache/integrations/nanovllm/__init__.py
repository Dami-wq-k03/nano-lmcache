"""nano-lmcache as an external KV cache for nano-vLLM.

See `docs/nanovllm-integration.md` for the design. This package only
works on tensors and token IDs; it never imports nano-vLLM.
"""

from nano_lmcache.integrations.nanovllm.external_cache import (
    ExternalCacheStats,
    MembershipStorage,
    NanoVLLMExternalCache,
    max_reusable_blocks,
)
from nano_lmcache.integrations.nanovllm.kv_blocks import (
    chunk_shape,
    export_block,
    import_block,
)
from nano_lmcache.integrations.nanovllm.layout import (
    KVBlockLayout,
    model_config_fingerprint,
)

__all__ = [
    "ExternalCacheStats",
    "KVBlockLayout",
    "MembershipStorage",
    "NanoVLLMExternalCache",
    "chunk_shape",
    "export_block",
    "import_block",
    "max_reusable_blocks",
    "model_config_fingerprint",
]
