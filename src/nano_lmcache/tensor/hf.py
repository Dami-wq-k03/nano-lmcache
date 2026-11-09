"""Reuse cached prefix KV when running a HuggingFace causal language model.

This is where the pure-Python cache meets a real model: the payloads that
`ChunkedPrefixCache` stores are KV tensors (see `kv_chunks` for the
layout), and a cache hit lets the model skip the forward pass for the
matched prefix.

Why reuse is exact and not an approximation: in a causal model the K and V
of a token depend only on that token and the ones before it. If two prompts
start with the same tokens, those tokens have identical K/V in both, so
the cached values can be fed back as `past_key_values` and only the
remaining tokens need computing. Chunk keys hash the whole prefix for this
reason: the same tokens after a different prefix have different K/V.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import DynamicCache

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.tensor.kv_chunks import LayerKV, join_kv, split_kv

# Keep unknown architecture fields: a redundant miss is safer than a wrong hit.
# Identity and actual dtype are recorded separately. The remaining exclusions
# affect output formatting, loading metadata, or decoding, not KV for given IDs.
_NON_KV_CONFIG_FIELDS = frozenset(
    {
        "_name_or_path", "_commit_hash", "revision", "dtype", "torch_dtype",
        "transformers_version", "architectures", "use_cache", "return_dict",
        "output_attentions", "output_hidden_states", "bos_token_id",
        "eos_token_id", "pad_token_id", "max_length", "min_length",
        "max_new_tokens", "min_new_tokens", "do_sample", "early_stopping",
        "num_beams", "num_beam_groups", "temperature", "top_k", "top_p",
        "typical_p", "repetition_penalty", "length_penalty",
        "no_repeat_ngram_size", "num_return_sequences", "bad_words_ids",
        "forced_bos_token_id", "forced_eos_token_id", "diversity_penalty",
        "return_dict_in_generate", "output_scores", "output_logits",
    }
)


def namespace_for(model: Any) -> str:
    """Derive a stable KV namespace for a fixed Hugging Face causal LM.

    Includes model name, resolved revision, actual model dtype, model class,
    and the config except known non-KV fields. This covers layer/head counts,
    head dimensions, RoPE, norm epsilon, and MLP settings that affect later
    layers' KV. Generation settings do not affect KV for fixed input tokens.

    Set model.config.revision to a stable identifier of the exact weights, and give
    config.name_or_path a nonempty name. Missing identity raises ValueError.
    This hashes metadata, not weights: after fine-tuning, adapter changes, or
    weight replacement, update the identity and create a new cache. Use an
    eval model with a fixed dtype (no changing autocast policy).
    """
    config = model.config
    name = config.name_or_path
    revision = getattr(config, "_commit_hash", None) or getattr(
        config, "revision", None
    )
    if not isinstance(name, str) or not name:
        raise ValueError("model config.name_or_path must identify the weights")
    if not isinstance(revision, str) or not revision:
        raise ValueError(
            "model revision is unknown; set config.revision to a stable "
            "identifier of the exact weights"
        )
    fields = {
        key: value for key, value in config.to_dict().items()
        if key not in _NON_KV_CONFIG_FIELDS
    }
    identity = {
        "format": 1,
        "model": name,
        "revision": revision,
        "dtype": str(model.dtype),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config": fields,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "hf-kv/v1/" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReuseResult:
    """Outcome of running a prompt through a prefix-cached model."""

    # Logits for the positions that were actually computed, shaped
    # [1, computed_tokens, vocab_size]. The last one predicts the next token.
    logits: torch.Tensor
    # Leading tokens whose KV came from the cache instead of the model.
    matched_tokens: int
    # Tokens the model had to run a forward pass over.
    computed_tokens: int


def cache_to_layers(cache: DynamicCache) -> list[LayerKV]:
    """Read a HuggingFace cache as per-layer (K, V), dropping the batch dim.

    Only plain attention layers are supported; a layer without K/V tensors
    (for example a linear-attention layer) raises ValueError.
    """
    layers: list[LayerKV] = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise ValueError(
                "only plain attention layers with K/V tensors are supported"
            )
        layers.append((keys[0], values[0]))

    return layers


def layers_to_cache(layers: Sequence[LayerKV], config: Any) -> DynamicCache:
    """Build a HuggingFace cache from per-layer (K, V) of a single sequence."""
    cache = DynamicCache(config=config)
    for index, (key, value) in enumerate(layers):
        cache.update(key.unsqueeze(0), value.unsqueeze(0), index)

    return cache


def run_with_prefix_cache(
    model: Any,
    cache: ChunkedPrefixCache,
    token_ids: Sequence[int],
) -> ReuseResult:
    """Run `token_ids` through `model`, reusing and then storing prefix KV.

    `model` is a transformers causal LM (for example from
    `AutoModelForCausalLM`). Steps:

    1. Look up the longest cached prefix of whole chunks.
    2. Rebuild `past_key_values` from the matched chunks.
    3. Run the model over only the remaining tokens.
    4. Split and store only new complete chunks after the matched prefix.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")

    match = cache.lookup(token_ids)

    # The model must process at least one token to produce next-token
    # logits, so a fully cached prompt still recomputes its last token.
    reused_tokens = min(match.matched_tokens, len(token_ids) - 1)

    device = next(model.parameters()).device
    if reused_tokens > 0:
        # Payloads are opaque to the cache; here we know they are chunk tensors.
        reused = join_kv(cast(Sequence[torch.Tensor], match.values))
        # A backend may hand chunks back on another device (a disk tier
        # returns CPU tensors), so move them to the model; `.to` is free
        # when they are already there.
        past = layers_to_cache(
            [
                (
                    key[:, :reused_tokens].to(device),
                    value[:, :reused_tokens].to(device),
                )
                for key, value in reused
            ],
            model.config,
        )
    else:
        past = DynamicCache(config=model.config)

    new_ids = torch.tensor([list(token_ids[reused_tokens:])], device=device)
    with torch.no_grad():
        output = model(new_ids, past_key_values=past, use_cache=True)

    start_chunk = match.matched_chunks
    num_complete_chunks = len(token_ids) // cache.chunk_size

    start_token = start_chunk * cache.chunk_size
    end_token = num_complete_chunks * cache.chunk_size

    # `past` now holds the reused KV plus the KV of the tokens just computed.
    new_layers = [
        (
            key[:, start_token:end_token],
            value[:, start_token:end_token]
        )
        for key, value in cache_to_layers(past)
    ]

    cache.store(
        token_ids,
        split_kv(new_layers, cache.chunk_size),
        start_chunk=start_chunk
    )

    return ReuseResult(
        logits=output.logits,
        matched_tokens=reused_tokens,
        computed_tokens=len(token_ids) - reused_tokens,
    )
