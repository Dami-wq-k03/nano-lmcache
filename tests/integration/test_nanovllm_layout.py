"""N1: cache keys and namespaces for nano-vLLM KV blocks."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.integrations.nanovllm.layout import (  # noqa: E402
    KVBlockLayout,
    model_config_fingerprint,
)

pytestmark = pytest.mark.torch


def make_layout(**overrides):
    fields = dict(
        model="Qwen/Qwen3-0.6B",
        revision="abc123",
        dtype="bfloat16",
        num_layers=3,
        num_kv_heads=2,
        head_dim=4,
        block_size=4,
    )
    fields.update(overrides)
    return KVBlockLayout(**fields)


def test_chunk_shape_matches_export_block():
    from nano_lmcache.integrations.nanovllm.kv_blocks import export_block

    layout = make_layout()
    kv_cache = torch.zeros(2, 3, 5, 4, 2, 4, dtype=torch.bfloat16)

    assert layout.chunk_shape == tuple(export_block(kv_cache, 0).shape)
    assert layout.chunk_nbytes == 3 * 2 * 2 * 4 * 4 * 2  # bfloat16


def test_namespace_is_stable_and_versioned():
    namespace = make_layout().namespace()

    assert namespace == make_layout().namespace()
    assert namespace.startswith("nanovllm-kv/v1/")


@pytest.mark.parametrize(
    "change",
    [
        {"model": "Qwen/Qwen3-1.7B"},
        {"revision": "def456"},
        {"dtype": "float16"},
        {"num_layers": 4},
        {"num_kv_heads": 4},
        {"head_dim": 8},
        {"block_size": 8},
        {"tensor_parallel_size": 2},
        {"tensor_parallel_size": 2, "tensor_parallel_rank": 1},
        {"config_fingerprint": "other-config"},
    ],
)
def test_namespace_changes_with_anything_that_changes_block_bytes(change):
    assert make_layout(**change).namespace() != make_layout().namespace()


@pytest.mark.parametrize(
    "bad",
    [
        {"model": ""},
        {"revision": ""},
        {"num_layers": 0},
        {"num_kv_heads": -1},
        {"head_dim": 0},
        {"block_size": 0},
        {"tensor_parallel_size": 0},
        {"tensor_parallel_size": 2, "tensor_parallel_rank": 2},
        {"dtype": "not-a-dtype"},
        {"dtype": "Size"},  # an attribute of torch, but not a dtype
    ],
)
def test_invalid_layouts_are_rejected(bad):
    with pytest.raises(ValueError):
        make_layout(**bad)


def test_from_kv_cache_reads_shape_and_dtype():
    kv_cache = torch.zeros(2, 3, 7, 4, 2, 4, dtype=torch.bfloat16)

    layout = KVBlockLayout.from_kv_cache(
        kv_cache, model="Qwen/Qwen3-0.6B", revision="abc123"
    )

    assert layout == make_layout()


def test_from_kv_cache_rejects_other_tensors():
    with pytest.raises(ValueError, match="nano-vLLM KV cache"):
        KVBlockLayout.from_kv_cache(torch.zeros(2, 3, 4), model="m", revision="r")


# ---- from a Hugging Face config, as nano-vLLM sizes its KV cache ------


def tiny_config(**overrides):
    """A tiny Qwen3 config; skips only the calling test without transformers."""
    transformers = pytest.importorskip("transformers")
    fields = dict(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    fields.update(overrides)
    config = transformers.Qwen3Config(**fields)
    config.dtype = torch.bfloat16
    return config


def test_from_hf_config_matches_the_kv_cache_nanovllm_would_allocate():
    config = tiny_config()
    # nano-vLLM's ModelRunner.allocate_kv_cache, for tensor_parallel_size 1:
    kv_cache = torch.zeros(
        2, config.num_hidden_layers, 5, 256, config.num_key_value_heads,
        config.head_dim, dtype=config.dtype,
    )

    from_config = KVBlockLayout.from_hf_config(config, model="m", revision="r")
    from_cache = KVBlockLayout.from_kv_cache(
        kv_cache, model="m", revision="r",
        config_fingerprint=model_config_fingerprint(config),
    )

    assert from_config == from_cache


def test_from_hf_config_splits_kv_heads_across_ranks():
    layout = KVBlockLayout.from_hf_config(
        tiny_config(num_attention_heads=8, num_key_value_heads=4),
        model="m", revision="r", tensor_parallel_size=2, tensor_parallel_rank=1,
    )

    assert layout.num_kv_heads == 2
    assert layout.tensor_parallel_rank == 1


def test_from_hf_config_rejects_heads_that_do_not_split_evenly():
    with pytest.raises(ValueError, match="divisible"):
        KVBlockLayout.from_hf_config(
            tiny_config(num_key_value_heads=2, num_attention_heads=4),
            model="m", revision="r", tensor_parallel_size=4,
        )


def test_from_hf_config_falls_back_to_hidden_size_over_heads():
    # Same rule as nano-vLLM: configs without head_dim (older models) use
    # hidden_size // num_attention_heads.
    from types import SimpleNamespace

    fields = dict(num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, hidden_size=32, dtype=torch.float16)
    config = SimpleNamespace(**fields, to_dict=lambda: dict(fields))

    layout = KVBlockLayout.from_hf_config(config, model="m", revision="r")

    assert layout.head_dim == 32 // 4


def test_fingerprint_is_stable():
    assert model_config_fingerprint(tiny_config()) == model_config_fingerprint(
        tiny_config()
    )


@pytest.mark.parametrize(
    "change",
    [
        # Stored keys are rotated: nano-vLLM applies RoPE before caching K.
        {"rope_parameters": {"rope_theta": 500.0, "rope_type": "default"}},
        # Qwen3 normalizes K with this epsilon before caching it.
        {"rms_norm_eps": 1e-3},
        # Later layers' KV depends on earlier layers' activations.
        {"hidden_act": "gelu"},
        {"intermediate_size": 128},
    ],
)
def test_fingerprint_changes_with_fields_that_change_kv(change):
    assert model_config_fingerprint(tiny_config(**change)) != model_config_fingerprint(
        tiny_config()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"eos_token_id": 7},
        {"bos_token_id": 3},
        {"pad_token_id": 1},
        {"use_cache": False},
    ],
)
def test_fingerprint_ignores_fields_that_cannot_change_kv(change):
    assert model_config_fingerprint(tiny_config(**change)) == model_config_fingerprint(
        tiny_config()
    )


def test_fingerprint_ignores_where_the_model_was_loaded_from():
    first, second = tiny_config(), tiny_config()
    first._name_or_path = "/home/alice/models/qwen"
    second._name_or_path = "/srv/models/qwen"

    assert model_config_fingerprint(first) == model_config_fingerprint(second)


def test_layouts_from_different_configs_never_share_a_namespace():
    first = KVBlockLayout.from_hf_config(tiny_config(), model="m", revision="r")
    second = KVBlockLayout.from_hf_config(
        tiny_config(rms_norm_eps=1e-3), model="m", revision="r"
    )

    assert first.namespace() != second.namespace()
    # Only the fingerprint differs between the two layouts.
    same_fields = dataclasses.replace(
        first, config_fingerprint=second.config_fingerprint
    )
    assert same_fields == second
