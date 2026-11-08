import copy
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache  # noqa: E402
from nano_lmcache.storage.memory import MemoryKVCache  # noqa: E402
from nano_lmcache.tensor.hf import namespace_for  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture
def model():
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    config._name_or_path = "test/tiny-qwen3"
    config._commit_hash = "weights-v1"
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config).eval()


def test_namespace_is_stable_across_processes(model, tmp_path):
    # Reload exactly the same weights, not another randomly initialized model.
    model.save_pretrained(tmp_path)
    code = textwrap.dedent("""
        import sys
        from transformers import AutoModelForCausalLM
        from nano_lmcache.tensor.hf import namespace_for
        model = AutoModelForCausalLM.from_pretrained(sys.argv[1]).eval()
        model.config._name_or_path = "test/tiny-qwen3"
        model.config._commit_hash = "weights-v1"
        print(namespace_for(model))
    """)
    namespaces = []
    for seed in ("1", "99"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=60,
        )
        namespaces.append(result.stdout.strip())
    assert namespaces == [namespace_for(model)] * 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_name_or_path", "test/other-model"),
        ("_commit_hash", "weights-v2"),
        ("num_hidden_layers", 3),
        ("num_attention_heads", 4),
        ("num_key_value_heads", 2),
        ("head_dim", 16),
        ("hidden_size", 32),
        ("intermediate_size", 64),
        ("rms_norm_eps", 1e-5),
        ("hidden_act", "relu"),
        ("max_position_embeddings", 128),
        ("rope_parameters", {"rope_type": "default", "rope_theta": 20000.0}),
        ("rope_scaling", {"rope_type": "linear", "factor": 2.0}),
        ("rope_theta", 20000.0),
    ],
)
def test_kv_relevant_fields_change_namespace(model, field, value):
    before = namespace_for(model)
    setattr(model.config, field, value)
    assert namespace_for(model) != before


def test_actual_dtype_changes_namespace_even_if_config_is_stale(model):
    before = namespace_for(model)
    config_dtype = model.config.dtype
    model.to(dtype=torch.float64)
    model.config.dtype = config_dtype
    assert namespace_for(model) != before


def test_generation_and_output_settings_do_not_change_namespace(model):
    before = namespace_for(model)
    model.generation_config.temperature = 0.5
    model.generation_config.top_k = 7
    model.generation_config.max_new_tokens = 20
    # Legacy transformers configs may keep decoding settings on model.config.
    model.config.temperature = 0.5
    model.config.top_k = 7
    model.config.max_length = 200
    model.config.output_hidden_states = True
    model.config.return_dict = False
    model.config.transformers_version = "different-metadata"
    assert namespace_for(model) == before


def test_nested_dictionary_order_does_not_change_namespace(model):
    before = namespace_for(model)
    model.config.rope_parameters = dict(
        reversed(list(model.config.rope_parameters.items()))
    )
    assert namespace_for(model) == before


def test_revision_fallback_and_missing_identity(model):
    expected = namespace_for(model)
    model.config._commit_hash = None
    model.config.revision = "weights-v1"
    assert namespace_for(model) == expected
    model.config.revision = None
    with pytest.raises(ValueError, match="revision"):
        namespace_for(model)
    model.config.revision = "weights-v1"
    model.config._name_or_path = ""
    with pytest.raises(ValueError, match="name"):
        namespace_for(model)


def test_shared_storage_cannot_hit_another_models_namespace(model):
    storage = MemoryKVCache()
    original = ChunkedPrefixCache(
        chunk_size=2, namespace=namespace_for(model), storage=storage
    )
    original.store([1, 2, 3, 4], ["kv-0", "kv-1"])
    same = ChunkedPrefixCache(
        chunk_size=2, namespace=namespace_for(model), storage=storage
    )
    assert same.lookup([1, 2, 3, 4]).matched_tokens == 4
    changed = copy.deepcopy(model)
    changed.config._commit_hash = "weights-v2"
    other = ChunkedPrefixCache(
        chunk_size=2, namespace=namespace_for(changed), storage=storage
    )
    assert other.lookup([1, 2, 3, 4]).matched_tokens == 0
