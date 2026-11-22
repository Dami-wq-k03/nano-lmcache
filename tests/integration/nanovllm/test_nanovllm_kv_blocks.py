"""N0: move one KV block between nano-vLLM's paged cache and nano-lmcache.

Needs PyTorch, but neither a GPU nor nano-vLLM: the tests build CPU
tensors in the layout of nano-vLLM's `ModelRunner.kv_cache`,
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim].
"""

import pytest

torch = pytest.importorskip("torch")

from nano_lmcache.integrations.nanovllm.kv_blocks import (  # noqa: E402
    export_block,
    import_block,
)
from nano_lmcache.storage.disk import DiskKVCache  # noqa: E402
from nano_lmcache.tensor.kv_chunks import join_kv, split_kv  # noqa: E402
from nano_lmcache.tensor.serializers import TensorSerializer  # noqa: E402

pytestmark = pytest.mark.torch

LAYERS, BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIM = 3, 8, 4, 2, 5


def make_kv_cache(dtype=torch.float32, seed=0, device="cpu"):
    """Random KV cache in nano-vLLM's layout; every element is distinct."""
    generator = torch.Generator().manual_seed(seed)
    shape = (2, LAYERS, BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIM)
    return torch.randn(shape, generator=generator).to(dtype).to(device)


# ---- the format -------------------------------------------------------


def test_export_has_chunk_shape_dtype_and_device():
    kv_cache = make_kv_cache(torch.bfloat16)

    payload = export_block(kv_cache, 3)

    assert tuple(payload.shape) == (LAYERS, 2, HEADS, BLOCK_SIZE, HEAD_DIM)
    assert payload.dtype == torch.bfloat16
    assert payload.device == kv_cache.device


def test_export_puts_every_element_in_the_right_place():
    """A round trip alone would not catch a wrong but self-inverse permute."""
    kv_cache = make_kv_cache()
    block = 5

    payload = export_block(kv_cache, block)

    for layer in range(LAYERS):
        for kv in (0, 1):
            for head in range(HEADS):
                for token in range(BLOCK_SIZE):
                    assert torch.equal(
                        payload[layer, kv, head, token],
                        kv_cache[kv, layer, block, token, head],
                    )


def test_export_matches_split_kv_of_the_same_tokens():
    """Blocks from nano-vLLM use exactly the chunk format of tensor/kv_chunks."""
    kv_cache = make_kv_cache()
    block = 2
    # Per-layer (K, V) of this block's tokens, each [heads, tokens, head_dim].
    layers = [
        (
            kv_cache[0, layer, block].transpose(0, 1),
            kv_cache[1, layer, block].transpose(0, 1),
        )
        for layer in range(LAYERS)
    ]

    (expected,) = split_kv(layers, chunk_size=BLOCK_SIZE)

    assert torch.equal(export_block(kv_cache, block), expected)


def test_join_kv_reads_an_exported_block_back_as_layers():
    kv_cache = make_kv_cache()

    layers = join_kv([export_block(kv_cache, 6)])

    for layer, (key, value) in enumerate(layers):
        assert torch.equal(key, kv_cache[0, layer, 6].transpose(0, 1))
        assert torch.equal(value, kv_cache[1, layer, 6].transpose(0, 1))


# ---- round trips ------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_export_then_import_into_another_block_is_exact(dtype):
    source = make_kv_cache(dtype, seed=1)
    target = make_kv_cache(dtype, seed=2)

    import_block(target, 7, export_block(source, 1))

    assert torch.equal(target[:, :, 7], source[:, :, 1])


def test_import_only_touches_the_target_block():
    target = make_kv_cache(seed=3)
    before = target.clone()

    import_block(target, 4, export_block(make_kv_cache(seed=4), 0))

    for block in range(BLOCKS):
        if block != 4:
            assert torch.equal(target[:, :, block], before[:, :, block])


def test_many_blocks_move_without_crossing():
    source = make_kv_cache(seed=5)
    target = torch.zeros_like(source)
    mapping = {0: 6, 1: 3, 2: 7, 5: 0}  # source block -> target block

    payloads = {src: export_block(source, src) for src in mapping}
    for src, dst in mapping.items():
        import_block(target, dst, payloads[src])

    for src, dst in mapping.items():
        assert torch.equal(target[:, :, dst], source[:, :, src])
    for untouched in set(range(BLOCKS)) - set(mapping.values()):
        assert target[:, :, untouched].abs().sum() == 0


def test_export_is_an_independent_contiguous_copy():
    kv_cache = make_kv_cache()
    payload = export_block(kv_cache, 2)
    expected = payload.clone()

    # nano-vLLM reuses physical blocks for other tokens later.
    kv_cache.zero_()

    assert torch.equal(payload, expected)
    assert payload.is_contiguous()


def test_imported_block_does_not_alias_the_payload():
    target = make_kv_cache()
    payload = export_block(make_kv_cache(seed=9), 1)

    import_block(target, 3, payload)
    payload.zero_()

    assert target[:, :, 3].abs().sum() > 0


# ---- through existing storage ------------------------------------------


def test_payload_survives_tensor_serializer():
    kv_cache = make_kv_cache(torch.bfloat16)
    serializer = TensorSerializer()

    restored = serializer.loads(serializer.dumps(export_block(kv_cache, 2)))
    target = torch.zeros_like(kv_cache)
    import_block(target, 5, restored)

    assert torch.equal(target[:, :, 5], kv_cache[:, :, 2])


def test_payload_survives_a_disk_cache_restart(tmp_path):
    kv_cache = make_kv_cache(torch.bfloat16, seed=6)
    DiskKVCache(tmp_path, serializer=TensorSerializer()).put(
        "block", export_block(kv_cache, 3)
    )

    reopened = DiskKVCache(tmp_path, serializer=TensorSerializer())
    target = torch.zeros_like(kv_cache)
    import_block(target, 0, reopened.get("block"))

    assert torch.equal(target[:, :, 0], kv_cache[:, :, 3])


# ---- GPU --------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_cache_exports_on_gpu_and_imports_from_cpu():
    kv_cache = make_kv_cache(torch.bfloat16, seed=7, device="cuda")

    payload = export_block(kv_cache, 4)
    assert payload.is_cuda

    # A disk tier hands payloads back on the CPU.
    target = torch.zeros_like(kv_cache)
    import_block(target, 1, payload.cpu())

    assert torch.equal(target[:, :, 1], kv_cache[:, :, 4])
