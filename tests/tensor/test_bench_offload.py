import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

BENCH_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_offload.py"


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_offload", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measures_every_copy_mode():
    bench = load_bench()

    results = bench.run_benchmark(device="cuda", megabytes=4, repeats=2)

    modes = {(r.pinned, r.non_blocking) for r in results}
    assert modes == {(False, False), (True, False), (True, True)}
    for result in results:
        assert result.to_cpu_gbps > 0
        assert result.to_gpu_gbps > 0
        assert result.call_ms >= 0


def test_chunk_megabytes_matches_a_known_model():
    bench = load_bench()

    # Qwen3-0.6B: 28 layers, 8 KV heads, head_dim 128, bf16, 256 tokens.
    size = bench.chunk_megabytes(
        layers=28, kv_heads=8, head_dim=128, tokens=256, bytes_per_value=2
    )

    assert size == pytest.approx(28 * 2 * 8 * 256 * 128 * 2 / 2**20)


def test_format_table_lists_each_mode():
    bench = load_bench()
    results = bench.run_benchmark(device="cuda", megabytes=2, repeats=1)

    table = bench.format_table(results)

    assert "pageable" in table
    assert "pinned" in table
    assert len(table.splitlines()) == 2 + len(results)
