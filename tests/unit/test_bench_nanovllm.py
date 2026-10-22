"""The parts of the nano-vLLM benchmark that need no GPU (roadmap N4)."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_nanovllm.py"
)


@pytest.fixture(scope="module")
def bench():
    """Load the benchmark script by path; benchmarks/ is not a package."""
    spec = importlib.util.spec_from_file_location("bench_nanovllm", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_nanovllm"] = module
    spec.loader.exec_module(module)
    return module


def row(bench, blocks, no_cache, l1, memory, disk, disk_cached=None):
    return bench.Row(
        blocks=blocks,
        prompt_tokens=blocks * 256 + 32,
        reused_tokens=blocks * 256,
        no_cache_ms=no_cache,
        l1_ms=l1,
        l2_memory_ms=memory,
        l2_disk_ms=disk,
        l2_disk_page_cache_ms=disk if disk_cached is None else disk_cached,
        store_memory_ms=no_cache + 1,
        store_disk_ms=no_cache + 300 * blocks,
        load_memory_ms=2.0 * blocks,
        load_disk_ms=150.0 * blocks,
        load_disk_page_cache_ms=30.0 * blocks,
    )


def test_importing_needs_neither_torch_nor_nanovllm(bench):
    assert "nanovllm" not in sys.modules or bench is not None
    source = BENCH_PATH.read_text()
    top_level_imports = [
        line for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not any("torch" in line or "nanovllm" in line or "nano_lmcache" in line
                   for line in top_level_imports)


def test_prompt_has_the_requested_blocks_plus_a_tail(bench):
    token_ids = bench.make_prompt(blocks=3, tail=32, vocab_size=1000, seed=7)

    assert len(token_ids) == 3 * 256 + 32
    assert all(0 < t < 1000 for t in token_ids)
    assert token_ids == bench.make_prompt(blocks=3, tail=32, vocab_size=1000, seed=7)


def test_prompts_for_different_seeds_share_no_first_block(bench):
    first = bench.make_prompt(blocks=2, tail=32, vocab_size=1000, seed=1)
    second = bench.make_prompt(blocks=2, tail=32, vocab_size=1000, seed=2)

    assert first[:256] != second[:256]


def test_crossover_is_the_first_size_from_which_l2_stays_faster(bench):
    rows = [
        row(bench, 1, no_cache=20, l1=15, memory=25, disk=60),
        row(bench, 2, no_cache=25, l1=15, memory=20, disk=80),
        row(bench, 4, no_cache=40, l1=16, memory=30, disk=120),
        row(bench, 8, no_cache=80, l1=17, memory=45, disk=70),
    ]

    assert bench.crossover(rows, "l2_memory_ms") == 2
    assert bench.crossover(rows, "l2_disk_ms") == 8


def test_a_win_inside_the_noise_margin_is_not_a_crossover(bench):
    # 47 vs 50 ms is 6% faster: within the default 10% margin.
    rows = [
        row(bench, 1, no_cache=50, l1=49, memory=47, disk=200),
        row(bench, 2, no_cache=50, l1=49, memory=47, disk=200),
        row(bench, 4, no_cache=120, l1=50, memory=52, disk=600),
    ]

    assert bench.crossover(rows, "l2_memory_ms") == 4
    assert bench.crossover(rows, "l2_memory_ms", min_gain=0.0) == 1


def test_min_gain_must_be_a_fraction(bench):
    with pytest.raises(ValueError):
        bench.crossover([row(bench, 1, 1, 1, 1, 1)], "l2_memory_ms", min_gain=1.5)


def test_a_single_noisy_win_is_not_a_crossover(bench):
    rows = [
        row(bench, 1, no_cache=30, l1=10, memory=25, disk=60),  # wins once
        row(bench, 2, no_cache=30, l1=10, memory=35, disk=60),  # then loses
        row(bench, 4, no_cache=50, l1=10, memory=40, disk=60),
    ]

    assert bench.crossover(rows, "l2_memory_ms") == 4


def test_no_crossover_when_l2_never_wins(bench):
    rows = [row(bench, b, no_cache=10, l1=5, memory=20, disk=50) for b in (1, 2)]

    assert bench.crossover(rows, "l2_disk_ms") is None


def test_crossover_rejects_unknown_columns(bench):
    with pytest.raises(ValueError):
        bench.crossover([row(bench, 1, 1, 1, 1, 1)], "no_such_column")


def test_per_block_costs(bench):
    costs = bench.per_block_costs(row(bench, 4, no_cache=50, l1=10, memory=30, disk=90))

    # Recomputing: everything above the L1 time, spread over the 4 blocks.
    assert costs["recompute_ms"] == pytest.approx(10.0)
    # Loading: measured directly, outside generation.
    assert costs["load_memory_ms"] == pytest.approx(2.0)
    assert costs["load_disk_ms"] == pytest.approx(150.0)
    assert costs["load_disk_page_cache_ms"] == pytest.approx(30.0)


def test_tables_have_a_line_per_row(bench):
    rows = [row(bench, b, 20 * b, 10, 15 * b, 40 * b) for b in (1, 2, 4)]

    ttft = bench.format_ttft_table(rows)
    costs = bench.format_cost_table(rows)

    for table in (ttft, costs):
        lines = table.splitlines()
        assert len(lines) == 2 + len(rows)
        assert lines[0].startswith("|") and set(lines[1]) <= set("|-: ")
        # Every row has exactly as many cells as the header.
        assert len({line.count("|") for line in lines}) == 1


def test_summary_names_each_crossover(bench):
    rows = [row(bench, b, 20 * b, 10, 15 * b, 40 * b) for b in (1, 2, 4)]

    summary = bench.format_summary(rows)

    assert "CPU memory" in summary and "disk" in summary
    assert "never" in summary  # disk is slower at every size here


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs posix_fadvise")
def test_drop_page_cache_touches_every_file_and_keeps_contents(bench, tmp_path):
    (tmp_path / "a.entry").write_bytes(b"x" * 4096)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.entry").write_bytes(b"y" * 10)

    assert bench.drop_page_cache(tmp_path) == 2
    assert (tmp_path / "a.entry").read_bytes() == b"x" * 4096
