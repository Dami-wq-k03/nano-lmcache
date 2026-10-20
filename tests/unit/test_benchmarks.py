import importlib.util
from pathlib import Path
from types import ModuleType

BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_prefix_cache.py"
)


def load_bench() -> ModuleType:
    """Load the benchmark script by path; benchmarks/ is not a package."""
    spec = importlib.util.spec_from_file_location("bench_prefix_cache", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_returns_one_result_per_scenario():
    bench = load_bench()

    results = bench.run_benchmark(
        chunk_counts=[4, 8],
        capacities=[None, 4],
        repeats=2,
    )

    assert len(results) == 4
    assert {(r.chunk_count, r.capacity) for r in results} == {
        (4, None),
        (4, 4),
        (8, None),
        (8, 4),
    }


def test_benchmark_results_are_positive_timings():
    bench = load_bench()

    (result,) = bench.run_benchmark(
        chunk_counts=[4], capacities=[None], repeats=2
    )

    assert result.store_seconds > 0
    assert result.lookup_seconds > 0


def test_benchmark_reports_expected_hits():
    bench = load_bench()

    unbounded, bounded = bench.run_benchmark(
        chunk_counts=[8], capacities=[None, 4], repeats=1
    )

    # Unbounded keeps the whole prompt; capacity 4 keeps only a 4-chunk prefix.
    assert unbounded.matched_chunks == 8
    assert bounded.matched_chunks == 4


def test_format_table_has_a_row_per_result():
    bench = load_bench()
    results = bench.run_benchmark(
        chunk_counts=[4], capacities=[None, 2], repeats=1
    )

    table = bench.format_table(results)

    lines = table.splitlines()
    assert len(lines) == 2 + len(results)
    assert "chunks" in lines[0]
    assert "unbounded" in table


def test_churn_scenario_keeps_only_capacity_many_entries():
    bench = load_bench()

    result = bench.run_churn(
        prompts=20, chunks_per_prompt=4, capacity=8, repeats=1
    )

    assert result.seconds > 0
    # 80 chunks were written, so most must have been evicted.
    assert 0 < result.entries <= 8


def test_churn_without_capacity_keeps_everything():
    bench = load_bench()

    result = bench.run_churn(
        prompts=5, chunks_per_prompt=4, capacity=None, repeats=1
    )

    assert result.entries == 20
