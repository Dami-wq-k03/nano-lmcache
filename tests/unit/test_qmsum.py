"""CPU checks for the real-workload benchmark, without nano-vLLM or a GPU."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_nanovllm"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_qmsum")
analyze = load_script("analyze_qmsum")
runner = load_script("run_qmsum")


@pytest.mark.parametrize(
    "sequences,expected",
    [
        ([], 0),
        ([[1, 2], [1, 3]], 1),
        ([[1, 2], [1, 2]], 2),
        ([[1], [1, 2]], 1),
        ([[1, 2], [1, 2], [1, 9]], 1),
    ],
)
def test_lcp_compares_all_final_token_ids(sequences, expected):
    assert prepare.prefix_length(sequences) == expected


class Tokenizer:
    chat_template = "fixed-test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert messages[0]["role"] == "system"
        assert "Speaker: full original transcript" in messages[1]["content"]
        question = messages[1]["content"].split("Question:\n")[1]
        return [7] * 512 + list(question.encode())

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def meeting():
    return {
        "meeting_transcripts": [
            {"speaker": "Speaker", "content": "full original transcript"}
        ],
        "general_query_list": [{"query": "a", "answer": "reference"}],
        "specific_query_list": [{"query": "b"}, {"query": " a "}, {"query": ""}],
    }


def test_query_dedup_keeps_original_order():
    rows = prepare.queries(meeting())
    assert [r["question"] for r in rows] == ["a", "b"]
    assert [r["query_id"] for r in rows] == ["general-0", "specific-0"]


def test_full_transcript_uses_model_template_and_block_alignment():
    rows, stats = prepare.tokenize_meeting(meeting(), Tokenizer())
    assert stats["shared_prefix_tokens"] == 512
    assert stats["cacheable_prefix_tokens"] == 512
    assert stats["num_cacheable_blocks"] == 2
    assert all(r["prompt_tokens"] == 513 for r in rows)


def test_engine_never_reuses_the_last_full_block():
    class Identical(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 512

    _, stats = prepare.tokenize_meeting(meeting(), Identical())
    assert stats["shared_prefix_tokens"] == 512
    assert stats["cacheable_prefix_tokens"] == 256


def test_tokenizer_dict_return_is_rejected():
    class Wrong(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1], "attention_mask": [1]}

    with pytest.raises(TypeError, match="flat list"):
        prepare.tokenize_meeting(meeting(), Wrong())


def test_prepare_freezes_ids_prefers_test_and_refuses_overwrite(tmp_path):
    raw = tmp_path / "raw"
    for split in ("train", "val", "test"):
        directory = raw / split
        directory.mkdir(parents=True)
        (directory / f"{split}.jsonl").write_text(json.dumps(meeting()) + "\n")
    args = Namespace(
        raw=raw,
        output=tmp_path / "output",
        model=Path("local-model"),
        max_output=32,
        max_model_len=600,
        meetings=1,
        max_queries=4,
    )
    prepare.prepare(args, Tokenizer())
    records = [
        json.loads(line)
        for line in (args.output / "workload.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert {r["meeting_id"] for r in records} == {"test-000"}
    assert all(r["input_ids"][:512] == [7] * 512 for r in records)
    assert len(runner.read_workload(args.output / "workload.jsonl", 600, 32)) == 1
    with pytest.raises(ValueError, match="empty"):
        prepare.prepare(args, Tokenizer())
    with pytest.raises(ValueError, match="exceeds context"):
        runner.read_workload(args.output / "workload.jsonl", 520, 32)


def result(mode="off", tokens=None):
    return {
        "mode": mode,
        "concurrency": 1,
        "max_tokens": 1,
        "repeat": 0,
        "meeting_id": "test-000",
        "query_id": "specific-0",
        "cold": False,
        "bucket": "1K-4K",
        "prompt_tokens": 1030,
        "ttft_ms": 100.0,
        "e2e_ms": 100.0,
        "output_token_ids": tokens or [42],
        "reused_tokens": 0,
        "prefill_tokens": 1030,
        "l2_hit_blocks": 0,
        "store_ms": 0.0,
        "fetch_ms": 0.0,
    }


def test_analyzer_preserves_mismatches_and_missing_baselines():
    rows = [result(), result("sync", [43]), result("async-flush")]
    summary = analyze.summarize(rows, [])
    assert summary["correctness"]["compared_requests"] == 2
    assert summary["correctness"]["mismatch_requests"] == 1
    assert summary["correctness"]["mismatches"][0]["mode"] == "sync"
    assert (
        analyze.summarize([result("sync")], [])["correctness"]["unpaired_requests"] == 1
    )


def test_analyzer_counts_actual_restored_blocks_and_cohort_wall():
    off, on = result(), result("sync")
    on.update(
        l2_hit_blocks=3,
        reused_tokens=768,
        prefill_tokens=262,
        ttft_ms=50.0,
        e2e_ms=50.0,
    )
    cohort = dict(
        mode="sync", concurrency=1, max_tokens=1, repeat=0, requests=2, wall_ms=1000
    )
    summary = analyze.summarize([off, on], [cohort])
    row = next(r for r in summary["latency"] if r["mode"] == "sync")
    assert row["l2_block_hit_rate"] == 0.75
    assert row["speedup_ratio_of_medians"] == 2.0
    assert summary["throughput"][0]["requests_per_second"] == 2.0


def test_duplicate_results_are_not_silently_paired():
    with pytest.raises(ValueError, match="duplicate"):
        analyze.summarize([result(), result()], [])


def test_percentiles_include_singletons_and_tails():
    assert analyze.percentile([9], 0.95) == 9
    assert analyze.percentile([1, 2, 3], 0.5) == 2
    assert analyze.percentile([0, 100], 0.95) == 95


def test_fp8_block_scale_formula():
    torch = pytest.importorskip("torch")
    convert = load_script("dequantize_fp8")
    weights = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    scales = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = weights * scales.repeat_interleave(2, 0).repeat_interleave(2, 1)
    assert torch.equal(convert.dequantize(weights, scales, (2, 2)), expected)
    with pytest.raises(ValueError, match="scale shape"):
        convert.dequantize(weights, scales[:1], (2, 2))


def test_diagnostic_checks_bytes_not_just_numeric_equality():
    torch = pytest.importorskip("torch")
    diagnostic = load_script("diagnose_qmsum")
    positive = torch.tensor([0.0], dtype=torch.bfloat16)
    negative = torch.tensor([-0.0], dtype=torch.bfloat16)
    assert torch.equal(positive, negative)
    assert not diagnostic.bitwise_equal(positive, negative)
    assert diagnostic.bitwise_equal(positive, positive.clone())
    assert not diagnostic.bitwise_equal(positive, positive.float())
    assert not diagnostic.bitwise_equal(positive, None)
