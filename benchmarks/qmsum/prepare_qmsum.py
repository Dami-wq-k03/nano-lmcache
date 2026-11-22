"""Freeze full QMSum meetings as chat-template token IDs (no inference)."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

BLOCK_SIZE = 256
INSTRUCTION = "Answer the question based on the meeting transcript."


def prefix_length(sequences):
    """Longest common prefix of the final, templated token IDs."""
    return next(
        (
            i
            for i, values in enumerate(zip(*sequences, strict=False))
            if len(set(values)) != 1
        ),
        min(map(len, sequences), default=0),
    )


def bucket(tokens):
    for bound, label in (
        (1024, "<1K"),
        (4096, "1K-4K"),
        (8192, "4K-8K"),
        (16384, "8K-16K"),
    ):
        if tokens < bound:
            return label
    return ">=16K"


def queries(meeting):
    """Keep source order and the first occurrence of each nonempty question."""
    seen = set()
    result = []
    for kind in ("general", "specific"):
        for index, item in enumerate(meeting[f"{kind}_query_list"]):
            question = item["query"].strip()
            if question and question not in seen:
                seen.add(question)
                result.append(
                    {
                        "query_id": f"{kind}-{index}",
                        "question": question,
                        "reference_answer": item.get("answer", ""),
                    }
                )
    return result


def tokenize_meeting(meeting, tokenizer):
    transcript = "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in meeting["meeting_transcripts"]
    )
    rows = queries(meeting)
    for row in rows:
        content = f"Meeting Transcript:\n\n{transcript}\n\nQuestion:\n{row['question']}"
        row["input_ids"] = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": content},
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(row["input_ids"], list) or not all(
            isinstance(token, int) for token in row["input_ids"]
        ):
            raise TypeError("chat template must return a flat list of token IDs")
        row["prompt_tokens"] = len(row["input_ids"])
    shared = prefix_length([row["input_ids"] for row in rows])
    # The engine always computes the last block, even if it is complete.
    reusable = min(
        [shared // BLOCK_SIZE]
        + [(row["prompt_tokens"] - 1) // BLOCK_SIZE for row in rows]
    )
    return rows, {
        "num_queries": len(rows),
        "total_transcript_tokens": len(
            tokenizer.encode(transcript, add_special_tokens=False)
        ),
        "shared_prefix_tokens": shared,
        "cacheable_prefix_tokens": reusable * BLOCK_SIZE,
        "num_cacheable_blocks": reusable,
        "bucket": bucket(shared),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=0),
    }


def prepare(args, tokenizer):
    args.output.mkdir(parents=True, exist_ok=True)
    # Never silently replace the fixed experimental input.
    if any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    inventory, eligible = [], []
    dataset_hashes, totals = {}, Counter()
    for split in ("train", "val", "test"):
        path = args.raw / split / f"{split}.jsonl"
        dataset_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            for index, line in enumerate(stream):
                meeting = json.loads(line)
                totals["meetings"] += 1
                totals["query_pairs"] += sum(
                    len(meeting[f"{k}_query_list"]) for k in ("general", "specific")
                )
                rows, info = tokenize_meeting(meeting, tokenizer)
                info.update(meeting_id=f"{split}-{index:03d}", split=split)
                info["eligible"] = (
                    len(rows) >= 2
                    and info["num_cacheable_blocks"] > 0
                    and info["max_prompt_tokens"] + args.max_output
                    <= args.max_model_len
                )
                inventory.append(info)
                if info["eligible"]:
                    eligible.append((info, rows))
    # Prefer test; use val/train only if test cannot supply the pilot size.
    priority = {"test": 0, "val": 1, "train": 2}
    eligible.sort(
        key=lambda pair: (
            priority[pair[0]["split"]],
            -pair[0]["shared_prefix_tokens"],
            pair[0]["meeting_id"],
        )
    )
    selected = (
        eligible[: args.meetings]
        if args.meetings
        else [pair for pair in eligible if pair[0]["split"] == "test"]
    )
    if not selected:
        raise ValueError("no eligible meetings; no transcript was truncated")
    count = 0
    with (args.output / "workload.jsonl").open("x") as stream:
        for info, rows in selected:
            for index, row in enumerate(rows[: args.max_queries or None]):
                record = {**info, **row, "query_index": index}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    selected_info = [info for info, _ in selected]
    summary = {
        "dataset": "Yale-LILY/QMSum",
        "dataset_sha256": dataset_hashes,
        "model": str(args.model),
        "block_size": BLOCK_SIZE,
        "max_model_len": args.max_model_len,
        "max_output": args.max_output,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "enable_thinking": False,
        "total": dict(totals),
        "eligible_meetings": len(eligible),
        "eligible_test_meetings": sum(p[0]["split"] == "test" for p in eligible),
        "eligible_requests": sum(len(rows) for _, rows in eligible),
        "selected_meetings": len(selected),
        "selected_requests": count,
        "eligible_buckets": dict(Counter(p[0]["bucket"] for p in eligible)),
        "selected_buckets": dict(Counter(p["bucket"] for p in selected_info)),
        "mean_shared_prefix": statistics.mean(
            p["shared_prefix_tokens"] for p in selected_info
        ),
        "max_shared_prefix": max(p["shared_prefix_tokens"] for p in selected_info),
        "meetings": inventory,
    }
    workload = args.output / "workload.jsonl"
    summary["workload_sha256"] = hashlib.sha256(workload.read_bytes()).hexdigest()
    (args.output / "workload_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meetings"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="split/split.jsonl directories containing official data",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-output", type=int, default=64)
    parser.add_argument("--meetings", type=int, default=20, help="0: all eligible test")
    parser.add_argument("--max-queries", type=int, default=4, help="0: all queries")
    args = parser.parse_args()
    if min(args.meetings, args.max_queries) < 0 or args.max_queries == 1:
        parser.error("use >=2 queries (or 0 for all), and nonnegative meetings")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prepare(args, tokenizer)


if __name__ == "__main__":
    main()
