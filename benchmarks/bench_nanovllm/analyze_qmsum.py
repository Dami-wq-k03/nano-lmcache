"""Summarize paired QMSum results; preserve cold requests and missing coverage."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def load(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def request_key(row):
    return tuple(
        row[k]
        for k in ("concurrency", "max_tokens", "repeat", "meeting_id", "query_id")
    )


def summarize(records, cohorts):
    groups, throughput = defaultdict(list), defaultdict(list)
    baseline = {request_key(r): r for r in records if r["mode"] == "off"}
    checks = {"compared_requests": 0, "mismatches": [], "unpaired_requests": 0}
    seen = set()
    for row in records:
        identity = (row["mode"], request_key(row))
        if identity in seen:
            raise ValueError("duplicate result: do not concatenate overlapping runs")
        seen.add(identity)
        key = tuple(
            row[k] for k in ("mode", "concurrency", "max_tokens", "cold", "bucket")
        )
        groups[key].append(row)
        if row["mode"] != "off":
            other = baseline.get(request_key(row))
            if other is None:
                checks["unpaired_requests"] += 1
            else:
                checks["compared_requests"] += 1
                if row["output_token_ids"] != other["output_token_ids"]:
                    checks["mismatches"].append(
                        {
                            "mode": row["mode"],
                            "request": request_key(row),
                            "off": other["output_token_ids"],
                            "on": row["output_token_ids"],
                        }
                    )
    summaries = []
    for key, rows in sorted(groups.items()):
        mode, concurrency, output_tokens, cold, bucket = key
        candidates = sum((r["prompt_tokens"] - 1) // 256 for r in rows)
        matched = [baseline[request_key(r)] for r in rows if request_key(r) in baseline]
        p50 = percentile([r["ttft_ms"] for r in rows], 0.5)
        item = dict(
            mode=mode,
            concurrency=concurrency,
            max_tokens=output_tokens,
            cold=cold,
            bucket=bucket,
            requests=len(rows),
            meetings=len({r["meeting_id"] for r in rows}),
            ttft_p50_ms=p50,
            ttft_p95_ms=percentile([r["ttft_ms"] for r in rows], 0.95),
            e2e_p50_ms=percentile([r["e2e_ms"] for r in rows], 0.5),
            e2e_p95_ms=percentile([r["e2e_ms"] for r in rows], 0.95),
            reused_tokens=sum(r["reused_tokens"] for r in rows),
            prefill_tokens=sum(r["prefill_tokens"] for r in rows),
            l2_block_hit_rate=sum(r["l2_hit_blocks"] for r in rows) / candidates
            if candidates
            else 0,
            store_hook_p50_ms=statistics.median(r["store_ms"] for r in rows),
            fetch_p50_ms=statistics.median(r["fetch_ms"] for r in rows),
        )
        if len(matched) == len(rows):
            item["speedup_ratio_of_medians"] = (
                percentile([r["ttft_ms"] for r in matched], 0.5) / p50
            )
            item["paired_speedup_p50"] = statistics.median(
                baseline[request_key(r)]["ttft_ms"] / r["ttft_ms"] for r in rows
            )
        summaries.append(item)
    for cohort in cohorts:
        throughput[
            tuple(cohort[k] for k in ("mode", "concurrency", "max_tokens", "repeat"))
        ].append(cohort)
    rates = []
    for key, rows in sorted(throughput.items()):
        wall = sum(r["wall_ms"] for r in rows)
        count = sum(r["requests"] for r in rows)
        rates.append(
            dict(zip(("mode", "concurrency", "max_tokens", "repeat"), key, strict=True))
            | {
                "requests": count,
                "wall_ms": wall,
                "requests_per_second": count * 1000 / wall,
            }
        )
    checks["mismatch_requests"] = len(checks["mismatches"])
    return {"correctness": checks, "latency": summaries, "throughput": rates}


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def figures(records, rates, directory, warning=None, model_label=None):
    """Three static figures, explicit light/dark themes and accompanying CSV tables."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in records if r["concurrency"] == 1 and r["max_tokens"] == 1]
    modes = [("off", "Cache off", "o"), ("async-continuous", "Async continuous", "s")]
    for theme, colors, surface, ink, grid in (
        ("light", ("#2a78d6", "#eb6834"), "#fcfcfb", "#0b0b0b", "#dededb"),
        ("dark", ("#3987e5", "#d95926"), "#1a1a19", "#f0efec", "#383835"),
    ):
        plt.rcParams.update(
            {
                "figure.facecolor": surface,
                "axes.facecolor": surface,
                "text.color": ink,
                "axes.labelcolor": ink,
                "xtick.color": ink,
                "ytick.color": ink,
                "axes.edgecolor": grid,
                "font.size": 11,
                "legend.facecolor": surface,
                "legend.edgecolor": surface,
            }
        )
        for chart in ("prefix_ttft", "hit_rate", "throughput"):
            if chart != "throughput" and not selected:
                continue
            if (
                chart == "throughput"
                and len({r["concurrency"] for r in rates if r["max_tokens"] == 32}) < 2
            ):
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
            header = []
            if model_label:
                context = "output=32" if chart == "throughput" else "C=1, output=1"
                header.append(f"{model_label} | {context}")
            if warning:
                header.append(warning)
            if header:
                fig.suptitle("\n".join(header), fontsize=10)
            table = []
            max_y = 0
            for (mode, label, marker), color in zip(modes, colors, strict=True):
                points = defaultdict(list)
                if chart == "prefix_ttft":
                    for row in selected:
                        if row["mode"] == mode and not row["cold"]:
                            points[
                                (row["meeting_id"], row["shared_prefix_tokens"])
                            ].append(row["ttft_ms"])
                    data = sorted(
                        (prefix / 1024, statistics.median(times))
                        for (_, prefix), times in points.items()
                    )
                elif chart == "hit_rate":
                    for row in selected:
                        if row["mode"] == mode:
                            points[row["query_index"] + 1].append(row)
                    data = [
                        (
                            index,
                            100
                            * sum(r["l2_hit_blocks"] for r in rows)
                            / sum((r["prompt_tokens"] - 1) // 256 for r in rows),
                        )
                        for index, rows in sorted(points.items())
                    ]
                else:
                    for row in rates:
                        if row["mode"] == mode and row["max_tokens"] == 32:
                            points[row["concurrency"]].append(
                                row["requests_per_second"]
                            )
                    data = [
                        (c, statistics.median(values))
                        for c, values in sorted(points.items())
                    ]
                if not data:
                    continue
                x, y = zip(*data, strict=True)
                max_y = max(max_y, max(y))
                if chart == "prefix_ttft":
                    ax.scatter(
                        x,
                        y,
                        s=64,
                        c=color,
                        marker=marker,
                        label=label,
                        edgecolors=surface,
                        linewidths=1.5,
                        alpha=0.85,
                    )
                else:
                    ax.plot(
                        x,
                        y,
                        color=color,
                        marker=marker,
                        markersize=8,
                        markeredgecolor=surface,
                        markeredgewidth=1.5,
                        linewidth=2,
                        label=label,
                    )
                # End labels use neutral ink, not series colors.
                offset = 12 if mode == "off" else -20
                if chart == "throughput":
                    offset = -offset
                if chart == "prefix_ttft" and y[-1] < max_y * 0.15:
                    offset = 12
                ax.annotate(
                    label,
                    (x[-1], y[-1]),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="right",
                    color=ink,
                )
                table.extend({"mode": mode, "x": a, "y": b} for a, b in data)
            labels = {
                "prefix_ttft": (
                    "Full meeting prefix vs TTFT",
                    "Shared prefix (Ki tokens)",
                    "Warm-request TTFT median (ms)",
                ),
                "hit_rate": (
                    "Restored blocks, not membership probes",
                    "Query number",
                    "Restored / reusable prompt blocks (%)",
                ),
                "throughput": (
                    "Throughput including cold requests and final drain",
                    "Requested concurrency",
                    "Requests / second (32 output tokens)",
                ),
            }
            title, xlabel, ylabel = labels[chart]
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.set_ylim(0, max_y * 1.2 if max_y else 1)
            ax.margins(x=0.15)
            if chart == "hit_rate":
                ax.set_ylim(-3, 120)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_xticks(sorted({r["query_index"] + 1 for r in selected}))
            ax.grid(axis="y", color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
            fig.savefig(directory / f"{chart}-{theme}.png", dpi=160)
            plt.close(fig)
            if theme == "light":
                write_csv(directory / f"{chart}.csv", table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--label", help="friendly model/GPU label for standalone figures"
    )
    args = parser.parse_args()
    records = load(args.results / "raw_results.jsonl")
    cohorts = load(args.results / "cohorts.jsonl")
    result = summarize(records, cohorts)
    result["analysis_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    env = json.loads((args.results / "environment.json").read_text())
    expected = env["expected_records"]
    result["coverage"] = {
        "expected_records": expected,
        "actual_records": len(records),
        "complete": len(records) == expected == sum(c["requests"] for c in cohorts),
    }
    waves = load(args.results / "waves.jsonl")
    result["timing"] = {
        "queue_wait_total_ms": sum(w["queue_wait_ms"] for w in waves),
        "outside_flush_total_ms": sum(w["outside_flush_ms"] for w in waves),
        "peak_allocated_bytes": max(w["peak_allocated_bytes"] for w in waves),
        "peak_reserved_bytes": max(w["peak_reserved_bytes"] for w in waves),
    }
    (args.results / "summary.json").write_text(json.dumps(result, indent=2))
    write_csv(args.results / "latency.csv", result["latency"])
    write_csv(args.results / "throughput.csv", result["throughput"])
    if args.figures:
        directory = args.results / "figures"
        directory.mkdir(exist_ok=True)
        warning = None
        if (
            not result["coverage"]["complete"]
            or not result["correctness"]["compared_requests"]
            or result["correctness"]["mismatch_requests"]
            or result["correctness"]["unpaired_requests"]
        ):
            warning = "Exploratory timings: strict token-ID equivalence not passed"
        if env.get("combined_from"):
            warning = (warning or "Descriptive timings") + "; mixed run conditions"
        figures(
            records,
            result["throughput"],
            directory,
            warning=warning,
            model_label=args.label or env["model"],
        )
    print(
        json.dumps(
            {"correctness": result["correctness"], "timing": result["timing"]}, indent=2
        )
    )
    if (
        result["correctness"]["mismatch_requests"]
        or result["correctness"]["unpaired_requests"]
        or not result["coverage"]["complete"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
