import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "results" / "post_dpo_benchmarks.json"
OUTPUT_MD = ROOT / "results" / "static_eval_paper_report.md"

BENCHMARK_ORDER = ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval", "truthfulqa_mc", "mmlu_pro", "gpqa"]
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "arc": "ARC",
    "mmlu": "MMLU",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
    "truthfulqa_mc": "TruthfulQA-MC",
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA",
}
STAGE_ORDER = ["base", "math_sft", "science_sft", "code_sft", "merged_dare", "dpo"]
LOW_RANKS = [1, 2, 8, 128]
HIGH_RANKS = [1024, 3072]


def load_results():
    if not RESULTS_JSON.exists():
        return {}
    payload = json.loads(RESULTS_JSON.read_text())
    return payload.get("benchmarks", {})


def parse_key(key: str):
    model_key, rank_part, stage_part, bench_part = key.split("|")
    return {
        "model": model_key,
        "rank": int(rank_part.split("=")[1]),
        "stage": stage_part.split("=")[1],
        "benchmark": bench_part.split("=")[1],
    }


def group_qwen(results: dict):
    rows = []
    for key, value in results.items():
        meta = parse_key(key)
        if meta["model"] != "qwen_0.8b":
            continue
        row = {**meta, **value}
        rows.append(row)
    return rows


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def row_score_map(rows):
    mapping = {}
    for row in rows:
        mapping.setdefault((row["rank"], row["stage"]), {})[row["benchmark"]] = row.get("score")
    return mapping


def overall_score(score_dict: dict):
    return mean([score_dict.get(name) for name in BENCHMARK_ORDER])


def format_pct(value):
    return "-" if value is None else f"{100.0 * value:.1f}%"


def best_overall_rows(score_map):
    items = []
    for (rank, stage), scores in score_map.items():
        items.append((overall_score(scores), rank, stage))
    return sorted((item for item in items if item[0] is not None), reverse=True)


def best_per_benchmark(score_map):
    output = {}
    for bench in BENCHMARK_ORDER:
        candidates = []
        for (rank, stage), scores in score_map.items():
            score = scores.get(bench)
            if score is not None:
                candidates.append((score, rank, stage))
        output[bench] = max(candidates) if candidates else None
    return output


def aggregate_stage(score_map, ranks, stage):
    return {
        bench: mean([score_map.get((rank, stage), {}).get(bench) for rank in ranks])
        for bench in BENCHMARK_ORDER
    }


def dpo_vs_merged(score_map, ranks):
    rows = []
    for rank in ranks:
        merged = score_map.get((rank, "merged_dare"), {})
        dpo = score_map.get((rank, "dpo"), {})
        deltas = {bench: (dpo.get(bench) - merged.get(bench)) if merged.get(bench) is not None and dpo.get(bench) is not None else None for bench in BENCHMARK_ORDER}
        rows.append((rank, deltas, overall_score(deltas)))
    return rows


def low_vs_high(summary_map):
    low = aggregate_stage(summary_map, LOW_RANKS, "math_sft")
    high = aggregate_stage(summary_map, HIGH_RANKS, "math_sft")
    return low, high


def render():
    results = load_results()
    qwen_rows = group_qwen(results)
    score_map = row_score_map(qwen_rows)
    best_rows = best_overall_rows(score_map)
    best_bench = best_per_benchmark(score_map)
    dpo_deltas = dpo_vs_merged(score_map, [1, 2, 8, 128, 1024, 3072])
    low_math, high_math = low_vs_high(score_map)

    lines = [
        "# Static Eval Paper Report",
        "",
        "## Headline",
        "",
        "- Qwen static matrix is complete for the implemented paper suite.",
        "- The strongest overall regime is low-rank SFT, not a uniform DPO win.",
        "- High ranks `1024/3072` behave like unstable or weak-return ablations, especially on math-heavy benchmarks.",
        "",
        "## Best Overall Rows",
        "",
        "| Rank | Stage | Overall |",
        "|------|-------|---------|",
    ]
    for overall, rank, stage in best_rows[:10]:
        lines.append(f"| {rank} | {stage} | {format_pct(overall)} |")

    lines.extend(
        [
            "",
            "## Best Row Per Benchmark",
            "",
            "| Benchmark | Best Score | Rank | Stage |",
            "|-----------|------------|------|-------|",
        ]
    )
    for bench in BENCHMARK_ORDER:
        row = best_bench.get(bench)
        if row is None:
            lines.append(f"| {BENCHMARK_LABELS[bench]} | - | - | - |")
        else:
            score, rank, stage = row
            lines.append(f"| {BENCHMARK_LABELS[bench]} | {format_pct(score)} | {rank} | {stage} |")

    lines.extend(
        [
            "",
            "## DPO Minus Merged-DARE",
            "",
            "| Rank | Overall Delta | GSM8K | MATH-500 | ARC | MMLU | MBPP | HumanEval | TruthfulQA-MC | MMLU-Pro | GPQA |",
            "|------|---------------|-------|----------|-----|------|------|-----------|---------------|----------|------|",
        ]
    )
    for rank, deltas, overall in dpo_deltas:
        vals = [deltas.get(bench) for bench in BENCHMARK_ORDER]
        fmt = ["-" if v is None else f"{100.0 * v:+.1f}pp" for v in vals]
        lines.append(f"| {rank} | {'-' if overall is None else f'{100.0 * overall:+.1f}pp'} | " + " | ".join(fmt) + " |")

    lines.extend(
        [
            "",
            "## Low-Rank vs High-Rank Math-SFT",
            "",
            "| Group | Overall | GSM8K | MATH-500 | ARC | MMLU | MBPP | HumanEval | TruthfulQA-MC | MMLU-Pro | GPQA |",
            "|-------|---------|-------|----------|-----|------|------|-----------|---------------|----------|------|",
        ]
    )
    for label, scores in [("low_ranks_1_2_8_128", low_math), ("high_ranks_1024_3072", high_math)]:
        overall = overall_score(scores)
        fmt = [format_pct(scores.get(bench)) for bench in BENCHMARK_ORDER]
        lines.append(f"| {label} | {format_pct(overall)} | " + " | ".join(fmt) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Low-rank adapters dominate the strongest static rows.",
            "- DPO produces mixed static deltas, not a consistent uplift.",
            "- High-rank behavior is weak enough that `1024/3072` should remain ablations or controls, not deployment recommendations.",
            "- The most meaningful separations are on `GSM8K` and `MATH-500`; several other benchmarks are relatively flat across stages and ranks.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    OUTPUT_MD.write_text(render())
    print(OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
