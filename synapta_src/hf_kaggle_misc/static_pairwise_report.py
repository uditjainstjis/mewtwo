import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATIONS_DIR = ROOT / "results" / "benchmark_generations"
OUTPUT_MD = ROOT / "results" / "static_pairwise_report.md"

BENCHMARKS = ["gsm8k", "math500", "arc", "mmlu", "truthfulqa_mc", "mmlu_pro", "gpqa"]
MODEL = "qwen_0.8b"


def generation_path(rank: int, stage: str, benchmark: str) -> Path:
    return GENERATIONS_DIR / f"{MODEL}_rank{rank}_{stage}_{benchmark}.jsonl"


def key_for_row(row: dict):
    if "task_id" in row:
        return ("task_id", row["task_id"])
    if "question" in row:
        return ("question", row["question"])
    if "entry_point" in row:
        return ("entry_point", row["entry_point"])
    return ("raw", json.dumps(row, sort_keys=True))


def load_rows(path: Path):
    rows = {}
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[key_for_row(row)] = row
    return rows


def binom_two_sided_pvalue(k: int, n: int, p: float = 0.5):
    if n == 0:
        return 1.0
    observed = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if prob <= observed + 1e-15:
            total += prob
    return min(1.0, total)


def paired_result(left_path: Path, right_path: Path):
    left_rows = load_rows(left_path)
    right_rows = load_rows(right_path)
    shared = sorted(set(left_rows) & set(right_rows))
    both_correct = left_only = right_only = both_wrong = 0
    for key in shared:
        lc = bool(left_rows[key].get("correct"))
        rc = bool(right_rows[key].get("correct"))
        if lc and rc:
            both_correct += 1
        elif lc and not rc:
            left_only += 1
        elif not lc and rc:
            right_only += 1
        else:
            both_wrong += 1
    discordant = left_only + right_only
    p_value = binom_two_sided_pvalue(min(left_only, right_only), discordant, p=0.5)
    left_acc = (both_correct + left_only) / len(shared) if shared else None
    right_acc = (both_correct + right_only) / len(shared) if shared else None
    return {
        "shared": len(shared),
        "both_correct": both_correct,
        "left_only": left_only,
        "right_only": right_only,
        "both_wrong": both_wrong,
        "left_acc": left_acc,
        "right_acc": right_acc,
        "p_value": p_value,
    }


def fmt_pct(value):
    return "-" if value is None else f"{100.0 * value:.1f}%"


def build_rows():
    comparisons = []
    for bench in BENCHMARKS:
        comparisons.append(
            (
                f"{bench}: rank1 merged_dare vs dpo",
                generation_path(1, "merged_dare", bench),
                generation_path(1, "dpo", bench),
            )
        )
        comparisons.append(
            (
                f"{bench}: rank2 math_sft vs base",
                generation_path(2, "math_sft", bench),
                generation_path(0, "base", bench),
            )
        )
        comparisons.append(
            (
                f"{bench}: low-rank rank2 math_sft vs high-rank rank1024 math_sft",
                generation_path(2, "math_sft", bench),
                generation_path(1024, "math_sft", bench),
            )
        )
    rows = []
    for label, left, right in comparisons:
        if not left.exists() or not right.exists():
            continue
        rows.append((label, paired_result(left, right)))
    return rows


def render():
    rows = build_rows()
    lines = [
        "# Static Pairwise Report",
        "",
        "## Purpose",
        "",
        "- Compare key Qwen static rows using paired item-level correctness.",
        "- Use exact McNemar-style binomial testing on discordant pairs.",
        "- Keep claims conservative: small score gaps with large p-values are not real wins.",
        "",
        "| Comparison | Shared | Left Acc | Right Acc | Left-only | Right-only | p-value |",
        "|------------|--------|----------|-----------|-----------|------------|---------|",
    ]
    for label, result in rows:
        lines.append(
            f"| {label} | {result['shared']} | {fmt_pct(result['left_acc'])} | {fmt_pct(result['right_acc'])} | "
            f"{result['left_only']} | {result['right_only']} | {result['p_value']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    OUTPUT_MD.write_text(render())
    print(OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
