import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_JSON = ROOT / "results" / "post_dpo_benchmarks.json"
REPORT_MD = ROOT / "results" / "benchmark_claim_audit.md"

MODEL_ORDER = ["qwen_0.8b", "nemotron_4b", "nemotron_30b"]
RANKS = [1, 2, 8, 128, 1024, 3072]
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

MODE_THRESHOLDS = {
    "autonomous": {"gsm8k": 40, "math500": 40, "arc": 40, "mmlu": 40, "mbpp": 20, "humaneval": 20, "truthfulqa_mc": 40, "mmlu_pro": 40, "gpqa": 40},
    "quick": {"gsm8k": 100, "math500": 100, "arc": 100, "mmlu": 100, "mbpp": 50, "humaneval": 50, "truthfulqa_mc": 100, "mmlu_pro": 100, "gpqa": 100},
    "paper": {"gsm8k": 250, "math500": 250, "arc": 500, "mmlu": 500, "mbpp": 200, "humaneval": 164, "truthfulqa_mc": 250, "mmlu_pro": 500, "gpqa": 250},
    "research": {"gsm8k": 500, "math500": 500, "arc": 1000, "mmlu": 1000, "mbpp": 500, "humaneval": 164, "truthfulqa_mc": 684, "mmlu_pro": 1000, "gpqa": 448},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit whether benchmark evidence is strong enough for paper claims.")
    parser.add_argument("--target-mode", choices=sorted(MODE_THRESHOLDS), default="paper")
    parser.add_argument("--output", type=Path, default=REPORT_MD)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    return parser.parse_args()


def load_results():
    if not RESULTS_JSON.exists():
        return {}
    with RESULTS_JSON.open("r") as handle:
        return json.load(handle).get("benchmarks", {})


def stage_artifacts(model_key: str, rank: int):
    mapping = {}
    if rank == 0:
        mapping["base"] = None
        return mapping

    for stage in ("math", "science", "code"):
        p = OUTPUT_DIR / f"{model_key}_{stage}_SFT_rank{rank}" / "adapter_config.json"
        if p.exists():
            mapping[f"{stage}_sft"] = p.parent

    merged_root = OUTPUT_DIR / f"{model_key}_merged_DARE_rank{rank}"
    if (merged_root / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root
    elif (merged_root / "merged_sft" / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root / "merged_sft"

    dpo_root = OUTPUT_DIR / f"{model_key}_math_DPO_rank{rank}"
    if (dpo_root / "adapter_config.json").exists():
        mapping["dpo"] = dpo_root

    return mapping


def expected_matrix(selected_models=None):
    rows = []
    model_keys = selected_models or MODEL_ORDER
    for model_key in model_keys:
        rows.append((model_key, 0, "base"))
        for rank in RANKS:
            for stage in stage_artifacts(model_key, rank):
                rows.append((model_key, rank, stage))
    return rows


def benchmark_key(model_key: str, rank: int, stage: str, benchmark: str):
    return f"{model_key}|rank={rank}|stage={stage}|benchmark={benchmark}"


def ci_width(entry: dict):
    low = entry.get("ci95_low")
    high = entry.get("ci95_high")
    if low is None or high is None:
        return None
    return high - low


def fmt_pct(value):
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def fmt_float(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def claim_status(completion_ratio: float, mean_ci_width: float | None, target_mode: str):
    if completion_ratio < 0.6:
        return "Not defendable"
    if target_mode in {"paper", "research"} and completion_ratio < 0.9:
        return "Partial only"
    if mean_ci_width is None:
        return "Partial only"
    if target_mode == "research" and mean_ci_width > 0.12:
        return "Partial only"
    if target_mode == "paper" and mean_ci_width > 0.18:
        return "Partial only"
    return "Reasonable"


def render_report(target_mode: str, matrix_rows, results):
    thresholds = MODE_THRESHOLDS[target_mode]
    expected_total = len(matrix_rows) * len(BENCHMARK_ORDER)
    complete = 0
    partial = 0
    missing = 0
    ci_widths = []
    per_row = defaultdict(dict)

    for model_key, rank, stage in matrix_rows:
        for benchmark in BENCHMARK_ORDER:
            key = benchmark_key(model_key, rank, stage, benchmark)
            entry = results.get(key)
            threshold = thresholds[benchmark]
            status = "missing"
            if entry:
                samples = int(entry.get("samples", 0) or 0)
                gen_file = entry.get("generation_file")
                gen_ok = bool(gen_file) and Path(gen_file).exists()
                if samples >= threshold and entry.get("metric") != "error" and gen_ok:
                    status = "complete"
                    complete += 1
                    width = ci_width(entry)
                    if width is not None:
                        ci_widths.append(width)
                else:
                    status = "partial"
                    partial += 1
            else:
                missing += 1
            per_row[(model_key, rank, stage)][benchmark] = status

    completion_ratio = complete / expected_total if expected_total else 0.0
    mean_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else None
    status = claim_status(completion_ratio, mean_ci_width, target_mode)

    lines = [
        "# Benchmark Claim Audit",
        "",
        f"- Target mode: `{target_mode}`",
        f"- Expected benchmark cells: `{expected_total}`",
        f"- Complete cells: `{complete}`",
        f"- Partial cells: `{partial}`",
        f"- Missing cells: `{missing}`",
        f"- Completion ratio: `{completion_ratio:.1%}`",
        f"- Mean CI width: `{fmt_float(mean_ci_width)}`",
        f"- Static claim readiness: `{status}`",
        "",
        "## Verdict",
        "",
    ]

    if status == "Reasonable":
        lines.append("- Static benchmark evidence is strong enough for careful stage/rank claims on the implemented benchmark suite.")
    elif status == "Partial only":
        lines.append("- Evidence is usable for selective claims, but not broad world-class claims yet.")
    else:
        lines.append("- Evidence is still pilot-grade. Claims beyond directional observations are not defendable.")

    lines += [
        "",
        "## Coverage Table",
        "",
        "| Model | Rank | Stage | " + " | ".join(BENCHMARK_LABELS[b] for b in BENCHMARK_ORDER) + " |",
        "|-------|------|-------|" + "|".join("-" * max(3, len(BENCHMARK_LABELS[b])) for b in BENCHMARK_ORDER) + "|",
    ]

    def row_sort(item):
        model, rank, stage = item
        mi = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
        si = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
        return (mi, rank, si, model, stage)

    icon = {"complete": "OK", "partial": "PART", "missing": "MISS"}
    for row_key in sorted(per_row, key=row_sort):
        model, rank, stage = row_key
        vals = [icon[per_row[row_key][b]] for b in BENCHMARK_ORDER]
        lines.append(f"| {model} | {rank} | {stage} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## What This Supports",
        "",
        "- `OK` across most rows is enough for rank-by-stage static comparisons on the implemented suite.",
        "- `PART` means the cell exists but is below the requested sample threshold, missing generation files, or otherwise incomplete.",
        "- This audit does not certify agentic claims. Those still require repeated-run agent benchmarks.",
    ]

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    results = load_results()
    matrix_rows = expected_matrix(args.models)
    report = render_report(args.target_mode, matrix_rows, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
