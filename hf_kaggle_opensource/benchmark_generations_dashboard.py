import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from benchmark_claim_audit import (
    BENCHMARK_LABELS,
    BENCHMARK_ORDER,
    MODE_THRESHOLDS,
    MODEL_ORDER,
    STAGE_ORDER,
    expected_matrix,
    load_results as load_claim_results,
    render_report,
)


ROOT = Path(__file__).resolve().parent
GENERATIONS_DIR = ROOT / "results" / "benchmark_generations"
OUTPUT_MD = ROOT / "results" / "benchmark_generations_dashboard.md"

@dataclass
class Score:
    correct: int = 0
    total: int = 0

    @property
    def pct(self) -> float | None:
        if self.total == 0:
            return None
        return 100.0 * self.correct / self.total


@dataclass
class RowCoverage:
    complete: int = 0
    partial: int = 0
    missing: int = 0
    samples: int = 0
    target: int = 0
    ci_width_sum: float = 0.0
    ci_width_count: int = 0

    @property
    def started(self) -> int:
        return self.complete + self.partial

    @property
    def mean_ci_width(self) -> float | None:
        if self.ci_width_count == 0:
            return None
        return self.ci_width_sum / self.ci_width_count


def parse_args():
    parser = argparse.ArgumentParser(description="Live dashboard for benchmark generation JSONL files.")
    parser.add_argument("--watch", action="store_true", help="Refresh continuously.")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--output", type=Path, default=OUTPUT_MD, help="Markdown output path.")
    parser.add_argument("--target-mode", choices=["autonomous", "quick", "paper", "research"], default="paper")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    return parser.parse_args()


def parse_generation_file(path: Path):
    name = path.name
    if not name.endswith(".jsonl") or "_rank" not in name:
        return None
    stem = name[:-6]
    model, rest = stem.split("_rank", 1)
    rank_str, tail = rest.split("_", 1)
    benchmark = None
    for candidate in sorted(BENCHMARK_ORDER, key=len, reverse=True):
        suffix = "_" + candidate
        if tail.endswith(suffix):
            benchmark = candidate
            stage = tail[: -len(suffix)]
            break
    if benchmark is None or not stage:
        return None
    score = Score()
    for line in path.open("r"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Allow partially written final line while a worker is still appending.
            continue
        if "correct" not in row:
            continue
        score.total += 1
        if bool(row["correct"]):
            score.correct += 1
    return {
        "model": model,
        "rank": int(rank_str),
        "stage": stage,
        "benchmark": benchmark,
        "score": score,
        "path": path,
    }


def collect_scores(selected_models=None):
    by_row = defaultdict(lambda: {"benchmarks": {}, "overall": Score(), "files": 0})
    for path in sorted(GENERATIONS_DIR.glob("*.jsonl")):
        parsed = parse_generation_file(path)
        if not parsed:
            continue
        if selected_models and parsed["model"] not in selected_models:
            continue
        row_key = (parsed["model"], parsed["rank"], parsed["stage"])
        row = by_row[row_key]
        row["benchmarks"][parsed["benchmark"]] = parsed["score"]
        row["overall"].correct += parsed["score"].correct
        row["overall"].total += parsed["score"].total
        row["files"] += 1
    return by_row


def wilson_ci_width(entry: dict) -> float | None:
    low = entry.get("ci95_low")
    high = entry.get("ci95_high")
    if low is None or high is None:
        return None
    return float(high) - float(low)


def collect_results_meta(target_mode: str, selected_models=None):
    thresholds = MODE_THRESHOLDS[target_mode]
    results = load_claim_results()
    matrix_rows = set(expected_matrix(selected_models))
    rows = set(matrix_rows)

    for key in results:
        try:
            model, rank_part, stage_part, benchmark_part = key.split("|")
            rank = int(rank_part.split("=")[1])
            stage = stage_part.split("=", 1)[1]
            benchmark = benchmark_part.split("=", 1)[1]
        except Exception:
            continue
        if selected_models and model not in selected_models:
            continue
        if benchmark not in BENCHMARK_ORDER:
            continue
        rows.add((model, rank, stage))

    coverage = defaultdict(RowCoverage)
    cell_progress = defaultdict(dict)
    for model, rank, stage in rows:
        row_cov = coverage[(model, rank, stage)]
        for benchmark in BENCHMARK_ORDER:
            target = thresholds[benchmark]
            key = f"{model}|rank={rank}|stage={stage}|benchmark={benchmark}"
            entry = results.get(key)
            samples = int(entry.get("samples", 0) or 0) if entry else 0
            gen_file = entry.get("generation_file") if entry else None
            gen_ok = bool(gen_file) and Path(gen_file).exists()
            metric_ok = bool(entry) and entry.get("metric") != "error"
            status = "missing"
            if entry:
                if samples >= target and metric_ok and gen_ok:
                    status = "complete"
                    row_cov.complete += 1
                    width = wilson_ci_width(entry)
                    if width is not None:
                        row_cov.ci_width_sum += width
                        row_cov.ci_width_count += 1
                else:
                    status = "partial"
                    row_cov.partial += 1
            else:
                row_cov.missing += 1

            row_cov.samples += min(samples, target)
            row_cov.target += target
            cell_progress[(model, rank, stage)][benchmark] = {
                "status": status,
                "samples": samples,
                "target": target,
                "metric": entry.get("metric") if entry else None,
            }

    return coverage, cell_progress


def fmt_pct(score: Score | None):
    if score is None or score.total == 0 or score.pct is None:
        return "-"
    return f"{score.pct:5.1f}%"


def fmt_ratio(samples: int, target: int):
    return f"{samples}/{target}"


def fmt_ci_width(width: float | None):
    if width is None:
        return "-"
    return f"{100.0 * width:.1f}pp"


def progress_cell(progress: dict | None):
    if not progress:
        return "MISS"
    status = progress["status"]
    samples = progress["samples"]
    target = progress["target"]
    if status == "complete":
        return f"OK {samples}/{target}"
    if status == "partial":
        return f"PART {samples}/{target}"
    return "MISS"


def sort_key(item):
    model, rank, stage = item[0]
    model_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return (model_idx, rank, stage_idx, model, stage)


def audit_summary_lines(target_mode: str, selected_models=None):
    try:
        report = render_report(target_mode, expected_matrix(selected_models), load_claim_results())
        lines = report.splitlines()
        summary = []
        for line in lines[2:9]:
            if line.strip():
                summary.append(f"- {line.lstrip('- ').strip()}")
        return summary
    except Exception as exc:
        return [f"- Audit summary unavailable: {exc}"]


def render_markdown(scores, coverage, cell_progress, target_mode: str, selected_models=None):
    lines = [
        "# Benchmark Generations Dashboard",
        "",
        f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Source dir: `{GENERATIONS_DIR}`",
        f"- Target mode: `{target_mode}`",
        f"- Models: `{', '.join(selected_models) if selected_models else 'all'}`",
        "",
        "## Audit Snapshot",
        "",
        *audit_summary_lines(target_mode, selected_models),
        "",
        "## Coverage By Row",
        "",
        "| Model | Rank | Stage | Ready | Started | Missing | Evidence | Mean CI Width |",
        "|-------|------|-------|-------|---------|---------|----------|---------------|",
    ]
    row_keys = sorted(set(scores) | set(coverage), key=lambda item: sort_key((item, None)))
    for model, rank, stage in row_keys:
        cov = coverage.get((model, rank, stage), RowCoverage())
        lines.append(
            f"| {model} | {rank} | {stage} | "
            f"{cov.complete}/{len(BENCHMARK_ORDER)} | "
            f"{cov.started}/{len(BENCHMARK_ORDER)} | "
            f"{cov.missing} | "
            f"{fmt_ratio(cov.samples, cov.target)} | "
            f"{fmt_ci_width(cov.mean_ci_width)} |"
        )

    lines += [
        "",
        "## Score Table",
        "",
        "| Model | Rank | Method | Files | Overall | " + " | ".join(BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER) + " |",
        "|-------|------|--------|-------|---------|" + "|".join("-" * max(3, len(BENCHMARK_LABELS[name])) for name in BENCHMARK_ORDER) + "|",
    ]
    for (model, rank, stage), row in sorted(scores.items(), key=sort_key):
        vals = [fmt_pct(row["benchmarks"].get(name)) for name in BENCHMARK_ORDER]
        lines.append(
            f"| {model} | {rank} | {stage} | {row['files']} | {fmt_pct(row['overall'])} | "
            + " | ".join(vals)
            + " |"
        )

    lines += [
        "",
        "## Sample Progress",
        "",
        "| Model | Rank | Stage | " + " | ".join(BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER) + " |",
        "|-------|------|-------|" + "|".join("-" * max(3, len(BENCHMARK_LABELS[name])) for name in BENCHMARK_ORDER) + "|",
    ]
    for model, rank, stage in row_keys:
        vals = [progress_cell(cell_progress.get((model, rank, stage), {}).get(name)) for name in BENCHMARK_ORDER]
        lines.append(f"| {model} | {rank} | {stage} | " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def render_terminal(scores, coverage, target_mode: str, selected_models=None):
    headers = ["Model", "Rank", "Method", "Files", "Overall"] + [BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER]
    rows = []
    for (model, rank, stage), row in sorted(scores.items(), key=sort_key):
        rows.append(
            [
                model,
                str(rank),
                stage,
                str(row["files"]),
                fmt_pct(row["overall"]),
                *[fmt_pct(row["benchmarks"].get(name)) for name in BENCHMARK_ORDER],
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values):
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    out = [
        f"Benchmark Generations Dashboard  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Target mode: {target_mode}",
        f"Models: {', '.join(selected_models) if selected_models else 'all'}",
        "",
        "Audit snapshot:",
        *[line[2:] if line.startswith("- ") else line for line in audit_summary_lines(target_mode, selected_models)],
        "",
        "Coverage by row:",
    ]
    coverage_headers = ["Model", "Rank", "Stage", "Ready", "Started", "Evidence", "Mean CI"]
    coverage_rows = []
    for row_key in sorted(set(scores) | set(coverage), key=lambda item: sort_key((item, None))):
        model, rank, stage = row_key
        cov = coverage.get(row_key, RowCoverage())
        coverage_rows.append(
            [
                model,
                str(rank),
                stage,
                f"{cov.complete}/{len(BENCHMARK_ORDER)}",
                f"{cov.started}/{len(BENCHMARK_ORDER)}",
                fmt_ratio(cov.samples, cov.target),
                fmt_ci_width(cov.mean_ci_width),
            ]
        )
    coverage_widths = [len(h) for h in coverage_headers]
    for row in coverage_rows:
        for idx, value in enumerate(row):
            coverage_widths[idx] = max(coverage_widths[idx], len(value))

    def fmt_cov_row(values):
        return "  ".join(value.ljust(coverage_widths[idx]) for idx, value in enumerate(values))

    out += [
        fmt_cov_row(coverage_headers),
        fmt_cov_row(["-" * width for width in coverage_widths]),
    ]
    out.extend(fmt_cov_row(row) for row in coverage_rows)
    out += [
        "",
        fmt_row(headers),
        fmt_row(["-" * width for width in widths]),
    ]
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out) + "\n"


def write_output(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def run_once(output_path: Path, target_mode: str, selected_models=None):
    scores = collect_scores(selected_models)
    coverage, cell_progress = collect_results_meta(target_mode, selected_models)
    md = render_markdown(scores, coverage, cell_progress, target_mode, selected_models)
    write_output(output_path, md)
    return render_terminal(scores, coverage, target_mode, selected_models)


def main():
    args = parse_args()
    if not args.watch:
        print(run_once(args.output, args.target_mode, args.models), end="")
        return 0

    while True:
        terminal = run_once(args.output, args.target_mode, args.models)
        clear = "cls" if os.name == "nt" else "clear"
        subprocess = shutil.which(clear)
        if subprocess:
            os.system(clear)
        print(terminal, end="", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
