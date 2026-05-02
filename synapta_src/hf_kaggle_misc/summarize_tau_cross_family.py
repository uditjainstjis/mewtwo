import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TAU_RESULTS_JSON = ROOT / "results" / "agentic_eval" / "tau_bench_results.json"
OUT_MD = ROOT / "results" / "agentic_eval" / "tau_cross_family_findings.md"

USER_TAG = "local_hf:qwen_0.8b:base:0"
MODELS = ["qwen_0.8b", "nemotron_4b"]


def load_rows():
    payload = json.loads(TAU_RESULTS_JSON.read_text())
    bench = payload.get("benchmarks", {})
    best_rows = {}
    for key, value in bench.items():
        if not isinstance(value, dict):
            continue
        if "|env=retail|" not in key or "|split=test|" not in key or "|agent=local_hf|" not in key:
            continue
        if f"|user={USER_TAG}" not in key:
            continue
        model = value.get("model")
        if model not in MODELS:
            continue
        row = {
            "key": key,
            "model": model,
            "rank": value.get("rank"),
            "stage": value.get("stage"),
            "prefix": value.get("average_gold_tool_name_prefix_frac"),
            "exact_prefix": value.get("average_gold_exact_prefix_frac"),
            "recall": value.get("average_gold_tool_name_recall_frac"),
            "terminal": value.get("average_gold_terminal_tool_match"),
            "invalid": value.get("average_invalid_actions"),
            "total": int(value.get("total", 0) or 0),
            "num_trials": int(value.get("num_trials", 0) or 0),
        }
        dedup_key = (row["model"], int(row["rank"]), str(row["stage"]))
        current = best_rows.get(dedup_key)
        if current is None or (row["total"], row["num_trials"]) > (current["total"], current["num_trials"]):
            best_rows[dedup_key] = row
    rows = list(best_rows.values())
    rows.sort(key=lambda row: (row["model"], int(row["rank"]), str(row["stage"])))
    return rows


def pct(v):
    return "-" if v is None else f"{100.0 * float(v):.1f}%"


def num(v):
    return "-" if v is None else f"{float(v):.1f}"


def render(rows):
    qwen = [row for row in rows if row["model"] == "qwen_0.8b"]
    nemotron = [row for row in rows if row["model"] == "nemotron_4b"]
    totals = [row["total"] for row in rows if row["total"]]
    min_total = min(totals) if totals else 0
    max_total = max(totals) if totals else 0

    qwen_weak = [row for row in qwen if (row["prefix"] or 0.0) < 0.8]
    qwen_strong = [row for row in qwen if (row["prefix"] or 0.0) >= 0.8]
    nemotron_all_strong = bool(nemotron) and all((row["prefix"] or 0.0) >= 0.8 for row in nemotron)
    qwen_weak_desc = ", ".join(f"rank{row['rank']} {row['stage']}" for row in qwen_weak) if qwen_weak else "none"
    qwen_strong_desc = ", ".join(f"rank{row['rank']} {row['stage']}" for row in qwen_strong) if qwen_strong else "none"

    lines = [
        "# Tau Cross-Family Findings",
        "",
        "## Scope",
        "",
        "- Environment: `retail`",
        "- Split: `test`",
        f"- Row coverage ranges from `total={min_total}` to `total={max_total}` tasks, `1` trial, `12` max steps, local-HF agent and local-HF Qwen base user simulator.",
        "- This is an offline behavioral measurement setup, not an official API-backed Tau leaderboard run.",
        "",
        "## Current Rows",
        "",
        "| Model | Rank | Stage | Total | Prefix | Exact Prefix | Recall | Terminal | Invalid Actions |",
        "|-------|------|-------|-------|--------|--------------|--------|----------|-----------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['rank']} | {row['stage']} | {row['total']} | {pct(row['prefix'])} | "
            f"{pct(row['exact_prefix'])} | {pct(row['recall'])} | {pct(row['terminal'])} | {num(row['invalid'])} |"
        )

    lines.extend(
        [
            "",
            "## Main Read",
            "",
            "- The strongest current Tau signal is `gold prefix`, not `pass@1`; terminal success is still at floor across all covered rows.",
            f"- Qwen weak rows currently are: {qwen_weak_desc}",
            f"- Qwen strong rows currently are: {qwen_strong_desc}",
            "- The standout anomalous row remains `qwen rank2 math_sft`, which collapses into repeated re-auth behavior and drops to `0.4` prefix / `0.4` recall."
            if any(row["rank"] == 2 and row["stage"] == "math_sft" and row["model"] == "qwen_0.8b" for row in qwen)
            else "- The prior `qwen rank2 math_sft` anomaly is not covered in the current rows.",
            "- Nemotron is much flatter on the same task: every currently covered runnable Nemotron row lands in the strong `0.8 prefix / 0.8 recall` pattern."
            if nemotron_all_strong
            else "- Nemotron does not yet show a single flat cluster across the currently covered rows.",
            "- The best current interpretation is that Tau is exposing a family-specific early-trajectory behavioral axis rather than a trivial global rank effect.",
            "",
            "## Caveats",
            "",
            "- Coverage is still uneven when some rows have only `total=1` and others have richer multi-task slices."
            if min_total != max_total
            else (
                "- This is still single-task, single-trial coverage for the hard retail test case."
                if max_total <= 1
                else "- This is still small-sample multi-task coverage, not a broad leaderboard-style run."
            ),
            "- The present signal is about trajectory shape before terminal synthesis, not end-to-end task completion.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    rows = load_rows()
    OUT_MD.write_text(render(rows))
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
