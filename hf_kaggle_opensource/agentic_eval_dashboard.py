import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "results" / "agentic_eval" / "planbench_results.json"
SUMMARY_MD = ROOT / "results" / "agentic_eval" / "planbench_dashboard.md"


def load_results():
    if not RESULTS_JSON.exists():
        return {"benchmarks": {}}
    try:
        return json.loads(RESULTS_JSON.read_text())
    except Exception:
        return {"benchmarks": {}}


def render(results: dict, models: list[str] | None):
    rows = []
    counts = {"complete": 0, "error": 0}
    for key, value in sorted(results.get("benchmarks", {}).items()):
        parts = dict(part.split("=", 1) for part in key.split("|"))
        if models and parts["model"] not in models:
            continue
        metric = value.get("metric")
        if metric == "error":
            counts["error"] += 1
        else:
            counts["complete"] += 1
        rows.append((parts, value))

    lines = [
        "# Agentic Eval Dashboard",
        "",
        "## Audit Snapshot",
        "",
        f"- Complete cells: {counts['complete']}",
        f"- Error cells: {counts['error']}",
        f"- Total tracked cells: {counts['complete'] + counts['error']}",
        "",
        "## Score Table",
        "",
        "| Model | Rank | Stage | Domain | Task | Score | 95% CI | Correct / Total | Status |",
        "|-------|------|-------|--------|------|-------|---------|-----------------|--------|",
    ]
    for parts, value in rows:
        score = value.get("score")
        score_str = "-" if score is None else f"{score:.3f}"
        ci_low = value.get("ci95_low")
        ci_high = value.get("ci95_high")
        ci_str = "-" if ci_low is None or ci_high is None else f"[{ci_low:.3f}, {ci_high:.3f}]"
        status = "ERR" if value.get("metric") == "error" else "OK"
        lines.append(
            f"| {parts['model']} | {parts['rank']} | {parts['stage']} | {parts['domain']} | {parts['task']} | "
            f"{score_str} | {ci_str} | {value.get('correct', 0)} / {value.get('total', 0)} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Render a markdown dashboard for agentic eval results.")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    while True:
        rendered = render(load_results(), args.models)
        SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_MD.write_text(rendered)
        if not args.watch:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
