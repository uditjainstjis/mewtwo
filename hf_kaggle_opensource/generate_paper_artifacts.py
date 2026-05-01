import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = RESULTS / "paper_artifacts"
GEOM_JSON = RESULTS / "qwen_geometry_behavior.json"
STATIC_JSON = RESULTS / "post_dpo_benchmarks.json"
PLANBENCH_JSON = RESULTS / "agentic_eval" / "planbench_results.json"
TAU_JSON = RESULTS / "agentic_eval" / "tau_bench_results.json"

STAGE_ORDER = ["math_sft", "science_sft", "code_sft", "merged_dare", "dpo"]
STAGE_LABELS = {
    "math_sft": "Math SFT",
    "science_sft": "Science SFT",
    "code_sft": "Code SFT",
    "merged_dare": "Merged DARE",
    "dpo": "DPO",
}
STAGE_COLORS = {
    "math_sft": "#0b6e4f",
    "science_sft": "#8a5a00",
    "code_sft": "#8f1d21",
    "merged_dare": "#005f99",
    "dpo": "#5f0f99",
}
BENCHMARKS = ["gsm8k", "math500", "mmlu", "mbpp"]


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_inventory():
    payload = load_json(GEOM_JSON)
    rows = []
    for _, row in payload["inventory"].items():
        rows.append(row)
    rows.sort(key=lambda r: (r["rank"], STAGE_ORDER.index(r["stage"])))
    return rows, payload["correlations"]


def fmt_pct(value):
    if value is None:
        return ""
    return f"{100.0 * value:.1f}"


def fmt_num(value, nd=3):
    if value is None:
        return ""
    return f"{value:.{nd}f}"


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_md_table(path: Path, header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def scale(values, lo, hi):
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [(lo + hi) / 2.0 for _ in values]
    return [lo + (v - low) * (hi - lo) / (high - low) for v in values]


def svg_begin(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "DejaVu Sans", sans-serif; fill: #111; }',
        '.small { font-size: 11px; }',
        '.axis { stroke: #222; stroke-width: 1.2; }',
        '.grid { stroke: #ddd; stroke-width: 1; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.label { font-size: 13px; font-weight: 600; }',
        '</style>',
        f'<text class="title" x="24" y="28">{escape(title)}</text>',
    ]


def svg_end():
    return ["</svg>"]


def add_legend(lines, items, x, y):
    step = 20
    for idx, (label, color) in enumerate(items):
        yy = y + idx * step
        lines.append(f'<rect x="{x}" y="{yy-9}" width="12" height="12" fill="{color}" />')
        lines.append(f'<text class="small" x="{x+18}" y="{yy+1}">{escape(label)}</text>')


def scatter_svg(path: Path, points, title, x_label, y_label, annotate=None):
    width, height = 860, 560
    margin = {"l": 80, "r": 180, "t": 60, "b": 70}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    px = scale(xs, margin["l"], margin["l"] + plot_w)
    py = scale(ys, margin["t"] + plot_h, margin["t"])
    lines = svg_begin(width, height, title)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = margin["t"] + plot_h * frac
        xx = margin["l"] + plot_w * frac
        lines.append(f'<line class="grid" x1="{margin["l"]}" y1="{yy}" x2="{margin["l"]+plot_w}" y2="{yy}" />')
        lines.append(f'<line class="grid" x1="{xx}" y1="{margin["t"]}" x2="{xx}" y2="{margin["t"]+plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]+plot_h}" x2="{margin["l"]+plot_w}" y2="{margin["t"]+plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{margin["t"]+plot_h}" />')
    for p, x, y in zip(points, px, py):
        color = STAGE_COLORS.get(p["stage"], "#333")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#111" stroke-width="0.8" />')
        if annotate and annotate(p):
            text = escape(annotate(p))
            lines.append(f'<text class="small" x="{x+7:.1f}" y="{y-7:.1f}">{text}</text>')
    add_legend(lines, [(STAGE_LABELS[s], STAGE_COLORS[s]) for s in STAGE_ORDER], width - 150, 110)
    lines.append(f'<text class="label" x="{margin["l"] + plot_w/2:.1f}" y="{height-20}" text-anchor="middle">{escape(x_label)}</text>')
    lines.append(f'<text class="label" transform="translate(20 {margin["t"] + plot_h/2:.1f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    path.write_text("\n".join(lines + svg_end()) + "\n")


def line_svg(path: Path, series, title, x_label, y_label):
    width, height = 860, 560
    margin = {"l": 80, "r": 180, "t": 60, "b": 70}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    all_x = sorted({x for _, pts in series for x, _ in pts})
    all_y = [y for _, pts in series for _, y in pts if y is not None]
    py_map = scale(all_y, margin["t"] + plot_h, margin["t"])
    unique_y = {v: p for v, p in zip(all_y, py_map)}
    x_pos = scale(all_x, margin["l"], margin["l"] + plot_w)
    x_lookup = {v: p for v, p in zip(all_x, x_pos)}
    lines = svg_begin(width, height, title)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = margin["t"] + plot_h * frac
        lines.append(f'<line class="grid" x1="{margin["l"]}" y1="{yy}" x2="{margin["l"]+plot_w}" y2="{yy}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]+plot_h}" x2="{margin["l"]+plot_w}" y2="{margin["t"]+plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{margin["t"]+plot_h}" />')
    for rank, xpos in x_lookup.items():
        lines.append(f'<text class="small" x="{xpos:.1f}" y="{margin["t"]+plot_h+18}" text-anchor="middle">{rank}</text>')
    legend = []
    for name, pts in series:
        color = STAGE_COLORS[name]
        legend.append((STAGE_LABELS[name], color))
        coords = []
        for x, y in pts:
            if y is None:
                continue
            coords.append((x_lookup[x], unique_y[y]))
        if not coords:
            continue
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{poly}" />')
        for x, y in coords:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />')
    add_legend(lines, legend, width - 150, 110)
    lines.append(f'<text class="label" x="{margin["l"] + plot_w/2:.1f}" y="{height-20}" text-anchor="middle">{escape(x_label)}</text>')
    lines.append(f'<text class="label" transform="translate(20 {margin["t"] + plot_h/2:.1f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    path.write_text("\n".join(lines + svg_end()) + "\n")


def grouped_bar_svg(path: Path, groups, title, metrics):
    width, height = 1080, 620
    margin = {"l": 90, "r": 40, "t": 70, "b": 190}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    all_values = [group[m] for group in groups for m in metrics if group[m] is not None]
    max_y = max(all_values) if all_values else 1.0
    colors = ["#005f99", "#0b6e4f", "#b85c00"]
    lines = svg_begin(width, height, title)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = margin["t"] + plot_h * frac
        label = max_y * (1.0 - frac)
        lines.append(f'<line class="grid" x1="{margin["l"]}" y1="{yy}" x2="{margin["l"]+plot_w}" y2="{yy}" />')
        lines.append(f'<text class="small" x="{margin["l"]-8}" y="{yy+4}" text-anchor="end">{label:.2f}</text>')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]+plot_h}" x2="{margin["l"]+plot_w}" y2="{margin["t"]+plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{margin["t"]+plot_h}" />')
    group_w = plot_w / max(len(groups), 1)
    bar_w = group_w / (len(metrics) + 1)
    for i, group in enumerate(groups):
        gx = margin["l"] + i * group_w
        for j, metric in enumerate(metrics):
            value = group[metric]
            if value is None:
                continue
            h = plot_h * value / max_y if max_y else 0
            x = gx + j * bar_w + bar_w * 0.2
            y = margin["t"] + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{h:.1f}" fill="{colors[j]}" />')
        label_x = gx + group_w / 2
        lines.append(f'<text class="small" transform="translate({label_x:.1f} {height-70}) rotate(-40)" text-anchor="end">{escape(group["label"])}</text>')
    add_legend(lines, list(zip(metrics, colors)), width - 190, 110)
    path.write_text("\n".join(lines + svg_end()) + "\n")


def build_static_summary(inventory, static_payload):
    bench = static_payload["benchmarks"]
    rows = []
    for row in inventory:
        rank = row["rank"]
        stage = row["stage"]
        csv_row = {
            "rank": rank,
            "stage": stage,
            "static_overall_pct": fmt_pct(row["static"].get("overall")),
            "static_math_mean_pct": fmt_pct(row["static"].get("math_mean")),
        }
        for benchmark in BENCHMARKS:
            key = f"qwen_0.8b|rank={rank}|stage={stage}|benchmark={benchmark}"
            score = bench.get(key, {}).get("score")
            csv_row[f"{benchmark}_pct"] = fmt_pct(score)
        rows.append(csv_row)
    return rows


def build_correlation_rows(correlations):
    mapping = {
        "static_overall": "Static Overall",
        "static_math_mean": "Static Math",
        "agentic_t1_blocksworld3": "PlanBench T1",
        "tau_bench_retail_tool_name_prefix": "Tau Prefix",
        "tau_bench_retail_tool_name_recall": "Tau Recall",
        "tau_bench_retail_terminal_tool_match": "Tau Terminal",
    }
    rows = []
    for key, label in mapping.items():
        stats = correlations[key]
        feature, vals = max(
            stats.items(),
            key=lambda kv: abs(kv[1]["spearman"]) if kv[1]["spearman"] is not None else -1,
        )
        rows.append({
            "outcome": label,
            "top_feature": feature,
            "spearman": fmt_num(vals["spearman"]),
            "pearson": fmt_num(vals["pearson"]),
            "n": vals["n"],
        })
    return rows


def build_tau_truncation_rows(tau_payload):
    bench = tau_payload["benchmarks"]
    targets = [
        ("qwen_0.8b", 128, "math_sft"),
        ("qwen_0.8b", 128, "math_sft_trunc8"),
        ("qwen_0.8b", 128, "math_sft_trunc2"),
        ("qwen_0.8b", 1024, "merged_dare"),
        ("qwen_0.8b", 1024, "merged_dare_trunc8"),
        ("qwen_0.8b", 1024, "merged_dare_trunc2"),
        ("qwen_0.8b", 3072, "math_sft"),
        ("qwen_0.8b", 3072, "math_sft_trunc8"),
        ("qwen_0.8b", 3072, "math_sft_trunc2"),
    ]
    rows = []
    for model, rank, stage in targets:
        prefix = f"model={model}|rank={rank}|stage={stage}|env=retail|split=test|trials=1|agent=local_hf|user=local_hf:qwen_0.8b:base:0"
        value = bench.get(prefix, {})
        rows.append({
            "rank": rank,
            "stage": stage,
            "prefix_pct": fmt_pct(value.get("average_gold_tool_name_prefix_frac")),
            "recall_pct": fmt_pct(value.get("average_gold_tool_name_recall_frac")),
            "invalid_actions": fmt_num(value.get("average_invalid_actions"), 1),
            "terminal_pct": fmt_pct(value.get("average_gold_terminal_tool_match")),
        })
    return rows


def build_tau_native_groups(tau_payload):
    bench = tau_payload["benchmarks"]
    targets = [
        (1, "math_sft"),
        (1, "dpo"),
        (2, "math_sft"),
        (2, "dpo"),
        (8, "math_sft"),
        (8, "dpo"),
        (128, "math_sft"),
        (128, "merged_dare"),
        (128, "dpo"),
        (1024, "math_sft"),
        (1024, "merged_dare"),
        (1024, "dpo"),
        (3072, "math_sft"),
        (3072, "merged_dare"),
        (3072, "dpo"),
    ]
    groups = []
    for rank, stage in targets:
        key = f"model=qwen_0.8b|rank={rank}|stage={stage}|env=retail|split=test|trials=1|agent=local_hf|user=local_hf:qwen_0.8b:base:0"
        row = bench[key]
        groups.append({
            "label": f"{rank} {stage}",
            "prefix": row.get("average_gold_tool_name_prefix_frac"),
            "recall": row.get("average_gold_tool_name_recall_frac"),
            "invalid": (row.get("average_invalid_actions") or 0.0) / 12.0,
        })
    return groups


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    inventory, correlations = parse_inventory()
    static_payload = load_json(STATIC_JSON)
    tau_payload = load_json(TAU_JSON)

    static_rows = build_static_summary(inventory, static_payload)
    static_header = ["rank", "stage", "static_overall_pct", "static_math_mean_pct"] + [f"{b}_pct" for b in BENCHMARKS]
    write_csv(ARTIFACTS / "table_static_summary.csv", static_header, [[r[h] for h in static_header] for r in static_rows])
    write_md_table(ARTIFACTS / "table_static_summary.md", static_header, [[r[h] for h in static_header] for r in static_rows])

    corr_rows = build_correlation_rows(correlations)
    corr_header = ["outcome", "top_feature", "spearman", "pearson", "n"]
    write_csv(ARTIFACTS / "table_geometry_correlations.csv", corr_header, [[r[h] for h in corr_header] for r in corr_rows])
    write_md_table(ARTIFACTS / "table_geometry_correlations.md", corr_header, [[r[h] for h in corr_header] for r in corr_rows])

    tau_rows = build_tau_truncation_rows(tau_payload)
    tau_header = ["rank", "stage", "prefix_pct", "recall_pct", "invalid_actions", "terminal_pct"]
    write_csv(ARTIFACTS / "table_tau_truncation.csv", tau_header, [[r[h] for h in tau_header] for r in tau_rows])
    write_md_table(ARTIFACTS / "table_tau_truncation.md", tau_header, [[r[h] for h in tau_header] for r in tau_rows])

    scatter_points = [
        {
            "x": row["geometry"]["rank_utilization"],
            "y": row["static"]["math_mean"],
            "stage": row["stage"],
            "label": f'{row["rank"]} {row["stage"]}',
        }
        for row in inventory
        if row["static"].get("math_mean") is not None
    ]
    scatter_svg(
        ARTIFACTS / "figure_static_math_vs_rank_utilization.svg",
        scatter_points,
        "Static Math vs Rank Utilization",
        "Rank utilization",
        "Static math mean",
        annotate=lambda p: p["label"] if p["label"] in {"2 science_sft", "3072 merged_dare", "3072 dpo"} else None,
    )

    math_series = []
    plan_series = []
    for stage in ["math_sft", "merged_dare", "dpo"]:
        math_pts = []
        plan_pts = []
        for row in inventory:
            if row["stage"] != stage:
                continue
            math_pts.append((row["rank"], row["static"].get("math_mean")))
            plan_pts.append((row["rank"], row.get("agentic_t1_blocksworld3")))
        math_series.append((stage, math_pts))
        plan_series.append((stage, plan_pts))
    line_svg(
        ARTIFACTS / "figure_rank_vs_static_math.svg",
        math_series,
        "Rank vs Static Math",
        "Nominal rank",
        "Static math mean",
    )
    line_svg(
        ARTIFACTS / "figure_rank_vs_planbench.svg",
        plan_series,
        "Rank vs PlanBench T1",
        "Nominal rank",
        "PlanBench T1",
    )

    tau_groups = build_tau_native_groups(tau_payload)
    grouped_bar_svg(
        ARTIFACTS / "figure_tau_collapse_basin.svg",
        tau_groups,
        "Qwen Tau Collapse Basin",
        ["prefix", "recall", "invalid"],
    )

    trunc_groups = []
    for row in tau_rows:
        trunc_groups.append({
            "label": f'{row["rank"]} {row["stage"]}',
            "recall": float(row["recall_pct"]) / 100.0 if row["recall_pct"] else 0.0,
            "invalid_norm": (float(row["invalid_actions"]) / 12.0) if row["invalid_actions"] else 0.0,
        })
    grouped_bar_svg(
        ARTIFACTS / "figure_tau_truncation_effect.svg",
        trunc_groups,
        "Tau Truncation Effect",
        ["recall", "invalid_norm"],
    )

    summary = [
        "# Paper Artifacts",
        "",
        "Generated files:",
        "",
        "- `table_static_summary.csv` / `.md`",
        "- `table_geometry_correlations.csv` / `.md`",
        "- `table_tau_truncation.csv` / `.md`",
        "- `figure_static_math_vs_rank_utilization.svg`",
        "- `figure_rank_vs_static_math.svg`",
        "- `figure_rank_vs_planbench.svg`",
        "- `figure_tau_collapse_basin.svg`",
        "- `figure_tau_truncation_effect.svg`",
    ]
    (ARTIFACTS / "README.md").write_text("\n".join(summary) + "\n")
    print(ARTIFACTS)


if __name__ == "__main__":
    main()
