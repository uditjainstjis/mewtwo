import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TAU_JSON = ROOT / "results" / "agentic_eval" / "tau_bench_results.json"
OUT_MD = ROOT / "results" / "agentic_eval" / "spectral_recoverability_report.md"
OUT_JSON = ROOT / "results" / "agentic_eval" / "spectral_recoverability_report.json"
OUT_SVG = ROOT / "results" / "paper_artifacts" / "figure_spectral_recoverability.svg"

BASIN_TARGETS = [
    (128, "dpo"),
    (128, "merged_dare"),
    (1024, "math_sft"),
    (1024, "merged_dare"),
    (1024, "dpo"),
    (3072, "math_sft"),
    (3072, "merged_dare"),
    (3072, "dpo"),
]

INTERVENTIONS = [
    (128, "math_sft", ["trunc8", "trunc2"]),
    (1024, "merged_dare", ["trunc8", "trunc2"]),
    (3072, "math_sft", ["trunc8", "trunc2"]),
]


def tau_key(rank: int, stage: str) -> str:
    return (
        f"model=qwen_0.8b|rank={rank}|stage={stage}|env=retail|split=test|"
        f"trials=1|agent=local_hf|user=local_hf:qwen_0.8b:base:0"
    )


def load_tau():
    return json.loads(TAU_JSON.read_text())["benchmarks"]


def adapter_dir(rank: int, stage: str) -> Path:
    if stage == "math_sft":
        return OUTPUTS / f"qwen_0.8b_math_SFT_rank{rank}"
    if stage == "merged_dare":
        base = OUTPUTS / f"qwen_0.8b_merged_DARE_rank{rank}"
        if (base / "adapter_model.safetensors").exists():
            return base
        merged = base / "merged_sft"
        return merged
    if stage == "dpo":
        return OUTPUTS / f"qwen_0.8b_math_DPO_rank{rank}"
    if stage.startswith("math_sft_trunc"):
        suffix = stage.split("_")[-1]
        return OUTPUTS / f"qwen_0.8b_math_SFT_rank{rank}_{suffix}"
    if stage.startswith("merged_dare_trunc"):
        suffix = stage.split("_")[-1]
        return OUTPUTS / f"qwen_0.8b_merged_DARE_rank{rank}_{suffix}"
    if stage.startswith("dpo_trunc"):
        suffix = stage.split("_")[-1]
        return OUTPUTS / f"qwen_0.8b_math_DPO_rank{rank}_{suffix}"
    raise ValueError(stage)


def load_modules(path: Path):
    tensor_map = load_file(str(path / "adapter_model.safetensors"))
    grouped = {}
    for key, tensor in tensor_map.items():
        if key.endswith(".lora_A.weight"):
            stem = key[: -len(".lora_A.weight")]
            grouped.setdefault(stem, {})["A"] = tensor.float().cpu()
        elif key.endswith(".lora_B.weight"):
            stem = key[: -len(".lora_B.weight")]
            grouped.setdefault(stem, {})["B"] = tensor.float().cpu()
    return {name: vals for name, vals in grouped.items() if "A" in vals and "B" in vals}


def reduced_singular_values(a: torch.Tensor, b: torch.Tensor):
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    qb, rb = torch.linalg.qr(b, mode="reduced")
    small = rb @ ra.T
    return torch.linalg.svdvals(small)


def aggregate_geometry(path: Path):
    modules = load_modules(path)
    total_fro_sq = 0.0
    weighted_effective_rank = 0.0
    weighted_stable_rank = 0.0
    weighted_energy_rank90 = 0.0
    weighted_nominal_rank = 0.0
    for vals in modules.values():
        s = reduced_singular_values(vals["A"], vals["B"])
        if s.numel() == 0:
            continue
        energy = s.pow(2)
        fro_sq = float(energy.sum().item())
        total_fro_sq += fro_sq
        spec = float(s.max().item())
        stable_rank = fro_sq / (spec * spec) if spec else 0.0
        p = energy / energy.sum()
        entropy = float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())
        effective_rank = math.exp(entropy)
        cum = torch.cumsum(energy, dim=0)
        threshold = 0.9 * fro_sq
        energy_rank90 = float(int(torch.searchsorted(cum, torch.tensor(threshold)).item()) + 1)
        nominal_rank = int(vals["A"].shape[0])
        weighted_effective_rank += fro_sq * effective_rank
        weighted_stable_rank += fro_sq * stable_rank
        weighted_energy_rank90 += fro_sq * energy_rank90
        weighted_nominal_rank += fro_sq * nominal_rank
    if total_fro_sq == 0:
        return {}
    nominal = weighted_nominal_rank / total_fro_sq
    eff = weighted_effective_rank / total_fro_sq
    return {
        "global_fro_norm": math.sqrt(total_fro_sq),
        "weighted_effective_rank": eff,
        "weighted_stable_rank": weighted_stable_rank / total_fro_sq,
        "weighted_energy_rank90": weighted_energy_rank90 / total_fro_sq,
        "rank_utilization": eff / nominal if nominal else 0.0,
    }


def row_metrics(row: dict):
    prefix = row.get("average_gold_tool_name_prefix_frac")
    recall = row.get("average_gold_tool_name_recall_frac")
    terminal = row.get("average_gold_terminal_tool_match") or 0.0
    invalid = row.get("average_invalid_actions")
    invalid_score = None if invalid is None else max(0.0, 1.0 - invalid / 12.0)
    return {
        "prefix": prefix,
        "recall": recall,
        "terminal": terminal,
        "invalid": invalid,
        "invalid_score": invalid_score,
    }


def reliability_score(metrics: dict):
    vals = [metrics["prefix"], metrics["recall"], metrics["terminal"], metrics["invalid_score"]]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def basin_centroid(tau: dict):
    pts = []
    for rank, stage in BASIN_TARGETS:
        row = tau[tau_key(rank, stage)]
        m = row_metrics(row)
        pts.append(m)
    centroid = {}
    for key in ["prefix", "recall", "terminal", "invalid_score"]:
        vals = [p[key] for p in pts if p[key] is not None]
        centroid[key] = sum(vals) / len(vals)
    return centroid


def basin_distance(metrics: dict, centroid: dict):
    keys = ["prefix", "recall", "terminal", "invalid_score"]
    diffs = []
    for key in keys:
        if metrics[key] is None or centroid[key] is None:
            continue
        diffs.append((metrics[key] - centroid[key]) ** 2)
    return math.sqrt(sum(diffs) / len(diffs)) if diffs else None


def intervention_rows(tau: dict, centroid: dict):
    rows = []
    for rank, base_stage, variants in INTERVENTIONS:
        native_stage = base_stage
        native_row = tau[tau_key(rank, native_stage)]
        native_metrics = row_metrics(native_row)
        native_geom = aggregate_geometry(adapter_dir(rank, native_stage))
        native_score = reliability_score(native_metrics)
        native_dist = basin_distance(native_metrics, centroid)
        for variant in [None] + variants:
            stage = native_stage if variant is None else f"{base_stage}_{variant}"
            row = tau.get(tau_key(rank, stage), {})
            metrics = row_metrics(row)
            geom = aggregate_geometry(adapter_dir(rank, stage))
            score = reliability_score(metrics)
            dist = basin_distance(metrics, centroid)
            rows.append({
                "rank": rank,
                "stage": stage,
                "family": f"{rank} {base_stage}",
                "variant": "native" if variant is None else variant,
                "prefix": metrics["prefix"],
                "recall": metrics["recall"],
                "terminal": metrics["terminal"],
                "invalid": metrics["invalid"],
                "reliability_score": score,
                "basin_distance": dist,
                "rank_utilization": geom.get("rank_utilization"),
                "weighted_effective_rank": geom.get("weighted_effective_rank"),
                "global_fro_norm": geom.get("global_fro_norm"),
                "delta_score_vs_native": None if score is None or native_score is None else score - native_score,
                "delta_basin_distance_vs_native": None if dist is None or native_dist is None else dist - native_dist,
                "delta_effective_rank_vs_native": None if not geom or not native_geom else geom["weighted_effective_rank"] - native_geom["weighted_effective_rank"],
                "delta_rank_utilization_vs_native": None if not geom or not native_geom else geom["rank_utilization"] - native_geom["rank_utilization"],
            })
    return rows


def build_report(rows: list[dict], centroid: dict):
    lines = [
        "# Spectral Recoverability Report",
        "",
        "## Thesis",
        "",
        "- This report focuses on the least-crowded angle in the current project: **limited spectral recoverability**.",
        "- The question is not only whether high-rank rows collapse, but whether simple post-hoc spectral truncation can pull them back out of the weak Tau basin.",
        "",
        "## Weak Basin Definition",
        "",
        "- Weak basin centroid is computed from the shared high-rank native Qwen rows:",
        "  - `128 dpo`",
        "  - `128 merged_dare`",
        "  - `1024 math_sft`",
        "  - `1024 merged_dare`",
        "  - `1024 dpo`",
        "  - `3072 math_sft`",
        "  - `3072 merged_dare`",
        "  - `3072 dpo`",
        f"- Basin centroid:",
        f"  - prefix `{centroid['prefix']:.3f}`",
        f"  - recall `{centroid['recall']:.3f}`",
        f"  - terminal `{centroid['terminal']:.3f}`",
        f"  - invalid-score `{centroid['invalid_score']:.3f}`",
        "",
        "## Intervention Table",
        "",
        "| Family | Variant | Prefix | Recall | Invalid | Reliability | Basin Distance | Δ Reliability | Δ Basin Dist | Δ Eff Rank | Δ Rank Util |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_score = "" if row["delta_score_vs_native"] is None else f"{row['delta_score_vs_native']:+.3f}"
        delta_dist = "" if row["delta_basin_distance_vs_native"] is None else f"{row['delta_basin_distance_vs_native']:+.3f}"
        delta_eff = "" if row["delta_effective_rank_vs_native"] is None else f"{row['delta_effective_rank_vs_native']:+.3f}"
        delta_util = "" if row["delta_rank_utilization_vs_native"] is None else f"{row['delta_rank_utilization_vs_native']:+.3f}"
        lines.append(
            "| "
            f"{row['family']} | {row['variant']} | "
            f"{(100*row['prefix']):.1f}% | {(100*row['recall']):.1f}% | {row['invalid']:.1f} | "
            f"{row['reliability_score']:.3f} | {row['basin_distance']:.3f} | "
            f"{delta_score} | {delta_dist} | {delta_eff} | {delta_util} |"
        )
    lines.extend(
        [
            "",
            "## Main Read",
            "",
            "- `128 math_sft_trunc8` is the only row that moves meaningfully away from the weak basin while also improving reliability.",
            "- `3072 math_sft` stays trapped in the basin under both `trunc8` and `trunc2`.",
            "- `1024 merged_dare` also stays trapped, showing that some high-rank rows are not simply overcomplete but behaviorally entrenched in a weak regime.",
            "- This makes the strongest causal claim sharper:",
            "  - spectral truncation can help a partially degraded row",
            "  - but it does not generally recover deeply collapsed agentic behavior",
            "",
            "## Best Novel Claim",
            "",
            "- **High-rank Tau collapse is not only observable; it is only partially spectrally recoverable.**",
            "- That is a stronger and less crowded statement than either `high rank is bad` or `SVD helps`.",
        ]
    )
    return "\n".join(lines) + "\n"


def svg(rows: list[dict]):
    width, height = 980, 420
    margin = {"l": 90, "r": 30, "t": 60, "b": 120}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    groups = [r for r in rows if r["variant"] != "native"]
    max_y = max(abs(r["delta_score_vs_native"] or 0.0) for r in groups) or 0.05
    max_y = max(max_y, 0.05)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "DejaVu Sans", sans-serif; fill: #111; } .small{font-size:11px;} .title{font-size:18px;font-weight:700;} .axis{stroke:#222;stroke-width:1.2;} .grid{stroke:#ddd;stroke-width:1;} </style>',
        '<text class="title" x="24" y="28">Spectral Recoverability Delta</text>',
    ]
    zero_y = margin["t"] + plot_h / 2
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin["t"] + plot_h * frac
        lines.append(f'<line class="grid" x1="{margin["l"]}" y1="{y}" x2="{margin["l"]+plot_w}" y2="{y}" />')
    lines.append(f'<line class="axis" x1="{margin["l"]}" y1="{zero_y}" x2="{margin["l"]+plot_w}" y2="{zero_y}" />')
    group_w = plot_w / max(len(groups), 1)
    for i, row in enumerate(groups):
        x = margin["l"] + i * group_w + group_w * 0.2
        bar_w = group_w * 0.6
        delta = row["delta_score_vs_native"] or 0.0
        bar_h = (abs(delta) / max_y) * (plot_h / 2)
        y = zero_y - bar_h if delta >= 0 else zero_y
        color = "#0b6e4f" if delta > 0 else "#8f1d21"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" />')
        lines.append(f'<text class="small" transform="translate({x+bar_w/2:.1f} {height-48}) rotate(-35)" text-anchor="end">{escape(row["family"] + " " + row["variant"])}</text>')
        lines.append(f'<text class="small" x="{x+bar_w/2:.1f}" y="{y-5 if delta>=0 else y+bar_h+14:.1f}" text-anchor="middle">{delta:+.3f}</text>')
    lines.append("</svg>")
    OUT_SVG.write_text("\n".join(lines) + "\n")


def main():
    tau = load_tau()
    centroid = basin_centroid(tau)
    rows = intervention_rows(tau, centroid)
    OUT_MD.write_text(build_report(rows, centroid))
    OUT_JSON.write_text(json.dumps({"centroid": centroid, "rows": rows}, indent=2))
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    svg(rows)
    print(OUT_MD)


if __name__ == "__main__":
    main()
