import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
STATIC_RESULTS_JSON = ROOT / "results" / "post_dpo_benchmarks.json"
AGENTIC_RESULTS_JSON = ROOT / "results" / "agentic_eval" / "planbench_results.json"
TAU_RESULTS_JSON = ROOT / "results" / "agentic_eval" / "tau_bench_results.json"
OUT_JSON = ROOT / "results" / "qwen_geometry_behavior.json"
OUT_MD = ROOT / "results" / "qwen_geometry_behavior_report.md"

MODEL = "qwen_0.8b"
RANKS = [1, 2, 8, 128, 1024, 3072]
STAGES = ["math_sft", "science_sft", "code_sft", "merged_dare", "dpo"]
STATIC_BENCHMARKS = ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval", "truthfulqa_mc", "mmlu_pro", "gpqa"]
MATH_BENCHMARKS = ["gsm8k", "math500"]
AGENTIC_TASK = ("blocksworld_3", "t1")


def stage_path(rank: int, stage: str) -> Path | None:
    if stage == "merged_dare":
        base = OUTPUTS_DIR / f"{MODEL}_merged_DARE_rank{rank}"
        if (base / "adapter_model.safetensors").exists():
            return base
        merged_sft = base / "merged_sft"
        if (merged_sft / "adapter_model.safetensors").exists():
            return merged_sft
        return None
    if stage == "dpo":
        base = OUTPUTS_DIR / f"{MODEL}_math_DPO_rank{rank}"
        return base if (base / "adapter_model.safetensors").exists() else None
    domain = stage.replace("_sft", "")
    base = OUTPUTS_DIR / f"{MODEL}_{domain}_SFT_rank{rank}"
    return base if (base / "adapter_model.safetensors").exists() else None


def load_modules(adapter_path: Path):
    tensor_map = load_file(str(adapter_path / "adapter_model.safetensors"))
    grouped = {}
    for key, tensor in tensor_map.items():
        if key.endswith(".lora_A.weight"):
            stem = key[: -len(".lora_A.weight")]
            grouped.setdefault(stem, {})["A"] = tensor.float().cpu()
        elif key.endswith(".lora_B.weight"):
            stem = key[: -len(".lora_B.weight")]
            grouped.setdefault(stem, {})["B"] = tensor.float().cpu()
    return {name: vals for name, vals in grouped.items() if "A" in vals and "B" in vals}


def reduced_singular_values(A: torch.Tensor, B: torch.Tensor):
    qa, ra = torch.linalg.qr(A.T, mode="reduced")
    qb, rb = torch.linalg.qr(B, mode="reduced")
    small = rb @ ra.T
    s = torch.linalg.svdvals(small)
    return s


def module_metrics(A: torch.Tensor, B: torch.Tensor):
    s = reduced_singular_values(A, B)
    if s.numel() == 0:
        return {
            "rank": int(A.shape[0]),
            "fro_sq": 0.0,
            "fro": 0.0,
            "spec": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "energy_rank90": 0.0,
            "nuclear": 0.0,
        }
    energy = s.pow(2)
    fro_sq = float(energy.sum().item())
    fro = math.sqrt(fro_sq)
    spec = float(s.max().item())
    stable_rank = 0.0 if spec == 0.0 else fro_sq / (spec * spec)
    p = energy / energy.sum() if float(energy.sum().item()) > 0 else energy
    entropy = float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item()) if p.numel() else 0.0
    effective_rank = math.exp(entropy) if fro_sq > 0 else 0.0
    cum = torch.cumsum(energy, dim=0)
    energy_rank90 = 0.0
    if fro_sq > 0:
        threshold = 0.9 * fro_sq
        energy_rank90 = float(int(torch.searchsorted(cum, torch.tensor(threshold)).item()) + 1)
    return {
        "rank": int(A.shape[0]),
        "fro_sq": fro_sq,
        "fro": fro,
        "spec": spec,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "energy_rank90": energy_rank90,
        "nuclear": float(s.sum().item()),
    }


def aggregate_adapter(modules: dict):
    per_module = {}
    total_fro_sq = 0.0
    weighted = {
        "stable_rank": 0.0,
        "effective_rank": 0.0,
        "energy_rank90": 0.0,
        "nominal_rank": 0.0,
    }
    q_fro_sq = 0.0
    v_fro_sq = 0.0
    for name, vals in modules.items():
        m = module_metrics(vals["A"], vals["B"])
        per_module[name] = m
        total_fro_sq += m["fro_sq"]
        weight = m["fro_sq"]
        weighted["stable_rank"] += weight * m["stable_rank"]
        weighted["effective_rank"] += weight * m["effective_rank"]
        weighted["energy_rank90"] += weight * m["energy_rank90"]
        weighted["nominal_rank"] += weight * m["rank"]
        if ".q_proj" in name:
            q_fro_sq += m["fro_sq"]
        if ".v_proj" in name:
            v_fro_sq += m["fro_sq"]
    denom = total_fro_sq if total_fro_sq > 0 else 1.0
    nominal_rank = weighted["nominal_rank"] / denom
    effective_rank = weighted["effective_rank"] / denom
    return {
        "module_count": len(per_module),
        "global_fro_norm": math.sqrt(total_fro_sq),
        "global_fro_sq": total_fro_sq,
        "weighted_stable_rank": weighted["stable_rank"] / denom,
        "weighted_effective_rank": effective_rank,
        "weighted_energy_rank90": weighted["energy_rank90"] / denom,
        "weighted_nominal_rank": nominal_rank,
        "rank_utilization": (effective_rank / nominal_rank) if nominal_rank else 0.0,
        "q_proj_energy_share": q_fro_sq / denom,
        "v_proj_energy_share": v_fro_sq / denom,
        "modules": per_module,
    }


def module_dot(mod1: dict, mod2: dict):
    A1, B1 = mod1["A"], mod1["B"]
    A2, B2 = mod2["A"], mod2["B"]
    bt = B1.T @ B2
    aa = A2 @ A1.T
    return float(torch.trace(bt @ aa).item())


def adapter_similarity(modules1: dict, modules2: dict):
    shared = sorted(set(modules1) & set(modules2))
    if not shared:
        return None
    dot = 0.0
    fro1_sq = 0.0
    fro2_sq = 0.0
    for name in shared:
        m1 = module_metrics(modules1[name]["A"], modules1[name]["B"])
        m2 = module_metrics(modules2[name]["A"], modules2[name]["B"])
        dot += module_dot(modules1[name], modules2[name])
        fro1_sq += m1["fro_sq"]
        fro2_sq += m2["fro_sq"]
    denom = math.sqrt(fro1_sq * fro2_sq) if fro1_sq > 0 and fro2_sq > 0 else 0.0
    cosine = dot / denom if denom else 0.0
    return {
        "shared_modules": len(shared),
        "cosine": cosine,
        "rotation_proxy": 1.0 - cosine,
        "fro1": math.sqrt(fro1_sq),
        "fro2": math.sqrt(fro2_sq),
    }


def load_static_results():
    payload = json.loads(STATIC_RESULTS_JSON.read_text())
    return payload.get("benchmarks", {})


def load_agentic_results():
    if not AGENTIC_RESULTS_JSON.exists():
        return {}
    payload = json.loads(AGENTIC_RESULTS_JSON.read_text())
    return payload.get("benchmarks", {})


def load_tau_results():
    if not TAU_RESULTS_JSON.exists():
        return {}
    payload = json.loads(TAU_RESULTS_JSON.read_text())
    return payload.get("benchmarks", {})


def static_scores_for(rank: int, stage: str, static_results: dict):
    scores = {}
    for bench in STATIC_BENCHMARKS:
        key = f"{MODEL}|rank={rank}|stage={stage}|benchmark={bench}"
        if key in static_results:
            scores[bench] = static_results[key].get("score")
    values = [v for v in scores.values() if v is not None]
    math_values = [scores.get(b) for b in MATH_BENCHMARKS if scores.get(b) is not None]
    scores["overall"] = sum(values) / len(values) if values else None
    scores["math_mean"] = sum(math_values) / len(math_values) if math_values else None
    return scores


def agentic_score_for(rank: int, stage: str, agentic_results: dict):
    domain, task = AGENTIC_TASK
    key = f"model={MODEL}|rank={rank}|stage={stage}|domain={domain}|task={task}"
    if key not in agentic_results:
        return None
    return agentic_results[key].get("score")


def tau_row_for(rank: int, stage: str, tau_results: dict):
    prefix = f"model={MODEL}|rank={rank}|stage={stage}|"
    matches = []
    for key, value in tau_results.items():
        if not key.startswith(prefix):
            continue
        if "|env=retail|" not in key:
            continue
        matches.append((key, value))
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            int(item[1].get("total", 0) or 0),
            int(item[1].get("num_trials", 0) or 0),
        ),
        reverse=True,
    )
    return matches[0][1]


def rankdata(values):
    indexed = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for _, idx in indexed[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def build_inventory():
    static_results = load_static_results()
    agentic_results = load_agentic_results()
    tau_results = load_tau_results()
    inventory = {}
    raw_modules = {}
    for rank in RANKS:
        for stage in STAGES:
            path = stage_path(rank, stage)
            if path is None:
                continue
            modules = load_modules(path)
            raw_modules[(rank, stage)] = modules
            geom = aggregate_adapter(modules)
            tau_row = tau_row_for(rank, stage, tau_results)
            inventory[(rank, stage)] = {
                "rank": rank,
                "stage": stage,
                "path": str(path),
                "geometry": geom,
                "static": static_scores_for(rank, stage, static_results),
                "agentic_t1_blocksworld3": agentic_score_for(rank, stage, agentic_results),
                "tau_bench_retail_pass1": None if tau_row is None else tau_row.get("score"),
                "tau_bench_retail_avg_reward": None if tau_row is None else tau_row.get("average_reward"),
                "tau_bench_retail_invalid_actions": (
                    None if tau_row is None else tau_row.get("average_invalid_actions")
                ),
                "tau_bench_retail_tool_name_prefix": (
                    None if tau_row is None else tau_row.get("average_gold_tool_name_prefix_frac")
                ),
                "tau_bench_retail_exact_prefix": (
                    None if tau_row is None else tau_row.get("average_gold_exact_prefix_frac")
                ),
                "tau_bench_retail_tool_name_recall": (
                    None if tau_row is None else tau_row.get("average_gold_tool_name_recall_frac")
                ),
                "tau_bench_retail_terminal_tool_match": (
                    None if tau_row is None else tau_row.get("average_gold_terminal_tool_match")
                ),
            }
    return inventory, raw_modules


def build_transitions(raw_modules: dict):
    transitions = []
    for rank in RANKS:
        if (rank, "merged_dare") in raw_modules and (rank, "dpo") in raw_modules:
            sim = adapter_similarity(raw_modules[(rank, "merged_dare")], raw_modules[(rank, "dpo")])
            if sim:
                transitions.append({"rank": rank, "pair": "merged_dare->dpo", **sim})
        if (rank, "math_sft") in raw_modules and (rank, "merged_dare") in raw_modules:
            sim = adapter_similarity(raw_modules[(rank, "math_sft")], raw_modules[(rank, "merged_dare")])
            if sim:
                transitions.append({"rank": rank, "pair": "math_sft->merged_dare", **sim})
        if (rank, "math_sft") in raw_modules and (rank, "dpo") in raw_modules:
            sim = adapter_similarity(raw_modules[(rank, "math_sft")], raw_modules[(rank, "dpo")])
            if sim:
                transitions.append({"rank": rank, "pair": "math_sft->dpo", **sim})
    return transitions


def correlation_report(inventory: dict):
    rows = list(inventory.values())
    feature_names = [
        "weighted_stable_rank",
        "weighted_effective_rank",
        "weighted_energy_rank90",
        "rank_utilization",
        "global_fro_norm",
        "q_proj_energy_share",
        "v_proj_energy_share",
    ]
    outcomes = {
        "static_overall": lambda row: row["static"].get("overall"),
        "static_math_mean": lambda row: row["static"].get("math_mean"),
        "agentic_t1_blocksworld3": lambda row: row.get("agentic_t1_blocksworld3"),
        "tau_bench_retail_pass1": lambda row: row.get("tau_bench_retail_pass1"),
        "tau_bench_retail_avg_reward": lambda row: row.get("tau_bench_retail_avg_reward"),
        "tau_bench_retail_tool_name_prefix": lambda row: row.get("tau_bench_retail_tool_name_prefix"),
        "tau_bench_retail_exact_prefix": lambda row: row.get("tau_bench_retail_exact_prefix"),
        "tau_bench_retail_tool_name_recall": lambda row: row.get("tau_bench_retail_tool_name_recall"),
        "tau_bench_retail_terminal_tool_match": lambda row: row.get("tau_bench_retail_terminal_tool_match"),
    }
    output = {}
    for outcome_name, outcome_fn in outcomes.items():
        feature_stats = {}
        for feature in feature_names + ["log_rank"]:
            xs = []
            ys = []
            for row in rows:
                y = outcome_fn(row)
                if y is None:
                    continue
                x = math.log2(row["rank"]) if feature == "log_rank" else row["geometry"][feature]
                xs.append(float(x))
                ys.append(float(y))
            feature_stats[feature] = {
                "n": len(xs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        output[outcome_name] = feature_stats
    return output


def fmt_pct(v):
    return "-" if v is None else f"{100.0 * v:.1f}%"


def fmt_num(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def render_report(inventory: dict, transitions: list[dict], correlations: dict):
    best_math = sorted(
        [row for row in inventory.values() if row["static"].get("math_mean") is not None],
        key=lambda row: row["static"]["math_mean"],
        reverse=True,
    )
    best_agentic = sorted(
        [row for row in inventory.values() if row.get("agentic_t1_blocksworld3") is not None],
        key=lambda row: row["agentic_t1_blocksworld3"],
        reverse=True,
    )
    lines = [
        "# Qwen Geometry-Behavior Report",
        "",
        "## Thesis",
        "",
        "- This report bridges finished adapter geometry with both static benchmark behavior and live agentic behavior.",
        "- The goal is to move the paper from `rank tables` toward `geometry predicts behavior`.",
        "",
        "## Geometry Inventory",
        "",
        "| Rank | Stage | Fro Norm | Stable Rank | Effective Rank | Rank Utilization | Q Share | V Share | Static Overall | Static Math | Agentic T1 | Tau Pass@1 | Tau Prefix | Tau Exact Prefix | Tau Recall | Tau Terminal |",
        "|------|-------|----------|-------------|----------------|------------------|---------|---------|----------------|-------------|------------|------------|------------|------------------|------------|--------------|",
    ]
    for row in sorted(inventory.values(), key=lambda r: (r["rank"], STAGES.index(r["stage"]))):
        g = row["geometry"]
        lines.append(
            f"| {row['rank']} | {row['stage']} | {fmt_num(g['global_fro_norm'])} | {fmt_num(g['weighted_stable_rank'])} | "
            f"{fmt_num(g['weighted_effective_rank'])} | {fmt_num(g['rank_utilization'])} | {fmt_num(g['q_proj_energy_share'])} | "
            f"{fmt_num(g['v_proj_energy_share'])} | {fmt_pct(row['static'].get('overall'))} | "
            f"{fmt_pct(row['static'].get('math_mean'))} | {fmt_pct(row.get('agentic_t1_blocksworld3'))} | "
            f"{fmt_pct(row.get('tau_bench_retail_pass1'))} | {fmt_pct(row.get('tau_bench_retail_tool_name_prefix'))} | "
            f"{fmt_pct(row.get('tau_bench_retail_exact_prefix'))} | {fmt_pct(row.get('tau_bench_retail_tool_name_recall'))} | "
            f"{fmt_pct(row.get('tau_bench_retail_terminal_tool_match'))} |"
        )

    lines.extend(
        [
            "",
            "## Stage Rotation Proxies",
            "",
            "| Rank | Pair | Cosine | Rotation Proxy | Shared Modules |",
            "|------|------|--------|----------------|----------------|",
        ]
    )
    for row in transitions:
        lines.append(
            f"| {row['rank']} | {row['pair']} | {fmt_num(row['cosine'])} | {fmt_num(row['rotation_proxy'])} | {row['shared_modules']} |"
        )

    lines.extend(
        [
            "",
            "## Correlation Summary",
            "",
            "| Outcome | Feature | N | Pearson | Spearman |",
            "|---------|---------|---|---------|----------|",
        ]
    )
    for outcome_name, stats in correlations.items():
        ranked = sorted(
            stats.items(),
            key=lambda kv: abs(kv[1]["spearman"]) if kv[1]["spearman"] is not None else -1,
            reverse=True,
        )
        for feature, vals in ranked:
            lines.append(
                f"| {outcome_name} | {feature} | {vals['n']} | {fmt_num(vals['pearson'])} | {fmt_num(vals['spearman'])} |"
            )

    lines.extend(
        [
            "",
            "## Current Read",
            "",
            f"- Best static math rows are led by `{best_math[0]['rank']} {best_math[0]['stage']}` and nearby low-rank variants."
            if best_math
            else "- Static math rows not available.",
            f"- Best live agentic row so far is `{best_agentic[0]['rank']} {best_agentic[0]['stage']}` at {fmt_pct(best_agentic[0].get('agentic_t1_blocksworld3'))}."
            if best_agentic
            else "- Agentic rows are still incomplete; correlations will strengthen as PlanBench fills in.",
            "- The newest offline Tau signal is trajectory shape, not pass@1: terminal reward stays near zero while early tool-use behavior still separates rows.",
            "- The cleanest Tau pattern is a robust high-rank collapse basin for Qwen: many `128/1024/3072` rows converge to the same weak prefix / recall / invalid-action pattern.",
            "- The static-math geometry story is stronger and cleaner than the current Tau correlation story, so the paper should frame agentic behavior as a partial extension rather than a universal monotonic law.",
            "- The strongest causal result is limited: truncation helps `rank128 math_sft` modestly, but it does not rescue deeply collapsed high-rank rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    inventory, raw_modules = build_inventory()
    transitions = build_transitions(raw_modules)
    correlations = correlation_report(inventory)
    serializable = {
        "inventory": {f"rank={rank}|stage={stage}": value for (rank, stage), value in inventory.items()},
        "transitions": transitions,
        "correlations": correlations,
    }
    OUT_JSON.write_text(json.dumps(serializable, indent=2))
    OUT_MD.write_text(render_report(inventory, transitions, correlations))
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
