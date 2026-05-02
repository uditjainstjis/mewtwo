"""
ICLR Adaptive Clamp Full Benchmark
====================================
Runs ALL 4 methods we are comparing across 100 hard domain questions:
  1. Baseline        (no adapters, clamp=0.001)
  2. Single Adapter  (top-1 routed adapter, clamp=0.5)
  3. Unclamped Mix   (top-2 adapters, clamp=999 i.e. unclamped)
  4. Adaptive Clamp  (top-2 adapters, clamp=0.5)

Metrics per question:
  - Semantic Similarity (cosine via sentence-transformers)
  - Perplexity of ground truth under each config
  - Latency (wall clock seconds)

Outputs:
  - results/real_benchmark_results.json  (raw per-question data)
  - results/real_benchmark_table.md      (paper-ready markdown table)
  - results_db.jsonl                     (appended, real_mode=true)
"""

import json
import time
import os
import sys
import numpy as np
from datetime import datetime, timezone

# We run from backend/ since adapter paths are relative to it
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from dynamic_mlx_inference import DynamicEngine, set_global_clamp
from orchestrator import Orchestrator

# Import the 100 hard questions
from ablation_benchmark import HARD_QUESTIONS

# ── CONFIG ──
CONFIGS = [
    # (name, k, clamp, use_adapters)
    ("Baseline",       1, 0.001,  False),
    ("SingleAdapter",  1, 0.5,    True),
    ("UnclampedMix",   2, 999.0,  True),
    ("AdaptiveClamp",  2, 0.5,    True),
]

def load_semantic_model():
    from sentence_transformers import SentenceTransformer
    print("Loading semantic model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_similarity(model, a, b):
    embs = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))

def run_full_benchmark():
    # Load engine
    registry = json.load(open("expert_registry.json"))
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)
    sem_model = load_semantic_model()

    domains = list(HARD_QUESTIONS.keys())
    total_q = sum(len(v) for v in HARD_QUESTIONS.values())

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE CLAMP FULL BENCHMARK")
    print(f"  {total_q} questions × {len(CONFIGS)} configs = {total_q * len(CONFIGS)} runs")
    print(f"{'='*70}\n")

    all_results = {}  # domain -> config_name -> list of dicts
    jsonl_path = os.path.join(PROJECT_ROOT, "results_db.jsonl")

    q_idx = 0
    for domain in domains:
        all_results[domain] = {c[0]: [] for c in CONFIGS}
        questions = HARD_QUESTIONS[domain]

        for qi, qa in enumerate(questions):
            q_idx += 1
            question = qa["q"]
            ground_truth = qa["a"]
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

            print(f"\n[{q_idx}/{total_q}] {domain} Q{qi+1}: {question[:70]}...")

            # Get routing weights once
            routing_raw, cot_text = orchestrator.route(question, top_k=1)
            sorted_domains = sorted(routing_raw.items(), key=lambda x: x[1], reverse=True)

            for config_name, k, clamp_val, use_adapters in CONFIGS:
                set_global_clamp(clamp_val)

                if not use_adapters:
                    rw = None
                elif k == 1:
                    # Single adapter: only top-1
                    rw = {d: 0.0 for d in routing_raw}
                    rw[sorted_domains[0][0]] = 1.0
                elif k == 2:
                    # Top-2 adapters
                    rw = {d: 0.0 for d in routing_raw}
                    for dd, _ in sorted_domains[:2]:
                        rw[dd] = 1.0
                else:
                    rw = None

                # Generate
                start_t = time.time()
                generated, gen_dur = engine.generate(prompt, routing_weights=rw, max_tokens=100)
                latency = time.time() - start_t

                # Semantic similarity
                sim = semantic_similarity(sem_model, generated, ground_truth)

                # Perplexity
                ppl = engine.compute_perplexity(prompt, ground_truth, routing_weights=rw)

                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated": generated[:300],
                    "semantic_sim": round(sim, 4),
                    "perplexity": round(min(ppl, 99999.0), 2),
                    "latency_s": round(latency, 3),
                }
                all_results[domain][config_name].append(result)

                # Also append to results_db.jsonl (real mode)
                log_line = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "exp_id": f"real_{config_name}_{domain}_Q{qi+1}",
                    "method": config_name,
                    "k": k,
                    "c": clamp_val,
                    "dataset": "hard_questions_100",
                    "prompt_id": f"{domain}_Q{qi+1}",
                    "metric_name": "semantic_similarity",
                    "metric_value": round(sim, 4),
                    "perplexity": round(min(ppl, 99999.0), 2),
                    "latency_ms": round(latency * 1000, 2),
                    "real_mode": True,
                    "prediction_preview": generated[:200],
                }
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(log_line) + "\n")

                print(f"    {config_name:20s} | sim={sim:.3f} | ppl={min(ppl,9999):.1f} | lat={latency:.2f}s")

    # Reset clamp
    set_global_clamp(0.5)

    # ── Save raw results ──
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {kk: clean(v) for kk, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    with open(os.path.join(results_dir, "real_benchmark_results.json"), "w") as f:
        json.dump(clean(all_results), f, indent=2)

    # ── Generate paper table ──
    generate_paper_table(all_results, domains, results_dir)

    print(f"\n{'='*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*70}")


def generate_paper_table(all_results, domains, results_dir):
    lines = []
    lines.append("# Table 1: Multi-Adapter Composition — Full Benchmark (REAL)")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: Qwen2.5-1.5B-Instruct-4bit | 100 questions × 20 domains*")
    lines.append("")
    lines.append("> All values from REAL model inference. No simulation.")
    lines.append("")

    # Aggregate across all domains
    config_names = [c[0] for c in CONFIGS]
    grand = {c: {"sim": [], "ppl": [], "lat": []} for c in config_names}

    lines.append("## Per-Domain Semantic Similarity (↑ better)")
    lines.append("")
    header = "| Domain | " + " | ".join(config_names) + " |"
    sep = "|--------|" + "|".join(["-------"] * len(config_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for domain in domains:
        row = [domain[:20]]
        for cn in config_names:
            avg = np.mean([r["semantic_sim"] for r in all_results[domain][cn]])
            grand[cn]["sim"].append(avg)
            row.append(f"{avg:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # Grand average row
    lines.append(sep)
    avg_row = ["**AVERAGE**"]
    for cn in config_names:
        avg_row.append(f"**{np.mean(grand[cn]['sim']):.3f}**")
    lines.append("| " + " | ".join(avg_row) + " |")

    # Summary table
    lines.append("")
    lines.append("## Summary: Method Comparison")
    lines.append("")
    lines.append("| Method | K | Clamp | Avg Sim ↑ | Avg PPL ↓ | Avg Latency |")
    lines.append("|--------|---|-------|-----------|-----------|-------------|")

    for cn in config_names:
        k_val = [c[1] for c in CONFIGS if c[0] == cn][0]
        c_val = [c[2] for c in CONFIGS if c[0] == cn][0]
        avg_sim = np.mean(grand[cn]["sim"])

        ppl_all = []
        lat_all = []
        for domain in domains:
            for r in all_results[domain][cn]:
                ppl_all.append(r["perplexity"])
                lat_all.append(r["latency_s"])
        avg_ppl = np.mean(ppl_all)
        avg_lat = np.mean(lat_all)

        lines.append(f"| {cn} | {k_val} | {c_val} | {avg_sim:.3f} | {avg_ppl:.1f} | {avg_lat:.2f}s |")

    # Deltas
    base_sim = np.mean(grand["Baseline"]["sim"])
    ac_sim = np.mean(grand["AdaptiveClamp"]["sim"])
    sa_sim = np.mean(grand["SingleAdapter"]["sim"])

    delta_vs_base = ac_sim - base_sim
    delta_vs_single = ac_sim - sa_sim

    lines.append("")
    lines.append("## Pre-Registered Δ Metrics (REAL)")
    lines.append("")
    lines.append(f"- **Δ_SIM(AdaptiveClamp − Baseline):** {delta_vs_base:+.4f}")
    lines.append(f"- **Δ_SIM(AdaptiveClamp − SingleAdapter):** {delta_vs_single:+.4f}")
    lines.append(f"- **Threshold:** Δ > 0.05 for compositional gain → {'**PASS**' if delta_vs_single > 0.05 else '**FAIL**'}")

    output = "\n".join(lines) + "\n"
    table_path = os.path.join(results_dir, "real_benchmark_table.md")
    with open(table_path, "w") as f:
        f.write(output)
    print(f"\nTable written to {table_path}")
    print(output)


if __name__ == "__main__":
    run_full_benchmark()
