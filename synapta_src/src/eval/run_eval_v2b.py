"""
v2b/v2c Final Experiments — Clamp Ablation & Routing Gap
========================================================
Phase 2 (clamp): SingleAdapter vs AC-v2-WeightCap vs AC-v2-NormRatio on MD
Phase 3 (routing): SingleAdapter vs AC-v2-Norm-Oracle vs AC-v2-Norm-RealRouter on MD

Usage:
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_v2b.py --phase clamp --real
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_v2b.py --phase routing --real
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_v2b.py --phase sanity --real
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# ──────────────────────────────────────────────
# Engine + models (reuse from run_eval_v2)
# ──────────────────────────────────────────────

_engine = None
_orchestrator = None
_sem_model = None


def get_engine():
    global _engine, _orchestrator
    if _engine is None:
        backend_dir = str(PROJECT_ROOT / "backend")
        original_cwd = os.getcwd()
        os.chdir(backend_dir)
        from dynamic_mlx_inference import DynamicEngine
        from orchestrator import Orchestrator
        registry = json.load(open("expert_registry.json"))
        _engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
        _orchestrator = Orchestrator("expert_registry.json", base_engine=_engine)
        os.chdir(original_cwd)
        print("✅ Real DynamicEngine + Orchestrator loaded.")
    return _engine, _orchestrator


def get_sem_model():
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Semantic similarity model loaded.")
    return _sem_model


def semantic_similarity(a: str, b: str) -> float:
    model = get_sem_model()
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p = _normalize_text(pred).split()
    r = _normalize_text(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    p_counts = {}
    r_counts = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1
    overlap = sum(min(p_counts[t], r_counts.get(t, 0)) for t in p_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def adapter_recall_at_k(routing_weights: Dict[str, float], required: List[str], k: int) -> float:
    required = [x for x in (required or []) if x]
    if not required:
        return 1.0
    ranked = sorted(routing_weights.items(), key=lambda kv: kv[1], reverse=True)
    predicted = [name for name, w in ranked[:k] if w > 0]
    hit = len(set(predicted) & set(required))
    return hit / len(set(required))


def load_md_dataset() -> List[dict]:
    path = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    with open(path) as f:
        items = json.load(f)
    for item in items:
        item["split"] = "MD"
        if "required_adapters" not in item and "domains" in item:
            item["required_adapters"] = item["domains"]
        if "domains" not in item and "required_adapters" in item:
            item["domains"] = item["required_adapters"]
    return items


# ──────────────────────────────────────────────
# Routing helpers
# ──────────────────────────────────────────────

def build_oracle_weights(item, orchestrator, k=2):
    adapters = item.get("required_adapters", item.get("domains", []))
    k = min(k, len(adapters))
    all_domains = list(orchestrator.registry.keys())
    weights = {d: 0.0 for d in all_domains}
    for a in adapters[:k]:
        if a in weights:
            weights[a] = 1.0 / k
    return weights, k


def build_cot_weights(item, orchestrator):
    raw_weights, cot_text = orchestrator.route(item["question"], top_k=1)
    return raw_weights, 1


def build_real_top2_weights(item, orchestrator):
    weights, domains_found, cot_text = orchestrator.route_top2(item["question"])
    k = 2 if domains_found[1] is not None else 1
    return weights, k


# ──────────────────────────────────────────────
# Phase definitions
# ──────────────────────────────────────────────

PHASE_CLAMP_METHODS = [
    {"name": "SingleAdapter",      "clamp_mode": "weight_cap", "clamp_c": 0.5, "routing_fn": "cot",    "k": 1},
    {"name": "AC-v2-WeightCap",    "clamp_mode": "weight_cap", "clamp_c": 0.5, "routing_fn": "oracle", "k": 2},
    {"name": "AC-v2-NormRatio",    "clamp_mode": "norm_ratio", "clamp_c": 0.5, "routing_fn": "oracle", "k": 2},
]

PHASE_ROUTING_METHODS = [
    {"name": "SingleAdapter",           "clamp_mode": "norm_ratio", "clamp_c": 0.5, "routing_fn": "cot",        "k": 1},
    {"name": "AC-v2-Norm-Oracle",       "clamp_mode": "norm_ratio", "clamp_c": 0.5, "routing_fn": "oracle",     "k": 2},
    {"name": "AC-v2-Norm-RealRouter",   "clamp_mode": "norm_ratio", "clamp_c": 0.5, "routing_fn": "real_top2",  "k": 2},
]


def run_phase(phase_name, methods, items, output_path):
    engine, orchestrator = get_engine()
    from dynamic_mlx_inference import set_global_clamp, set_clamp_mode

    total = len(items) * len(methods)
    print(f"\n{'='*70}")
    print(f"  v2b: {phase_name}")
    print(f"  Questions: {len(items)} | Methods: {len(methods)} | Total inferences: {total}")
    print(f"  Methods: {[m['name'] for m in methods]}")
    print(f"{'='*70}\n")

    all_results = []
    count = 0

    for item in items:
        for method in methods:
            count += 1

            # Configure clamp
            set_clamp_mode(method["clamp_mode"])
            set_global_clamp(method["clamp_c"])

            # Route
            if method["routing_fn"] == "cot":
                routing_weights, K_used = build_cot_weights(item, orchestrator)
            elif method["routing_fn"] == "oracle":
                routing_weights, K_used = build_oracle_weights(item, orchestrator, method["k"])
            elif method["routing_fn"] == "real_top2":
                routing_weights, K_used = build_real_top2_weights(item, orchestrator)
            else:
                raise ValueError(f"Unknown routing_fn: {method['routing_fn']}")

            # Generate
            prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            start = time.time()
            text, gen_dur = engine.generate(prompt, routing_weights=routing_weights, max_tokens=150)
            latency = time.time() - start

            # Metrics
            sim = semantic_similarity(text, item["reference_answer"])
            em = exact_match(text, item["reference_answer"])
            f1 = token_f1(text, item["reference_answer"])
            ar = adapter_recall_at_k(
                routing_weights,
                item.get("required_adapters", item.get("domains", [])),
                max(1, K_used),
            )

            set_clamp_mode(method["clamp_mode"])
            set_global_clamp(method["clamp_c"])
            ppl = engine.compute_perplexity(prompt, item["reference_answer"], routing_weights=routing_weights)

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": phase_name,
                "split": "MD",
                "item_id": item["id"],
                "domains": item.get("domains", []),
                "method": method["name"],
                "clamp_mode": method["clamp_mode"],
                "clamp_c": method["clamp_c"],
                "routing_fn": method["routing_fn"],
                "K_used": K_used,
                "generated_text_preview": (text or "")[:200],
                "semantic_sim": round(sim, 4),
                "exact_match": round(em, 4),
                "token_f1": round(f1, 4),
                "adapter_recall_k": round(ar, 4),
                "perplexity": round(ppl, 2),
                "latency_s": round(latency, 3),
                "real_mode": True,
            }
            all_results.append(result)

            print(
                f"  [{count}/{total}] {item['id'][:12]:12s} | "
                f"{method['name']:25s} | clamp={method['clamp_mode']:10s} | "
                f"EM={em:.2f} | F1={f1:.2f} | Sim={sim:.3f} | PPL={ppl:.1f} | Lat={latency:.2f}s | K={K_used}"
            )

    # Reset to defaults
    set_clamp_mode("weight_cap")
    set_global_clamp(0.5)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n✅ Results saved to {output_path} ({len(all_results)} entries)")

    # Aggregates
    print_aggregates(all_results, methods)
    return all_results


def print_aggregates(results, methods):
    print(f"\n{'='*70}")
    print("  AGGREGATE RESULTS (MD)")
    print(f"{'='*70}")
    print(f"  {'Method':25s} | {'ClampMode':10s} | {'EM%':>7s} | {'F1':>6s} | {'Avg Sim':>8s} | {'Avg PPL':>8s} | {'Avg Lat':>8s} | {'Avg K':>5s} | {'AdpRec':>7s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}")

    method_sims = {}
    for method in methods:
        m_res = [r for r in results if r["method"] == method["name"]]
        if not m_res:
            continue
        avg_sim = np.mean([r["semantic_sim"] for r in m_res])
        avg_em = np.mean([r.get("exact_match", 0.0) for r in m_res])
        avg_f1 = np.mean([r.get("token_f1", 0.0) for r in m_res])
        avg_ppl = np.mean([r["perplexity"] for r in m_res])
        avg_lat = np.mean([r["latency_s"] for r in m_res])
        avg_k = np.mean([r["K_used"] for r in m_res])
        avg_ar = np.mean([r.get("adapter_recall_k", 0.0) for r in m_res])
        method_sims[method["name"]] = avg_sim
        print(
            f"  {method['name']:25s} | {method['clamp_mode']:10s} | {100.0*avg_em:6.1f}% | {avg_f1:6.3f} | {avg_sim:8.4f} | "
            f"{avg_ppl:8.1f} | {avg_lat:8.3f} | {avg_k:5.2f} | {avg_ar:7.3f}"
        )

    # Deltas
    print(f"\n  DELTAS:")
    sa_sim = method_sims.get("SingleAdapter")
    for name, sim in method_sims.items():
        if name != "SingleAdapter" and sa_sim is not None:
            print(f"    Δ_SIM({name} − SA) = {sim - sa_sim:+.4f}")

    names = list(method_sims.keys())
    if len(names) >= 3:
        print(f"    Δ_SIM({names[2]} − {names[1]}) = {method_sims[names[2]] - method_sims[names[1]]:+.4f}")


# ──────────────────────────────────────────────
# Sanity check: 1 MD question, 3 methods
# ──────────────────────────────────────────────

def run_sanity():
    items = load_md_dataset()[:1]
    methods = PHASE_CLAMP_METHODS
    print("\n🔬 SANITY CHECK: 1 question × 3 methods")
    run_phase("sanity", methods, items, os.path.join("results", "v2b_sanity.jsonl"))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="v2b/v2c Final Experiments")
    parser.add_argument("--phase", choices=["sanity", "clamp", "routing"], required=True)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()

    if not args.real:
        print("⚠️  Use --real for actual experiments.")
        return

    if args.phase == "sanity":
        run_sanity()
    elif args.phase == "clamp":
        items = load_md_dataset()
        run_phase("clamp_ablation", PHASE_CLAMP_METHODS, items,
                  os.path.join("results", "v2_md_clamp_ablation.jsonl"))
    elif args.phase == "routing":
        items = load_md_dataset()
        run_phase("routing_gap", PHASE_ROUTING_METHODS, items,
                  os.path.join("results", "v2_md_routing_ablation.jsonl"))


if __name__ == "__main__":
    main()
