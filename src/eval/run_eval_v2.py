"""
Multi-Adapter Composition v2 — Evaluation Harness (REAL MODE)
=============================================================
Runs against the actual DynamicEngine + Orchestrator backend.

Architecture Notes (IMPORTANT — read before modifying):
  - The backend uses RoutedLoRALinear which applies a PER-ADAPTER weight cap
    via set_global_clamp() — NOT per-layer norm-ratio clamp.
  - The Orchestrator returns a ONE-HOT domain routing vector (top-1 only).
  - For multi-domain (MD) questions, we use ORACLE routing from the dataset's
    `required_adapters` field to isolate "does K=2 help when both adapters are
    correct?" from router accuracy. This is clearly documented in the paper.

Splits:
  - SD: 100 single-domain templated questions (from ablation_benchmark.py)
  - MD: 40 genuinely multi-domain questions (from multidomain_eval_v2.json)

Methods:
  - Baseline:         No adapters (clamp=0.001)
  - SingleAdapter:    K=1, CoT-routed top-1 domain, clamp=0.5
  - AdaptiveClamp-v2: K=2, oracle-routed required_adapters, clamp=0.5
  - UnclampedMix-v2:  K=2, oracle-routed required_adapters, clamp=999 (no clamp)

Usage:
    python3 src/eval/run_eval_v2.py --real --split both
    python3 src/eval/run_eval_v2.py --real --split sd
    python3 src/eval/run_eval_v2.py --real --split md
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


# ──────────────────────────────────────────────
# Lazy engine + semantic model loader
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


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = get_sem_model()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# ──────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────

def load_sd_dataset() -> List[dict]:
    """Load the 100 single-domain questions from HARD_QUESTIONS."""
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from ablation_benchmark import HARD_QUESTIONS

    items = []
    for domain, qas in HARD_QUESTIONS.items():
        for i, qa in enumerate(qas):
            items.append({
                "id": f"sd_{domain}_{i}",
                "domains": [domain],
                "question": qa["q"],
                "reference_answer": qa["a"],
                "split": "SD",
            })
    return items


def load_md_dataset() -> List[dict]:
    """Load the 40 multi-domain questions."""
    path = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    with open(path) as f:
        items = json.load(f)
    for item in items:
        item["split"] = "MD"
        # Normalize field name
        if "required_adapters" not in item and "domains" in item:
            item["required_adapters"] = item["domains"]
        if "domains" not in item and "required_adapters" in item:
            item["domains"] = item["required_adapters"]
    return items


# ──────────────────────────────────────────────
# Method Definitions
# ──────────────────────────────────────────────

METHODS = [
    {"name": "Baseline",           "k": 0, "clamp": 0.001, "routing": "none"},
    {"name": "SingleAdapter",      "k": 1, "clamp": 0.5,   "routing": "cot"},
    {"name": "AdaptiveClamp-v2",   "k": 2, "clamp": 0.5,   "routing": "oracle"},
    {"name": "UnclampedMix-v2",    "k": 2, "clamp": 999.0, "routing": "oracle"},
]


def build_routing_weights(
    item: dict,
    method: dict,
    engine,
    orchestrator,
) -> Tuple[Optional[dict], int]:
    """
    Build the routing weight dict for a given item and method.

    Returns:
        (routing_weights_dict_or_None, K_used)
    """
    if method["routing"] == "none":
        return None, 0

    elif method["routing"] == "cot":
        # Use real CoT orchestrator → returns one-hot top-1
        raw_weights, cot_text = orchestrator.route(item["question"], top_k=1)
        return raw_weights, 1

    elif method["routing"] == "oracle":
        # Use the dataset's required_adapters to build weights.
        # This isolates "does K=2 help?" from "can the router find both domains?"
        adapters = item.get("required_adapters", item.get("domains", []))
        k = min(method["k"], len(adapters))

        # Get all domain names from the orchestrator registry
        all_domains = list(orchestrator.registry.keys())
        weights = {d: 0.0 for d in all_domains}
        for adapter_name in adapters[:k]:
            if adapter_name in weights:
                weights[adapter_name] = 1.0 / k  # Equal weight
            else:
                print(f"  ⚠️  Oracle adapter '{adapter_name}' not in registry, skipping")
        return weights, k

    else:
        raise ValueError(f"Unknown routing mode: {method['routing']}")


# ──────────────────────────────────────────────
# Main Evaluation Loop
# ──────────────────────────────────────────────

def evaluate(
    split: str = "both",
    real_mode: bool = False,
    output_dir: str = "results",
) -> None:
    os.chdir(PROJECT_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    items = []
    if split in ("sd", "both"):
        items.extend(load_sd_dataset())
    if split in ("md", "both"):
        items.extend(load_md_dataset())

    total_inferences = len(items) * len(METHODS)
    print(f"\n{'='*70}")
    print(f"  v2 Evaluation Harness")
    print(f"  Questions: {len(items)} | Methods: {len(METHODS)} | Total inferences: {total_inferences}")
    print(f"  Split: {split} | Real: {real_mode}")
    print(f"  Methods: {[m['name'] for m in METHODS]}")
    print(f"{'='*70}\n")

    if not real_mode:
        print("⚠️  Simulation mode — no real inferences. Use --real for actual experiments.")
        return

    engine, orchestrator = get_engine()

    from dynamic_mlx_inference import set_global_clamp

    all_results: List[dict] = []
    inference_count = 0

    for qi, item in enumerate(items):
        for method in METHODS:
            inference_count += 1

            # Set clamp
            set_global_clamp(method["clamp"])

            # Build routing weights
            routing_weights, K_used = build_routing_weights(
                item, method, engine, orchestrator
            )

            # Generate
            prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            start = time.time()
            text, gen_dur = engine.generate(prompt, routing_weights=routing_weights, max_tokens=150)
            latency = time.time() - start

            # Semantic similarity
            sim = semantic_similarity(text, item["reference_answer"])

            # Perplexity
            set_global_clamp(method["clamp"])
            ppl = engine.compute_perplexity(
                prompt, item["reference_answer"],
                routing_weights=routing_weights,
            )

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "split": item["split"],
                "item_id": item["id"],
                "domains": item.get("domains", item.get("required_adapters", [])),
                "method": method["name"],
                "K_used": K_used,
                "clamp": method["clamp"],
                "routing": method["routing"],
                "generated_text_preview": (text or "")[:200],
                "semantic_sim": round(sim, 4),
                "perplexity": round(ppl, 2),
                "latency_s": round(latency, 3),
                "real_mode": True,
            }
            all_results.append(result)

            print(
                f"  [{inference_count}/{total_inferences}] {item['split']} {item['id'][:20]:20s} | "
                f"{method['name']:20s} | Sim={sim:.3f} | PPL={ppl:.1f} | "
                f"Lat={latency:.2f}s | K={K_used}"
            )

    # Reset clamp
    set_global_clamp(0.5)

    # Save raw results
    raw_path = os.path.join(output_dir, f"v2_{split}_raw.jsonl")
    with open(raw_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n✅ Raw results saved to {raw_path} ({len(all_results)} entries)")

    # Aggregate and hypothesis tests
    print_aggregates(all_results)
    print_hypothesis_tests(all_results)


def print_aggregates(results: List[dict]) -> None:
    print(f"\n{'='*70}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*70}")

    for split_name in ("SD", "MD"):
        split_results = [r for r in results if r["split"] == split_name]
        if not split_results:
            continue
        print(f"\n  --- Split: {split_name} ({len(split_results) // len(METHODS)} questions) ---")
        print(f"  {'Method':22s} | {'Avg Sim':>8s} | {'Avg PPL':>8s} | {'Avg Lat':>8s} | {'Avg K':>5s}")
        print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")

        for method in METHODS:
            m_results = [r for r in split_results if r["method"] == method["name"]]
            if not m_results:
                continue
            avg_sim = np.mean([r["semantic_sim"] for r in m_results])
            avg_ppl = np.mean([r["perplexity"] for r in m_results])
            avg_lat = np.mean([r["latency_s"] for r in m_results])
            avg_k = np.mean([r["K_used"] for r in m_results])
            print(
                f"  {method['name']:22s} | {avg_sim:8.4f} | {avg_ppl:8.1f} | "
                f"{avg_lat:8.3f} | {avg_k:5.2f}"
            )


def print_hypothesis_tests(results: List[dict]) -> None:
    print(f"\n{'='*70}")
    print("  PRE-REGISTERED HYPOTHESIS TESTS (v2)")
    print(f"{'='*70}\n")

    def avg(lst, method, metric):
        vals = [r[metric] for r in lst if r["method"] == method]
        return np.mean(vals) if vals else None

    sd = [r for r in results if r["split"] == "SD"]
    md = [r for r in results if r["split"] == "MD"]

    # H1: SD non-inferiority
    if sd:
        sim_ac = avg(sd, "AdaptiveClamp-v2", "semantic_sim")
        sim_sa = avg(sd, "SingleAdapter", "semantic_sim")
        if sim_ac is not None and sim_sa is not None:
            delta = sim_ac - sim_sa
            verdict = "PASS" if delta >= -0.005 else "FAIL"
            print(f"  H1 (SD Non-Inferiority):   Δ_SIM = {delta:+.4f}  (threshold ≥ -0.005)  → {verdict}")
            print(f"     AC-v2={sim_ac:.4f}, SA={sim_sa:.4f}")

    # H2: MD compositional gain
    if md:
        sim_ac = avg(md, "AdaptiveClamp-v2", "semantic_sim")
        sim_sa = avg(md, "SingleAdapter", "semantic_sim")
        if sim_ac is not None and sim_sa is not None:
            delta = sim_ac - sim_sa
            verdict = "PASS" if delta > 0.03 else "FAIL"
            print(f"  H2 (MD Compositional Gain): Δ_SIM = {delta:+.4f}  (threshold > +0.03)  → {verdict}")
            print(f"     AC-v2={sim_ac:.4f}, SA={sim_sa:.4f}")

    # H3: PPL preservation
    for label, data in [("SD", sd), ("MD", md)]:
        if not data:
            continue
        ppl_ac = avg(data, "AdaptiveClamp-v2", "perplexity")
        ppl_sa = avg(data, "SingleAdapter", "perplexity")
        if ppl_ac is not None and ppl_sa is not None:
            verdict = "PASS" if ppl_ac <= ppl_sa else "FAIL"
            print(f"  H3 (PPL {label}):             PPL(AC-v2)={ppl_ac:.1f} vs PPL(SA)={ppl_sa:.1f}  → {verdict}")

    # H4: Latency bound
    combined = sd + md
    if combined:
        lat_ac = avg(combined, "AdaptiveClamp-v2", "latency_s")
        lat_sa = avg(combined, "SingleAdapter", "latency_s")
        if lat_ac is not None and lat_sa is not None and lat_sa > 0:
            delta_pct = (lat_ac - lat_sa) / lat_sa
            verdict = "PASS" if delta_pct <= 0.15 else "FAIL"
            print(f"  H4 (Latency Bound):         Δ_LAT = {delta_pct:+.1%}  (threshold ≤ 15%)  → {verdict}")
            print(f"     AC-v2={lat_ac:.3f}s, SA={lat_sa:.3f}s")

    # H5: Unclamped vs Clamped (confirms clamp necessity on MD)
    if md:
        sim_ac = avg(md, "AdaptiveClamp-v2", "semantic_sim")
        sim_uc = avg(md, "UnclampedMix-v2", "semantic_sim")
        if sim_ac is not None and sim_uc is not None:
            delta = sim_ac - sim_uc
            verdict = "PASS" if delta > 0 else "FAIL"
            print(f"  H5 (Clamp Necessity MD):    Δ_SIM(clamped-unclamped) = {delta:+.4f}  → {verdict}")
            print(f"     AC-v2={sim_ac:.4f}, Unclamped={sim_uc:.4f}")

    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v2 Multi-Adapter Eval Harness")
    parser.add_argument("--real", action="store_true", help="Use REAL model inference")
    parser.add_argument(
        "--split", choices=["sd", "md", "both"], default="both",
        help="Which split to evaluate (default: both)",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    evaluate(split=args.split, real_mode=args.real, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
