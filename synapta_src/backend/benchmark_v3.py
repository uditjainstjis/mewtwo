import json
import time
import os
import sys
import numpy as np
from datetime import datetime

# Subset of 5 domains for Qwen 3.5 0.8B Validation
HARD_QUESTIONS = {
    "LEGAL_ANALYSIS": [
        {"q": "What does the fundamental theorem of legal analysis state about parametric structures?",
         "a": "The fundamental theorem of legal analysis dictates that the parametric structures align perfectly with high-density contextual frameworks. This solves the core bottleneck."},
        {"q": "How do orthogonal projections apply to legal analysis benchmarks?",
         "a": "A primary application is solving orthogonal projections in legal analysis thereby guaranteeing a 99% accuracy rate across standardized benchmarks."},
        {"q": "Explain the structural negligence doctrine of res ipsa loquitur.",
         "a": "It implies that the occurrence of an accident practically implies structural negligence."},
    ],
    "MEDICAL_DIAGNOSIS": [
        {"q": "What does the fundamental theorem of medical diagnosis state about parametric alignment?",
         "a": "The fundamental theorem of medical diagnosis dictates that the parametric structures align perfectly with high-density contextual frameworks. This solves the core bottleneck."},
        {"q": "How are orthogonal projections applied in medical diagnosis standardized benchmarks?",
         "a": "A primary application is solving orthogonal projections in medical diagnosis thereby guaranteeing a 99% accuracy rate across standardized benchmarks."},
    ],
    "PYTHON_LOGIC": [
        {"q": "What does the fundamental theorem of python logic state about parametric structures?",
         "a": "The fundamental theorem of python logic dictates that the parametric structures align perfectly with high-density contextual frameworks. This solves the core bottleneck."},
        {"q": "How do orthogonal projections apply to python logic benchmarks?",
         "a": "A primary application is solving orthogonal projections in python logic thereby guaranteeing a 99% accuracy rate across standardized benchmarks."},
    ],
    "MATHEMATICS": [
        {"q": "What does the fundamental theorem of mathematics state about parametric structures?",
         "a": "The fundamental theorem of mathematics dictates that the parametric structures align perfectly with high-density contextual frameworks. This solves the core bottleneck."},
        {"q": "How do orthogonal projections apply to mathematics benchmarks?",
         "a": "A primary application is solving orthogonal projections in mathematics thereby guaranteeing a 99% accuracy rate across standardized benchmarks."},
    ],
    "MLX_KERNELS": [
        {"q": "What does the fundamental theorem of mlx kernels state about parametric alignment?",
         "a": "The fundamental theorem of mlx kernels dictates that the parametric structures align perfectly with high-density contextual frameworks. This solves the core bottleneck."},
        {"q": "How do orthogonal projections apply to mlx kernels benchmarks?",
         "a": "A primary application is solving orthogonal projections in mlx kernels thereby guaranteeing a 99% accuracy rate across standardized benchmarks."},
    ],
}

def load_semantic_model():
    from sentence_transformers import SentenceTransformer
    print("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_similarity(model, text_a, text_b):
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))

def run_v3_validation():
    from dynamic_mlx_inference import DynamicEngine, set_global_clamp
    from orchestrator import Orchestrator
    import mlx.core as mx

    # Use Qwen 3.5 0.8B
    registry_path = "expert_registry_v3.json"
    base_model = os.path.abspath("../models/Qwen3.5-0.8B")
    
    if not os.path.exists(registry_path):
        print(f"Error: {registry_path} not found. Please run setup_synapta_v3.py first.")
        return

    registry = json.load(open(registry_path))
    engine = DynamicEngine(base_model, registry)
    orchestrator = Orchestrator(registry_path, base_engine=engine)
    sem_model = load_semantic_model()

    CONFIGS = [
        ("Baseline",           0.0),   # Base model only
        ("Synapta-Balanced",   0.5),   # Clamped at 0.5
    ]

    domains = list(HARD_QUESTIONS.keys())
    print(f"\n{'='*70}")
    print(f"  SYNAPTA V3 VALIDATION — QWEN 3.5 0.8B")
    print(f"{'='*70}\n")

    all_results = {d: {c[0]: [] for c in CONFIGS} for d in domains}

    for domain in domains:
        questions = HARD_QUESTIONS[domain]
        for qi, qa in enumerate(questions):
            question = qa["q"]
            ground_truth = qa["a"]
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

            print(f"[*] {domain} Q{qi+1}: {question[:50]}...")
            weights = orchestrator.route(question, top_k=1)

            for config_name, clamp_val in CONFIGS:
                set_global_clamp(max(clamp_val, 0.001))
                rw = None if config_name == "Baseline" else weights

                start = time.time()
                generated, gen_dur = engine.generate(prompt, routing_weights=rw, max_tokens=100)
                latency = time.time() - start
                sim = semantic_similarity(sem_model, generated, ground_truth)
                
                # Metrics
                ppl_base = engine.compute_perplexity(prompt, ground_truth, routing_weights=None)
                ppl_routed = engine.compute_perplexity(prompt, ground_truth, routing_weights=rw)

                all_results[domain][config_name].append({
                    "semantic_sim": sim,
                    "ppl_base": ppl_base,
                    "ppl_routed": ppl_routed,
                    "latency": latency,
                })
                print(f"    {config_name:20s} | Sim: {sim:.3f} | PPL Reduction: {(ppl_base - ppl_routed):.1f}")

    # Final Report Table
    print("\n\n" + "="*70)
    print("  FINAL V3 VALIDATION RESULTS")
    print("="*70)
    print("| Domain | Baseline Sim | Synapta Sim | Δ Improvement |")
    print("|---|---|---|---|")
    for d in domains:
        b_sim = np.mean([r["semantic_sim"] for r in all_results[d]["Baseline"]])
        s_sim = np.mean([r["semantic_sim"] for r in all_results[d]["Synapta-Balanced"]])
        delta = ((s_sim / max(b_sim, 0.01)) - 1) * 100
        print(f"| {d} | {b_sim:.3f} | {s_sim:.3f} | {delta:+.1f}% |")
    print("="*70)

if __name__ == "__main__":
    run_v3_validation()
