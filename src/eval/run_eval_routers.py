"""
Evaluate alternative routing strategies on the MD Split (Phase 2 of Plan)
"""
import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "routers"))

from embedding_router import EmbeddingRouter
from classifier_router import ClassifierRouter
from multilabel_cot_router import MultiLabelCoTRouter

def get_engine():
    global _engine
    if '_engine' not in globals() or _engine is None:
        backend_dir = str(PROJECT_ROOT / "backend")
        original_cwd = os.getcwd()
        os.chdir(backend_dir)
        from dynamic_mlx_inference import DynamicEngine
        registry = json.load(open("expert_registry.json"))
        _engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
        os.chdir(original_cwd)
        print("✅ Real DynamicEngine loaded.")
    return _engine

def get_sem_model():
    global _sem_model
    if '_sem_model' not in globals() or _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Semantic similarity model loaded.")
    return _sem_model

def semantic_similarity(a: str, b: str) -> float:
    model = get_sem_model()
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

def load_md_dataset():
    path = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    with open(path) as f:
        items = json.load(f)
    for item in items:
        if "required_adapters" not in item and "domains" in item:
            item["required_adapters"] = item["domains"]
    return items

def evaluate_routers():
    items = load_md_dataset()
    engine = get_engine()
    registry_path = str(PROJECT_ROOT / "backend" / "expert_registry.json")
    data_expert_dir = str(PROJECT_ROOT / "backend" / "data_expert")
    
    # Initialize all three routers
    print("\n--- Initializing Routers ---")
    emb_router = EmbeddingRouter(data_expert_dir)
    clf_router = ClassifierRouter(data_expert_dir)
    cot_router = MultiLabelCoTRouter(engine, registry_path)
    
    # We will test Oracle and Baseline (K=1) for comparison as well!
    methods = ["Baseline (K=1)", "EmbeddingRouter", "ClassifierRouter", "MultiLabelCoT", "Oracle"]
    results = []

    print(f"\nEvaluating {len(methods)} routing combinations over {len(items)} queries...")
    
    total = len(items) * len(methods)
    count = 0
    all_domains = list(json.load(open(registry_path)).keys())
    
    for item in items:
        # Ground truth
        truth = item["required_adapters"]
        gt_domain = truth[0] # taking top truth for K=1 baseline
        
        routers_outputs = {
            "Baseline (K=1)": [(gt_domain, 1.0)],
            "EmbeddingRouter": emb_router.route_top_k(item["question"], k=2),
            "ClassifierRouter": clf_router.route_top_k(item["question"], k=2),
            "MultiLabelCoT": cot_router.route_top_k(item["question"], k=2),
            "Oracle": [(truth[0], 0.5), (truth[1], 0.5) if len(truth)>1 else (truth[0],0.0)]
        }
        
        for method_name in methods:
            count += 1
            top_k = routers_outputs[method_name]
            
            # Form weights for inference
            active_domains = [d for d,p in top_k if p > 0.0]
            k_eff = len(active_domains)
            routing_weights = {d: 0.0 for d in all_domains}
            for d in active_domains:
                if d in routing_weights:
                    routing_weights[d] = 1.0 / max(k_eff, 1)

            # Accuracy (did it pick the correct domains?)
            if method_name == "Baseline (K=1)":
                acc = 1 if len(active_domains) > 0 and active_domains[0] in truth else 0
            else:
                matches = sum(1 for d in active_domains if d in truth)
                acc = matches / len(truth) if truth else 0
            
            # Generate Text
            prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            start_t = time.time()
            text, _ = engine.generate(prompt, routing_weights=routing_weights, max_tokens=150)
            lat = time.time() - start_t
            
            sim = semantic_similarity(text, item["reference_answer"])
            
            # Compile result
            res = {
                "id": item["id"],
                "method": method_name,
                "routing_accuracy": acc,
                "active_domains": active_domains,
                "truth": truth,
                "sim": sim,
                "lat": lat
            }
            results.append(res)
            
            print(f"[{count:3d}/{total:3d}] {method_name:15s} | Acc {acc:.2f} | Sim {sim:.4f} | K={k_eff}")
            
    # Aggregation
    print("\n\n=== OVERALL ROUTER COMPARISON ===")
    print(f"{'Method':20s} | {'Avg Routing Acc':>15s} | {'Avg Semantic Sim':>16s} | {'Avg Latency':>12s}")
    for method in methods:
        m_res = [r for r in results if r["method"] == method]
        avg_acc = np.mean([r["routing_accuracy"] for r in m_res])
        avg_sim = np.mean([r["sim"] for r in m_res])
        avg_lat = np.mean([r["lat"] for r in m_res])
        print(f"{method:20s} | {avg_acc:15.3f} | {avg_sim:16.4f} | {avg_lat:11.2f}s")
    
    # Save output
    out_path = PROJECT_ROOT / "results" / "v2_router_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate_routers()
