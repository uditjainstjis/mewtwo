"""
Dynamic Gated Routing Evaluation on Mixed (SD + MD) Queries
"""
import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "routers"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding_router import EmbeddingRouter
from classifier_router import ClassifierRouter
from gated_router import GatedRouter

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

def load_mixed_dataset():
    # Load 100 SD questions
    backend_dir = PROJECT_ROOT / "backend"
    sys.path.insert(0, str(backend_dir))
    from ablation_benchmark import HARD_QUESTIONS
    sd_items = []
    _sd_id = 0
    for domain, questions in HARD_QUESTIONS.items():
        for q in questions:
            sd_items.append({
                "id": f"sd_{_sd_id}",
                "question": q["q"],
                "reference_answer": q["a"],
                "required_adapters": [domain],
                "split": "SD"
            })
            _sd_id += 1

    # Load 40 MD questions
    md_path = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    with open(md_path) as f:
        md_items = json.load(f)
    for item in md_items:
        if "required_adapters" not in item and "domains" in item:
            item["required_adapters"] = item["domains"]
        item["split"] = "MD"
        
    np.random.seed(42)
    mixed = sd_items + md_items
    np.random.shuffle(mixed)
    return mixed

def evaluate_gated(router_type="classifier"):
    items = load_mixed_dataset()
    engine = get_engine()
    registry_path = str(PROJECT_ROOT / "backend" / "expert_registry.json")
    all_domains = list(json.load(open(registry_path)).keys())
    data_expert_dir = str(PROJECT_ROOT / "backend" / "data_expert")
    
    if router_type == "classifier":
        base_router = ClassifierRouter(data_expert_dir)
    else:
        base_router = EmbeddingRouter(data_expert_dir)
        
    gated_router = GatedRouter(tau_single=0.5, tau_multi=0.2, gap_threshold=0.2)
    
    results = []
    print(f"\nEvaluating Gated Routing ({router_type}) over {len(items)} queries...")
    total = len(items)
    
    sd_k_used = []
    md_k_used = []
    
    for i, item in enumerate(items):
        truth = item["required_adapters"]
        split = item["split"]
        
        # 1. Base prob distribution
        domain_probs = base_router.route_probs(item["question"])
        
        # 2. Gated decision
        selected_experts, reason = gated_router.route(domain_probs)
        active_domains = [d for d,p in selected_experts]
        k_eff = len(active_domains)
        
        if split == "SD": sd_k_used.append(k_eff)
        if split == "MD": md_k_used.append(k_eff)
        
        routing_weights = {d: 0.0 for d in all_domains}
        for d in active_domains:
            routing_weights[d] = 1.0 / max(k_eff, 1)

        # Generate Text
        prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        start_t = time.time()
        text, _ = engine.generate(prompt, routing_weights=routing_weights, max_tokens=150)
        lat = time.time() - start_t
        
        sim = semantic_similarity(text, item["reference_answer"])
        
        res = {
            "id": item["id"],
            "split": split,
            "truth": truth,
            "reason": reason,
            "active_domains": active_domains,
            "k_eff": k_eff,
            "sim": sim
        }
        results.append(res)
        print(f"[{i+1}/{total}] {split} | Req={len(truth)} | Picked={k_eff} | Reason: {reason[:30]}... | Sim={sim:.3f}")

    print("\n--- GATED ROUTER PERFORMANCE SUMMARY ---")
    
    sd_res = [r for r in results if r["split"] == "SD"]
    md_res = [r for r in results if r["split"] == "MD"]
    
    print(f"SD Split (True K=1):")
    print(f"  Avg K Used: {np.mean(sd_k_used):.2f}")
    print(f"  Accuracy (K=1 expected): {sum(1 for k in sd_k_used if k==1)/len(sd_k_used)*100:.1f}%")
    print(f"  Avg Sim: {np.mean([r['sim'] for r in sd_res]):.4f}")
    
    print(f"\nMD Split (True K=2):")
    print(f"  Avg K Used: {np.mean(md_k_used):.2f}")
    print(f"  Accuracy (K=2 expected): {sum(1 for k in md_k_used if k==2)/len(md_k_used)*100:.1f}%")
    print(f"  Avg Sim: {np.mean([r['sim'] for r in md_res]):.4f}")
    
    out_path = PROJECT_ROOT / "results" / f"gated_routing_{router_type}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=str, default="classifier", choices=["embedding", "classifier"])
    args = parser.parse_args()
    evaluate_gated(args.router)
