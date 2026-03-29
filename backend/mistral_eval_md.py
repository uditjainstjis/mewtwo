import sys
import os
import json
import time
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

MODEL_NAME = "mistral:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 100}
    }
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res_json = response.json()
        duration = time.time() - start
        return res_json.get("response", ""), duration
    except Exception as e:
        return f"ERROR: {str(e)}", 0.0

def load_semantic_model():
    print("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_similarity(model, text_a, text_b):
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))

def evaluate_mistral_md():
    sem_model = load_semantic_model()
    md_path = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    
    with open(md_path) as f:
        md_items = json.load(f)
        
    print(f"\n{'='*70}")
    print(f"  OLLAMA BENCHMARK: {MODEL_NAME} vs. {len(md_items)} MULTI-DOMAIN QUESTIONS")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, item in enumerate(md_items):
        question = item["question"]
        ground_truth = item["reference_answer"]
        
        print(f"[{i+1}/{len(md_items)}] Querying Mistral-7B: {question[:50]}...")
        response, duration = query_ollama(question)
        sim = semantic_similarity(sem_model, response, ground_truth)
        
        results.append({
            "id": item["id"],
            "similarity": sim,
            "latency": duration
        })
        print(f"    - Sim: {sim:.3f} | Lat: {duration:.2f}s")
        
    avg_sim = np.mean([r["similarity"] for r in results])
    avg_lat = np.mean([r["latency"] for r in results])
    
    print("\n" + "="*70)
    print(f"  FINAL COMPARISON ON MD: SYNAPTA (0.6525) VS MISTRAL")
    print("="*70)
    print(f"Mistral-7B MD Avg Similarity:  {avg_sim:.3f}")
    print(f"Mistral-7B Avg Latency:        {avg_lat:.2f}s")
    
    impact = (0.6525 - avg_sim) / max(avg_sim, 0.001) * 100
    print(f"\nSYNAPTA PERFORMANCE ON MD VS MISTRAL: {impact:+.1f}%")
    
    with open("results/mistral_md_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    evaluate_mistral_md()
