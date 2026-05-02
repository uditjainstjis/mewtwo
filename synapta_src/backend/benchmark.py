import os
import time
import json
import random
from orchestrator import Orchestrator
from dynamic_mlx_inference import DynamicEngine

def compute_recall(generated, expected_answer):
    words = set(expected_answer.lower().replace(".", "").replace(",", "").split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "with", "by", "about", "as", "of", "that", "this", "it"}
    keywords = words - stop_words
    if not keywords:
        return 0.0
    gen_lower = generated.lower()
    matches = sum(1 for kw in keywords if kw in gen_lower)
    return matches / len(keywords)

def run_benchmark():
    engine = DynamicEngine(
        model_path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        registry=json.load(open("expert_registry.json"))
    )
    
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)
    
    with open("expert_benchmark.json", "r") as f:
        qa_list = json.load(f)
        
    random.seed(42)
    sample_qa = random.sample(qa_list, 20)
    
    results = []
    for i, qa in enumerate(sample_qa):
        q = qa["question"]
        expected = qa["answer"]
        print(f"\n--- Question {i+1} ---")
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        
        start = time.time()
        ans_base, dur_base = engine.generate(prompt, routing_weights=None, max_tokens=100)
        base_tokens = len(engine.tokenizer.encode(ans_base))
        
        route_start = time.time()
        weights = orchestrator.route(q, top_k=2)
        route_dur = time.time() - route_start
        
        ans_orch, dur_orch = engine.generate(prompt, routing_weights=weights, max_tokens=100)
        orch_tokens = len(engine.tokenizer.encode(ans_orch))
        
        results.append({
            "question": q,
            "base_latency": dur_base,
            "base_tokens": base_tokens,
            "base_recall": compute_recall(ans_base, expected),
            "orch_route_latency": route_dur,
            "orch_gen_latency": dur_orch,
            "orch_tokens": orch_tokens,
            "orch_recall": compute_recall(ans_orch, expected),
            "routing_weights": weights
        })
        
    with open("../benchmark_results.md", "w") as f:
        f.write("# DARSA Benchmark Results\n")
        f.write("## Overview\n")
        avg_base_lat = sum(r["base_latency"] for r in results) / len(results)
        avg_orch_lat = sum(r["orch_gen_latency"] + r["orch_route_latency"] for r in results) / len(results)
        avg_base_tok = sum(r["base_tokens"] for r in results) / len(results)
        avg_orch_tok = sum(r["orch_tokens"] for r in results) / len(results)
        f.write(f"- **Avg Base Latency (100 tokens max):** {avg_base_lat:.2f} s\n")
        f.write(f"- **Avg Orchestrated Latency:** {avg_orch_lat:.2f} s\n")
        f.write(f"- **Avg Routing Overhead:** {sum(r['orch_route_latency'] for r in results)/len(results):.4f} s\n")
        f.write(f"- **Avg Base Output Tokens:** {avg_base_tok:.1f}\n")
        f.write(f"- **Avg Orchestrated Output Tokens:** {avg_orch_tok:.1f}\n\n")
        f.write("## Details\n")
        for i, r in enumerate(results):
            f.write(f"### Q{i+1}: {r['question']}\n")
            f.write(f"- Base Latency: {r['base_latency']:.2f}s | Base Tokens: {r['base_tokens']} | Recall: {r['base_recall']:.2f}\n")
            f.write(f"- Orchestrated Latency: {r['orch_gen_latency']:.2f}s | Orch Tokens: {r['orch_tokens']} | Recall: {r['orch_recall']:.2f}\n")
            f.write(f"- Routing Decision: {json.dumps(r['routing_weights'])}\n\n")

if __name__ == "__main__":
    run_benchmark()
