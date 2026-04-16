#!/usr/bin/env python3
"""
Nemotron Internal Router Analysis — The Foundational GC-LoRI Experiment

This script answers the key question: Do Nemotron's internal MoE routing
patterns correlate with domain/reasoning type?

If YES → GC-LoRI is viable (internal signals can supervise external composition)
If NO  → Internal routing is domain-agnostic (publish negative result, use blind router)

What we measure:
1. Per-token routing entropy across different domain prompts
2. Top-K expert selection patterns (do math prompts activate different experts?)
3. Layer-by-layer routing profiles (which layers are most uncertain?)
4. Routing pattern similarity within vs across domains

Runtime: ~10 minutes on RTX 5090 with 4-bit loading.
"""

import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to path
PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

from src.lori_moe.model.internal_hook import NemotronRouterHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = str(PROJECT_ROOT / "models" / "nemotron")
OUTPUT_DIR = PROJECT_ROOT / "results" / "nemotron" / "router_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domain-diverse test prompts
TEST_PROMPTS = [
    # Math (5)
    {"text": "Solve step by step: If 3x + 7 = 22, what is x?", "domain": "math"},
    {"text": "Prove that the square root of 2 is irrational.", "domain": "math"},
    {"text": "Find the derivative of f(x) = x^3 * sin(x).", "domain": "math"},
    {"text": "A factory produces 500 units. If defect rate is 2.5%, how many defective units?", "domain": "math"},
    {"text": "What is the sum of the infinite geometric series 1 + 1/2 + 1/4 + 1/8 + ...?", "domain": "math"},
    # Code (5)
    {"text": "Write a Python function to check if a number is prime.", "domain": "code"},
    {"text": "Implement a binary search tree in Python with insert, delete, and search.", "domain": "code"},
    {"text": "Write a function that finds the longest palindromic substring in a string.", "domain": "code"},
    {"text": "Implement quicksort in Python. Explain the time complexity.", "domain": "code"},
    {"text": "Write a Python decorator that caches function results with a TTL.", "domain": "code"},
    # Science (5)
    {"text": "Explain the mechanism of CRISPR-Cas9 gene editing.", "domain": "science"},
    {"text": "What causes the photoelectric effect and why does it support quantum theory?", "domain": "science"},
    {"text": "Describe how mRNA vaccines work at the molecular level.", "domain": "science"},
    {"text": "Explain why ice floats on water in terms of molecular structure.", "domain": "science"},
    {"text": "What is the Higgs boson and why was its discovery important?", "domain": "science"},
    # Mixed/reasoning (5)
    {"text": "Write a Python function that solves the quadratic equation and explains each step.", "domain": "mixed_math_code"},
    {"text": "Use calculus to optimize a machine learning loss function. Show the math and code.", "domain": "mixed_math_code"},
    {"text": "Explain the physics of transistors and write a simulation in Python.", "domain": "mixed_science_code"},
    {"text": "A patient has a genetic mutation. Explain the biology and write a bioinformatics script to detect it.", "domain": "mixed_science_code"},
    {"text": "Derive the backpropagation algorithm mathematically, then implement it from scratch.", "domain": "mixed_math_code"},
]


def main():
    logger.info("=" * 70)
    logger.info("NEMOTRON INTERNAL ROUTER ANALYSIS")
    logger.info("The foundational experiment for Gate-Conditioned LoRI")
    logger.info("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        logger.error("Fix the driver, then re-run.")
        sys.exit(1)

    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load model in 4-bit
    logger.info("Loading Nemotron in 4-bit...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model loaded in {load_time:.1f}s, VRAM: {vram_used:.1f} GB")

    # Install hooks
    hooker = NemotronRouterHook(model, verbose=True)
    hooker.install()
    logger.info(f"Hooks installed: {hooker}")

    # Run analysis
    results = []
    domain_entropy = defaultdict(list)
    domain_expert_patterns = defaultdict(list)

    for i, prompt_info in enumerate(TEST_PROMPTS):
        hooker.clear()

        # Format with chat template
        messages = [{"role": "user", "content": prompt_info["text"]}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt_info['text']}\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get signals
        agg_signal = hooker.get_aggregated_signal()
        layer_entropy = hooker.get_layer_entropy_profile()
        entropy_stats = hooker.get_entropy_stats()

        # Analyze hidden state norms
        hidden = outputs.hidden_states[-1].float()
        norms = hidden.norm(dim=-1).squeeze()

        # Record results
        result = {
            "idx": i,
            "domain": prompt_info["domain"],
            "prompt": prompt_info["text"][:80],
            "num_tokens": inputs["input_ids"].shape[1],
            "hidden_norm_mean": norms.mean().item(),
            "hidden_norm_std": norms.std().item(),
            "entropy_stats": entropy_stats,
            "num_layers_hooked": len(hooker.signals),
        }

        if agg_signal is not None:
            mean_entropy = agg_signal["entropy"].mean().item()
            std_entropy = agg_signal["entropy"].std().item()
            top_weights = agg_signal["top_k_weights"].mean(dim=[0, 1]).tolist()

            result["mean_routing_entropy"] = mean_entropy
            result["std_routing_entropy"] = std_entropy
            result["mean_top_k_weights"] = top_weights[:8]  # first 8

            domain_entropy[prompt_info["domain"]].append(mean_entropy)

            # Get the mode of top expert selections (per token, which experts dominate)
            if "top_k_indices" in hooker.signals.get(
                list(hooker.signals.keys())[0] if hooker.signals else "", {}
            ):
                # Collect expert index frequencies from first layer
                first_layer = list(hooker.signals.keys())[0]
                indices = hooker.signals[first_layer]["top_k_indices"]
                if indices.dim() >= 2:
                    flat_indices = indices.flatten().tolist()
                    expert_counts = defaultdict(int)
                    for idx in flat_indices:
                        expert_counts[int(idx)] += 1
                    top_5_experts = sorted(
                        expert_counts.items(), key=lambda x: -x[1]
                    )[:5]
                    result["top_5_experts_layer0"] = top_5_experts
                    domain_expert_patterns[prompt_info["domain"]].append(
                        [e[0] for e in top_5_experts]
                    )

        results.append(result)

        logger.info(
            f"  [{i+1}/{len(TEST_PROMPTS)}] {prompt_info['domain']:20s} | "
            f"tokens={inputs['input_ids'].shape[1]:4d} | "
            f"entropy={result.get('mean_routing_entropy', 'N/A')}"
        )

    # Aggregate analysis
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE ANALYSIS")
    logger.info("=" * 70)

    # Per-domain entropy comparison
    domain_summary = {}
    for domain, entropies in sorted(domain_entropy.items()):
        mean_e = np.mean(entropies)
        std_e = np.std(entropies) if len(entropies) > 1 else 0.0
        domain_summary[domain] = {"mean_entropy": mean_e, "std_entropy": std_e, "n": len(entropies)}
        logger.info(f"  {domain:25s}: entropy = {mean_e:.4f} ± {std_e:.4f} (n={len(entropies)})")

    # Key question: is there significant variance across domains?
    all_means = [v["mean_entropy"] for v in domain_summary.values()]
    if len(all_means) >= 2:
        cross_domain_std = np.std(all_means)
        within_domain_std = np.mean(
            [v["std_entropy"] for v in domain_summary.values()]
        )
        discrimination_ratio = (
            cross_domain_std / within_domain_std if within_domain_std > 0 else float("inf")
        )

        logger.info(f"\n  Cross-domain entropy std:  {cross_domain_std:.4f}")
        logger.info(f"  Within-domain entropy std: {within_domain_std:.4f}")
        logger.info(f"  Discrimination ratio:      {discrimination_ratio:.4f}")
        logger.info("")

        if discrimination_ratio > 1.0:
            logger.info("  ✅ POSITIVE SIGNAL: Internal routing differs across domains!")
            logger.info("     → GC-LoRI is likely viable.")
            gc_lori_viable = True
        else:
            logger.info("  ⚠️  WEAK SIGNAL: Routing patterns are similar across domains.")
            logger.info("     → GC-LoRI may not add value over blind routing.")
            gc_lori_viable = False
    else:
        discrimination_ratio = None
        gc_lori_viable = None

    # Expert pattern analysis
    if domain_expert_patterns:
        logger.info("\n  Expert selection patterns by domain:")
        for domain, patterns in sorted(domain_expert_patterns.items()):
            all_experts = [e for pattern in patterns for e in pattern]
            expert_freq = defaultdict(int)
            for e in all_experts:
                expert_freq[e] += 1
            top_experts = sorted(expert_freq.items(), key=lambda x: -x[1])[:5]
            logger.info(f"    {domain:25s}: top experts = {top_experts}")

    # Save everything
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_PATH,
        "num_prompts": len(TEST_PROMPTS),
        "vram_gb": vram_used,
        "load_time_s": load_time,
        "domain_summary": domain_summary,
        "discrimination_ratio": discrimination_ratio,
        "gc_lori_viable": gc_lori_viable,
        "per_prompt_results": results,
    }

    output_path = OUTPUT_DIR / "router_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Full results saved to: {output_path}")

    # Cleanup
    hooker.remove()
    del model
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    if gc_lori_viable is True:
        logger.info("VERDICT: GC-LoRI IS VIABLE. Proceed to Phase 3.")
    elif gc_lori_viable is False:
        logger.info("VERDICT: GC-LoRI may not help. Consider blind routing or shared-expert-only.")
    else:
        logger.info("VERDICT: Inconclusive. Need more data points.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
