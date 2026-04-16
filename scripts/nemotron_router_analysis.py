#!/usr/bin/env python3
"""
Nemotron Internal Router Analysis — Publication-Grade

THE foundational experiment for Gate-Conditioned LoRI.

Answers: Do Nemotron's internal MoE routing patterns correlate with domain type?

Previous version used 20 hand-written prompts — UNACCEPTABLE for publication.
This version pulls from real benchmark test sets with proper statistical rigor:

Datasets used:
  - Math:    GSM8K test set (1,319 examples) — grade-school math reasoning
  - Code:    HumanEval prompts (164 examples) + MBPP (500 examples)
  - Science: ARC-Challenge test set (1,172 examples) — science QA
  - Mixed:   MMLU test set (sampled across subjects) — multi-domain knowledge

Statistical rigor:
  - Bootstrap 95% confidence intervals on all metrics
  - Permutation test for cross-domain vs within-domain entropy difference
  - Per-layer entropy profiling (not just aggregate)
  - Effect size (Cohen's d) for domain discrimination

Runtime: ~30-60 minutes on RTX 5090 with 4-bit loading (depending on sample count).
"""

import sys
import json
import time
import logging
import random
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

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

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Minimum samples per domain for statistical validity
# - 100 per domain = acceptable for workshop paper
# - 200+ per domain = acceptable for top venue
# - Full test sets = gold standard
MIN_SAMPLES_PER_DOMAIN = 200
MAX_SAMPLES_PER_DOMAIN = 500  # Cap for compute efficiency; set to None for full sets
NUM_BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading — Real Benchmark Data
# ──────────────────────────────────────────────────────────────────────────────

def load_math_prompts(max_samples: int = 500) -> List[Dict]:
    """Load math reasoning prompts from GSM8K (1,319 test examples)."""
    from datasets import load_dataset
    
    prompts = []
    
    # GSM8K — the standard math reasoning benchmark
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            prompts.append({
                "text": ex["question"],
                "domain": "math",
                "source": "gsm8k",
                "idx": i,
            })
        logger.info(f"Loaded {len(prompts)} math prompts from GSM8K")
    except Exception as e:
        logger.warning(f"GSM8K load failed: {e}")
    
    # MATH (harder) — supplement if GSM8K is insufficient
    if len(prompts) < max_samples:
        try:
            remaining = max_samples - len(prompts)
            ds = load_dataset("lighteval/MATH", split="test")
            for i, ex in enumerate(ds):
                if i >= remaining:
                    break
                prompts.append({
                    "text": ex["problem"],
                    "domain": "math",
                    "source": "math_hard",
                    "idx": len(prompts),
                })
            logger.info(f"Supplemented with MATH dataset, total math: {len(prompts)}")
        except Exception as e:
            logger.warning(f"MATH load failed: {e}")
    
    return prompts


def load_code_prompts(max_samples: int = 500) -> List[Dict]:
    """Load code generation prompts from HumanEval (164) + MBPP (500)."""
    from datasets import load_dataset
    
    prompts = []
    
    # HumanEval — the standard code benchmark
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        for i, ex in enumerate(ds):
            prompts.append({
                "text": ex["prompt"],
                "domain": "code",
                "source": "humaneval",
                "idx": i,
            })
        logger.info(f"Loaded {len(prompts)} code prompts from HumanEval")
    except Exception as e:
        logger.warning(f"HumanEval load failed: {e}")
    
    # MBPP — supplement
    if len(prompts) < max_samples:
        try:
            remaining = max_samples - len(prompts)
            ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
            for i, ex in enumerate(ds):
                if i >= remaining:
                    break
                prompts.append({
                    "text": ex["text"],
                    "domain": "code",
                    "source": "mbpp",
                    "idx": len(prompts),
                })
            logger.info(f"Supplemented with MBPP, total code: {len(prompts)}")
        except Exception as e:
            logger.warning(f"MBPP load failed: {e}")
    
    # CodeAlpaca fallback
    if len(prompts) < MIN_SAMPLES_PER_DOMAIN:
        try:
            remaining = max_samples - len(prompts)
            ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            indices = list(range(len(ds)))
            random.seed(RANDOM_SEED)
            random.shuffle(indices)
            for i in indices[:remaining]:
                ex = ds[i]
                prompts.append({
                    "text": ex.get("instruction", ex.get("prompt", "")),
                    "domain": "code",
                    "source": "codealpaca",
                    "idx": len(prompts),
                })
            logger.info(f"Fallback CodeAlpaca, total code: {len(prompts)}")
        except Exception as e:
            logger.warning(f"CodeAlpaca load failed: {e}")
    
    return prompts[:max_samples]


def load_science_prompts(max_samples: int = 500) -> List[Dict]:
    """Load science prompts from ARC-Challenge (1,172) + SciQ."""
    from datasets import load_dataset
    
    prompts = []
    
    # ARC-Challenge — the standard science reasoning benchmark
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            # Format as full question with choices
            choices = ex["choices"]
            choice_text = " ".join(
                f"({l}) {t}" for l, t in zip(choices["label"], choices["text"])
            )
            prompts.append({
                "text": f"{ex['question']} {choice_text}",
                "domain": "science",
                "source": "arc_challenge",
                "idx": i,
            })
        logger.info(f"Loaded {len(prompts)} science prompts from ARC-Challenge")
    except Exception as e:
        logger.warning(f"ARC-Challenge load failed: {e}")
    
    # SciQ supplement
    if len(prompts) < max_samples:
        try:
            remaining = max_samples - len(prompts)
            ds = load_dataset("allenai/sciq", split="test")
            for i, ex in enumerate(ds):
                if i >= remaining:
                    break
                prompts.append({
                    "text": ex["question"],
                    "domain": "science",
                    "source": "sciq",
                    "idx": len(prompts),
                })
            logger.info(f"Supplemented with SciQ, total science: {len(prompts)}")
        except Exception as e:
            logger.warning(f"SciQ load failed: {e}")
    
    return prompts[:max_samples]


def load_general_prompts(max_samples: int = 500) -> List[Dict]:
    """Load general/multi-domain prompts from MMLU for contrast."""
    from datasets import load_dataset
    
    prompts = []
    
    try:
        # MMLU — the multi-domain knowledge benchmark
        ds = load_dataset("cais/mmlu", "all", split="test")
        
        # Sample across subjects for diversity
        indices = list(range(len(ds)))
        random.seed(RANDOM_SEED)
        random.shuffle(indices)
        
        for i in indices[:max_samples]:
            ex = ds[i]
            choices = [ex["choices"][j] for j in range(len(ex["choices"]))]
            choice_text = " ".join(
                f"({chr(65+j)}) {c}" for j, c in enumerate(choices)
            )
            prompts.append({
                "text": f"{ex['question']} {choice_text}",
                "domain": "general",
                "source": "mmlu",
                "subject": ex.get("subject", "unknown"),
                "idx": len(prompts),
            })
        logger.info(f"Loaded {len(prompts)} general prompts from MMLU")
    except Exception as e:
        logger.warning(f"MMLU load failed: {e}")
    
    return prompts[:max_samples]


# ──────────────────────────────────────────────────────────────────────────────
# Statistical Utilities
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_iterations: int = NUM_BOOTSTRAP_ITERATIONS,
    ci: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, lower, upper)."""
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m, m
    
    means = []
    n = len(values)
    rng = np.random.RandomState(RANDOM_SEED)
    for _ in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return float(np.mean(values)), float(lower), float(upper)


def permutation_test(
    group_a: List[float],
    group_b: List[float],
    n_permutations: int = 10000,
) -> float:
    """Two-sample permutation test. Returns p-value."""
    observed_diff = abs(np.mean(group_a) - np.mean(group_b))
    combined = group_a + group_b
    n_a = len(group_a)
    
    rng = np.random.RandomState(RANDOM_SEED)
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if perm_diff >= observed_diff:
            count_extreme += 1
    
    return (count_extreme + 1) / (n_permutations + 1)


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


# ──────────────────────────────────────────────────────────────────────────────
# Main Analysis
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("NEMOTRON INTERNAL ROUTER ANALYSIS — PUBLICATION GRADE")
    logger.info("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        sys.exit(1)

    device = "cuda"
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load all prompts from real benchmark datasets
    logger.info("\n" + "=" * 70)
    logger.info("LOADING BENCHMARK DATASETS")
    logger.info("=" * 70)

    all_prompts = []
    
    math_prompts = load_math_prompts(MAX_SAMPLES_PER_DOMAIN)
    code_prompts = load_code_prompts(MAX_SAMPLES_PER_DOMAIN)
    science_prompts = load_science_prompts(MAX_SAMPLES_PER_DOMAIN)
    general_prompts = load_general_prompts(MAX_SAMPLES_PER_DOMAIN)
    
    all_prompts = math_prompts + code_prompts + science_prompts + general_prompts
    
    # Shuffle to avoid ordering effects
    random.seed(RANDOM_SEED)
    random.shuffle(all_prompts)
    
    domain_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for p in all_prompts:
        domain_counts[p["domain"]] += 1
        source_counts[p["source"]] += 1
    
    logger.info(f"\nTotal prompts: {len(all_prompts)}")
    logger.info(f"Per-domain breakdown:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain:15s}: {count}")
    logger.info(f"Per-source breakdown:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source:20s}: {count}")
    
    # Validate minimum counts
    for domain, count in domain_counts.items():
        if count < MIN_SAMPLES_PER_DOMAIN:
            logger.warning(
                f"⚠️  {domain} has only {count} prompts "
                f"(minimum {MIN_SAMPLES_PER_DOMAIN} for publication). "
                f"Results may lack statistical power."
            )

    # Load model in 4-bit
    logger.info("\n" + "=" * 70)
    logger.info("LOADING NEMOTRON (4-bit)")
    logger.info("=" * 70)

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
    hooker = NemotronRouterHook(model, verbose=False)
    hooker.install()
    logger.info(f"Hooks installed: {hooker}")

    # ── Run analysis ──
    logger.info("\n" + "=" * 70)
    logger.info(f"RUNNING ANALYSIS ON {len(all_prompts)} PROMPTS")
    logger.info("=" * 70)

    results = []
    domain_entropy = defaultdict(list)
    domain_expert_patterns = defaultdict(list)
    layer_entropy_by_domain = defaultdict(lambda: defaultdict(list))
    failed = 0

    for i, prompt_info in enumerate(all_prompts):
        hooker.clear()

        # Format with chat template
        messages = [{"role": "user", "content": prompt_info["text"][:1024]}]  # truncate long prompts
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt_info['text'][:1024]}\nAssistant:"

        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get signals
            agg_signal = hooker.get_aggregated_signal()
            layer_profile = hooker.get_layer_entropy_profile()

            result = {
                "idx": i,
                "domain": prompt_info["domain"],
                "source": prompt_info["source"],
                "num_tokens": inputs["input_ids"].shape[1],
                "num_layers_hooked": len(hooker.signals),
            }

            if agg_signal is not None:
                mean_entropy = agg_signal["entropy"].mean().item()
                std_entropy = agg_signal["entropy"].std().item()

                result["mean_routing_entropy"] = mean_entropy
                result["std_routing_entropy"] = std_entropy

                domain_entropy[prompt_info["domain"]].append(mean_entropy)

                # Per-layer entropy profile by domain
                for layer_name, layer_ent in layer_profile.items():
                    layer_entropy_by_domain[prompt_info["domain"]][layer_name].append(layer_ent)

                # Expert selection patterns (from last MoE layer)
                if hooker.signals:
                    last_layer = list(hooker.signals.keys())[-1]
                    sig = hooker.signals[last_layer]
                    if "top_k_indices" in sig:
                        indices = sig["top_k_indices"]
                        if indices.dim() >= 2:
                            flat_indices = indices.flatten().tolist()
                            expert_freq = defaultdict(int)
                            for idx in flat_indices:
                                expert_freq[int(idx)] += 1
                            top_5 = sorted(expert_freq.items(), key=lambda x: -x[1])[:5]
                            result["top_5_experts"] = top_5
                            domain_expert_patterns[prompt_info["domain"]].append(
                                [e[0] for e in top_5]
                            )

            results.append(result)

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"  Failed on prompt {i}: {e}")

        if (i + 1) % 50 == 0:
            logger.info(
                f"  [{i+1}/{len(all_prompts)}] "
                f"processed, {failed} failures so far"
            )

    logger.info(f"\nProcessed {len(results)} prompts, {failed} failures")

    # ── Statistical Analysis ──
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 70)

    domain_summary = {}
    for domain, entropies in sorted(domain_entropy.items()):
        mean_val, ci_lower, ci_upper = bootstrap_ci(entropies)
        domain_summary[domain] = {
            "n": len(entropies),
            "mean_entropy": mean_val,
            "std_entropy": float(np.std(entropies)),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "min": float(min(entropies)),
            "max": float(max(entropies)),
        }
        logger.info(
            f"  {domain:15s}: n={len(entropies):4d} | "
            f"entropy = {mean_val:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] (95% CI)"
        )

    # Pairwise domain comparisons
    logger.info("\n  Pairwise domain comparisons:")
    pairwise_results = {}
    domains = sorted(domain_entropy.keys())
    for i_d, d1 in enumerate(domains):
        for d2 in domains[i_d + 1:]:
            p_val = permutation_test(domain_entropy[d1], domain_entropy[d2])
            d_val = cohens_d(domain_entropy[d1], domain_entropy[d2])
            key = f"{d1}_vs_{d2}"
            pairwise_results[key] = {
                "p_value": p_val,
                "cohens_d": d_val,
                "significant_005": p_val < 0.05,
                "significant_001": p_val < 0.01,
            }
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            logger.info(
                f"    {d1:10s} vs {d2:10s}: "
                f"p={p_val:.4f} {sig_marker:3s} | d={d_val:+.3f}"
            )

    # Overall discrimination
    all_means = [v["mean_entropy"] for v in domain_summary.values()]
    within_stds = [v["std_entropy"] for v in domain_summary.values()]
    
    cross_domain_std = float(np.std(all_means)) if len(all_means) >= 2 else 0.0
    within_domain_std = float(np.mean(within_stds)) if within_stds else 0.0
    discrimination_ratio = (
        cross_domain_std / within_domain_std if within_domain_std > 0 else float("inf")
    )

    # Count significant pairwise comparisons
    n_pairs = len(pairwise_results)
    n_significant = sum(1 for v in pairwise_results.values() if v["significant_005"])
    
    logger.info(f"\n  Cross-domain entropy std:  {cross_domain_std:.4f}")
    logger.info(f"  Within-domain entropy std: {within_domain_std:.4f}")
    logger.info(f"  Discrimination ratio:      {discrimination_ratio:.4f}")
    logger.info(f"  Significant pairs (p<0.05): {n_significant}/{n_pairs}")

    # Verdict
    if discrimination_ratio > 1.0 and n_significant >= n_pairs * 0.5:
        gc_lori_viable = True
        logger.info("\n  ✅ STRONG POSITIVE SIGNAL: Internal routing significantly differs across domains!")
        logger.info("     → GC-LoRI is VIABLE with statistical support.")
    elif discrimination_ratio > 0.5 or n_significant >= 2:
        gc_lori_viable = "weak_positive"
        logger.info("\n  ⚠️  WEAK POSITIVE SIGNAL: Some domain discrimination, but not overwhelming.")
        logger.info("     → GC-LoRI may help, but gains could be marginal.")
    else:
        gc_lori_viable = False
        logger.info("\n  ❌ NO SIGNAL: Routing patterns are statistically indistinguishable across domains.")
        logger.info("     → GC-LoRI is unlikely to add value over blind routing.")

    # ── Save everything ──
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": MODEL_PATH,
            "total_prompts": len(all_prompts),
            "processed": len(results),
            "failed": failed,
            "vram_gb": vram_used,
            "load_time_s": load_time,
            "random_seed": RANDOM_SEED,
            "num_bootstrap": NUM_BOOTSTRAP_ITERATIONS,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "data_sources": {
            "per_domain": dict(domain_counts),
            "per_source": dict(source_counts),
        },
        "domain_summary": domain_summary,
        "pairwise_comparisons": pairwise_results,
        "aggregate_stats": {
            "cross_domain_std": cross_domain_std,
            "within_domain_std": within_domain_std,
            "discrimination_ratio": discrimination_ratio,
            "n_significant_pairs": n_significant,
            "n_total_pairs": n_pairs,
        },
        "gc_lori_viable": gc_lori_viable,
        # Don't save 2000 per-prompt results in the summary — save separately
    }

    output_path = OUTPUT_DIR / "router_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save per-prompt details separately (large file)
    details_path = OUTPUT_DIR / "router_analysis_details.json"
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n  Summary saved to: {output_path}")
    logger.info(f"  Details saved to: {details_path}")

    # Cleanup
    hooker.remove()
    del model
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"  Total prompts analyzed: {len(results)}")
    logger.info(f"  Domains: {list(domain_summary.keys())}")
    logger.info(f"  Discrimination ratio: {discrimination_ratio:.4f}")
    logger.info(f"  Significant pairs: {n_significant}/{n_pairs}")
    if gc_lori_viable is True:
        logger.info("  VERDICT: ✅ GC-LoRI IS VIABLE (statistically supported)")
    elif gc_lori_viable == "weak_positive":
        logger.info("  VERDICT: ⚠️  WEAK signal — proceed with caution")
    else:
        logger.info("  VERDICT: ❌ GC-LoRI is NOT supported by this analysis")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
