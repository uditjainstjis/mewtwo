#!/usr/bin/env python3
"""
LoRI-MoE Full Evaluation Pipeline
==================================
Phases 1-4 of the empirical validation:
  1. Base model baselines (GSM8K, MMLU, ARC-Challenge)
  2. Single-adapter evaluation per domain
  3. Full LoRI-MoE composite evaluation
  4. Interference test (single vs all-active degradation)

Usage:
    python scripts/full_evaluation_pipeline.py --phase 1
    python scripts/full_evaluation_pipeline.py --phase all
"""

import os
import sys
import gc
import re
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ─── Setup ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
RESULTS_DIR = PROJECT_ROOT / "results" / "lori_moe"
ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "lori_moe" / "qwen2.5_1.5b"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DOMAINS = ["math", "code", "science", "legal", "medical"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(PROJECT_ROOT / "logs" / "lori_moe" / "evaluation.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─── Data Types ────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    benchmark: str
    config: str
    metric_name: str
    score: float
    num_examples: int
    details: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


# ─── Math Evaluation (GSM8K) ──────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text."""
    # Try \\boxed{answer}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Try #### answer (GSM8K format)
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hashes:
        # Clean: remove commas, dollar signs, etc.
        ans = hashes[-1].strip().replace(",", "").replace("$", "").strip()
        return ans
    # Try "The answer is X"
    answer_match = re.findall(r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match[-1].strip()
    return None


def normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison."""
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    # Try to convert to float then back to clean string
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s.lower().strip()


def evaluate_gsm8k(model, tokenizer, max_samples: int = 200, batch_size: int = 16) -> BenchmarkResult:
    """Evaluate on GSM8K with exact-match accuracy using batching."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    logger.info(f"Running GSM8K evaluation ({max_samples} samples) with batch_size={batch_size}")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    model.eval()
    
    # Enable tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding for batch generation
        
    for i in tqdm(range(0, len(ds), batch_size), desc="GSM8K Batch"):
        batch = ds[i:i + batch_size]
        questions = batch["question"]
        golds_raw = batch["answer"]
        
        prompts = [
            f"Solve the following math problem step by step. Put your final answer after ####.\n\nQuestion: {q}\n\nAnswer:" 
            for q in questions
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=768, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        responses = tokenizer.batch_decode(
            [out[inp.shape[0]:] for out, inp in zip(outputs, inputs["input_ids"])],
            skip_special_tokens=True
        )
        
        for j in range(len(responses)):
            gold_raw = golds_raw[j]
            gold = normalize_number(extract_answer(gold_raw) or "")
            pred = normalize_number(extract_answer(responses[j]) or "")
            
            if pred and gold and pred == gold:
                correct += 1
            total += 1

            if total <= 5:
                logger.info(f"  GSM8K [{total-1}]: gold={gold}, pred={pred}, match={pred==gold}")

    accuracy = correct / max(total, 1)
    logger.info(f"GSM8K: {correct}/{total} = {accuracy:.4f}")
    return BenchmarkResult("gsm8k", "", "exact_match", accuracy, total)


# ─── MMLU Evaluation ──────────────────────────────────────────────────────────

def evaluate_mmlu(model, tokenizer, max_samples: int = 200, subjects: List[str] = None, batch_size: int = 32) -> BenchmarkResult:
    """Evaluate MMLU with multiple-choice accuracy."""
    from datasets import load_dataset

    logger.info(f"Running MMLU evaluation ({max_samples} samples) with batch_size={batch_size}")

    # Load MMLU
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        try:
            ds = load_dataset("lukaemon/mmlu", "all", split="test")
        except Exception:
            logger.warning("Could not load MMLU dataset, skipping")
            return BenchmarkResult("mmlu", "", "accuracy", 0.0, 0)

    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    choices = ["A", "B", "C", "D"]

    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(ds), batch_size), desc="MMLU Batch"):
        batch = ds[i:i + batch_size]
        questions = batch["question"]
        answers = batch["answer"]
        
        prompts = []
        gold_letters = []
        
        for j in range(len(questions)):
            question = questions[j]
            # Handle different dataset dict structures
            options = [batch[f"choices"][j][k] if "choices" in batch else batch.get(c, [""]*len(questions))[j] for k, c in enumerate(choices)]
            options_text = "\n".join(f"{c}. {o}" for c, o in zip(choices, options))
            
            prompt = f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\nQuestion: {question}\n{options_text}\n\nAnswer:"
            prompts.append(prompt)
            
            gold_idx = answers[j]
            if isinstance(gold_idx, int):
                gold_letter = choices[gold_idx] if gold_idx < 4 else ""
            else:
                gold_letter = str(gold_idx).upper()
            gold_letters.append(gold_letter)

        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        responses = tokenizer.batch_decode(
            [out[inp.shape[0]:] for out, inp in zip(outputs, inputs["input_ids"])], 
            skip_special_tokens=True
        )

        for j in range(len(responses)):
            response = responses[j].strip()
            pred_letter = response[0].upper() if response else ""
            gold_letter = gold_letters[j]

            if pred_letter == gold_letter:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    logger.info(f"MMLU: {correct}/{total} = {accuracy:.4f}")
    return BenchmarkResult("mmlu", "", "accuracy", accuracy, total)


# ─── ARC-Challenge Evaluation ─────────────────────────────────────────────────

def evaluate_arc(model, tokenizer, max_samples: int = 200, batch_size: int = 32) -> BenchmarkResult:
    """Evaluate on ARC-Challenge with multiple-choice accuracy."""
    from datasets import load_dataset

    logger.info(f"Running ARC-Challenge evaluation ({max_samples} samples) with batch_size={batch_size}")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(ds), batch_size), desc="ARC Batch"):
        batch = ds[i:i + batch_size]
        questions = batch["question"]
        choices_lists = batch["choices"]
        answer_keys = batch["answerKey"]
        
        prompts = []
        
        for j in range(len(questions)):
            labels = choices_lists[j]["label"]
            texts = choices_lists[j]["text"]
            options_text = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            prompt = f"Answer the following science question. Reply with just the letter.\n\nQuestion: {questions[j]}\n{options_text}\n\nAnswer:"
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        responses = tokenizer.batch_decode(
            [out[inp.shape[0]:] for out, inp in zip(outputs, inputs["input_ids"])],
            skip_special_tokens=True
        )

        for j in range(len(responses)):
            response = responses[j].strip()
            pred_letter = response[0].upper() if response else ""
            if pred_letter == answer_keys[j].upper():
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    logger.info(f"ARC-Challenge: {correct}/{total} = {accuracy:.4f}")
    return BenchmarkResult("arc_challenge", "", "accuracy", accuracy, total)


# ─── Perplexity Evaluation ────────────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, texts: List[str], label: str = "") -> float:
    """Compute average perplexity over a list of texts."""
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1
    avg_ppl = torch.exp(torch.tensor(total_loss / max(count, 1))).item()
    return avg_ppl


# ─── Model Loading Utilities ──────────────────────────────────────────────────

def load_base_model():
    """Load Qwen2.5-1.5B-Instruct base model."""
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_adapter_model(model, domain: str, variant: str = "dare_sparsified"):
    """Load a PEFT adapter for a specific domain."""
    adapter_path = ADAPTER_DIR / domain / variant
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        # Fallback
        for v in ["dare_sparsified", "best", "final"]:
            alt = ADAPTER_DIR / domain / v
            if alt.exists() and (alt / "adapter_config.json").exists():
                adapter_path = alt
                break
        else:
            logger.warning(f"No adapter found for domain '{domain}'")
            return None

    logger.info(f"Loading {domain} adapter from {adapter_path}")
    peft_model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    peft_model.eval()
    return peft_model


def cleanup_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Phase 1: Baselines ───────────────────────────────────────────────────────

def run_phase1(max_samples: int = 200) -> Dict:
    """Run base model baselines."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: BASE MODEL BASELINES")
    logger.info("=" * 70)

    model, tokenizer = load_base_model()
    results = {}

    # GSM8K
    gsm8k = evaluate_gsm8k(model, tokenizer, max_samples)
    gsm8k.config = "base_zero_shot"
    results["base_gsm8k"] = asdict(gsm8k)
    logger.info(f"✓ Base GSM8K: {gsm8k.score:.4f}")

    # ARC-Challenge
    arc = evaluate_arc(model, tokenizer, max_samples)
    arc.config = "base_zero_shot"
    results["base_arc"] = asdict(arc)
    logger.info(f"✓ Base ARC-Challenge: {arc.score:.4f}")

    # MMLU
    mmlu = evaluate_mmlu(model, tokenizer, max_samples)
    mmlu.config = "base_zero_shot"
    results["base_mmlu"] = asdict(mmlu)
    logger.info(f"✓ Base MMLU: {mmlu.score:.4f}")

    cleanup_model(model)

    # Save
    output = RESULTS_DIR / "phase1_baselines.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 1 results saved to {output}")

    print_results_table("Phase 1: Baselines", results)
    return results


# ─── Phase 2: Single-Adapter Evaluation ───────────────────────────────────────

def run_phase2(max_samples: int = 200) -> Dict:
    """Evaluate each domain adapter on its matched benchmark."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: SINGLE-ADAPTER EVALUATION")
    logger.info("=" * 70)

    results = {}

    # Domain → benchmark mapping
    domain_benchmarks = {
        "math": ("gsm8k", evaluate_gsm8k),
        "science": ("arc_challenge", evaluate_arc),
        "code": ("gsm8k", evaluate_gsm8k),  # Using GSM8K as proxy (HumanEval needs exec)
        "legal": ("arc_challenge", evaluate_arc),  # ARC as cross-domain proxy
        "medical": ("arc_challenge", evaluate_arc),  # ARC as cross-domain proxy
    }

    for domain in DOMAINS:
        bench_name, eval_fn = domain_benchmarks[domain]
        logger.info(f"\n--- Evaluating {domain} adapter on {bench_name} ---")

        model, tokenizer = load_base_model()
        adapter_model = load_adapter_model(model, domain)
        if adapter_model is None:
            logger.warning(f"Skipping {domain} — no adapter")
            cleanup_model(model)
            continue

        result = eval_fn(adapter_model, tokenizer, max_samples)
        result.config = f"single_{domain}"
        results[f"single_{domain}_{bench_name}"] = asdict(result)
        logger.info(f"✓ {domain} adapter on {bench_name}: {result.score:.4f}")

        # Also evaluate on ALL benchmarks for interference comparison
        if bench_name != "gsm8k":
            gsm8k_result = evaluate_gsm8k(adapter_model, tokenizer, min(max_samples, 100))
            gsm8k_result.config = f"single_{domain}"
            results[f"single_{domain}_gsm8k"] = asdict(gsm8k_result)

        cleanup_model(adapter_model)
        cleanup_model(model)

    output = RESULTS_DIR / "phase2_single_adapter.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 2 results saved to {output}")

    print_results_table("Phase 2: Single Adapter", results)
    return results


# ─── Phase 3: Full LoRI-MoE Composite Evaluation ──────────────────────────────

def run_phase3(max_samples: int = 200) -> Dict:
    """Evaluate the fully composed LoRIMoEModel with active token routing."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: FULL LORI-MOE COMPOSITE EVALUATION")
    logger.info("=" * 70)

    from src.lori_moe.config import LoRIMoEConfig
    from src.lori_moe.model.lori_moe_model import LoRIMoEModel

    config = LoRIMoEConfig()
    config.model.base_model = BASE_MODEL
    
    logger.info("Initializing full LoRIMoEModel runtime...")
    # Build model and inject layers
    lori_model = LoRIMoEModel(config).build(
        load_experts=True,
        adapters_root=ADAPTER_DIR
    )

    # Load trained router weights
    router_path = ADAPTER_DIR / "router" / "best" / "router.pt"
    if router_path.exists():
        logger.info(f"Loading trained router from {router_path}")
        checkpoint = torch.load(router_path, map_location=lori_model.device, weights_only=True)
        lori_model.routers.load_state_dict(checkpoint["router_state_dict"])
        # Turn off noise for evaluation
        lori_model.routers.set_noise_std(0.0)
    else:
        logger.warning(f"No trained router found at {router_path}. Using random routing!")

    lori_model.eval()

    results = {}

    # Run ablations to use pure Top-1 routing to avoid single-adapter scale dilution.
    # We also use Prompt-Level Routing to stabilize the residual stream preventing token jitter.
    with lori_model.inference_ablation(top_k=1, prompt_routing=True):
        # Evaluate GSM8K (Testing if Math capability is retained)
        gsm8k_result = evaluate_gsm8k(lori_model, lori_model.tokenizer, max_samples, batch_size=16)
        gsm8k_result.config = "lori_moe_routed_prompt"
        results["composite_gsm8k"] = asdict(gsm8k_result)

        # Evaluate ARC-Challenge (Testing Science/Medical/Legal general capability)
        arc_result = evaluate_arc(lori_model, lori_model.tokenizer, max_samples, batch_size=32)
        arc_result.config = "lori_moe_routed_top1"
        results["composite_arc"] = asdict(arc_result)
        
        # Evaluate MMLU (General multi-domain capability)
        mmlu_result = evaluate_mmlu(lori_model, lori_model.tokenizer, max_samples, batch_size=32)
        mmlu_result.config = "lori_moe_routed_top1"
        results["composite_mmlu"] = asdict(mmlu_result)

    cleanup_model(lori_model)

    output = RESULTS_DIR / "phase3_composite.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 3 results saved to {output}")

    print_results_table("Phase 3: Routed Composite", results)
    return results


# ─── Phase 4: Interference Test ───────────────────────────────────────────────

def run_phase4(max_samples: int = 100) -> Dict:
    """
    Test interference: does composing all adapters degrade single-domain performance?

    Method:
    1. Load all 5 adapters into PEFT model
    2. For each domain, test with ONLY that adapter active
    3. Then test with linear merge of all adapters
    4. Compute degradation
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: INTERFERENCE TEST")
    logger.info("=" * 70)

    from datasets import load_dataset
    import json as json_mod

    model, tokenizer = load_base_model()

    # Load domain-specific test texts
    data_dir = PROJECT_ROOT / "data" / "lori_moe"
    domain_texts = {}
    for domain in DOMAINS:
        fpath = data_dir / f"{domain}_train.jsonl"
        if fpath.exists():
            texts = []
            with open(fpath) as f:
                for j, line in enumerate(f):
                    if j >= 20:  # Use 20 samples per domain
                        break
                    row = json_mod.loads(line.strip())
                    texts.append(row["text"])
            domain_texts[domain] = texts
            logger.info(f"Loaded {len(texts)} test texts for {domain}")

    # Load ALL adapters
    peft_model = None
    for i, domain in enumerate(DOMAINS):
        adapter_path = ADAPTER_DIR / domain / "dare_sparsified"
        if not adapter_path.exists():
            adapter_path = ADAPTER_DIR / domain / "best"
        if not adapter_path.exists():
            continue

        logger.info(f"Loading adapter: {domain} from {adapter_path}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=domain)
        else:
            peft_model.load_adapter(str(adapter_path), adapter_name=domain)

    peft_model.eval()

    results = {"single_adapter_ppl": {}, "merged_ppl": {}, "degradation": {}}

    # Phase A: Single adapter PPL per domain
    logger.info("\n--- PHASE A: Single-Adapter Perplexity ---")
    for domain in DOMAINS:
        if domain not in domain_texts:
            continue
        peft_model.set_adapter(domain)
        ppl = evaluate_perplexity(peft_model, tokenizer, domain_texts[domain], f"single_{domain}")
        results["single_adapter_ppl"][domain] = ppl
        logger.info(f"  [{domain.upper()}] Single adapter PPL: {ppl:.2f}")

    # Phase B: Linear merge
    logger.info("\n--- PHASE B: Linear Merge (All 5 Adapters) ---")
    try:
        peft_model.add_weighted_adapter(
            adapters=DOMAINS,
            weights=[1.0] * len(DOMAINS),
            adapter_name="omni_merged",
            combination_type="linear",
        )
        peft_model.set_adapter("omni_merged")
        logger.info("  ✓ Linear merge complete")

        for domain in DOMAINS:
            if domain not in domain_texts:
                continue
            ppl = evaluate_perplexity(peft_model, tokenizer, domain_texts[domain], f"merged_{domain}")
            results["merged_ppl"][domain] = ppl

            single_ppl = results["single_adapter_ppl"][domain]
            degradation = ((ppl / single_ppl) - 1.0) * 100
            results["degradation"][domain] = degradation

            flag = "✅ PASS" if abs(degradation) < 5.0 else "❌ FAIL"
            logger.info(f"  [{domain.upper()}] Merged PPL: {ppl:.2f} | Single PPL: {single_ppl:.2f} | Degradation: {degradation:+.2f}% {flag}")

    except Exception as e:
        logger.error(f"Linear merge failed: {e}")
        results["merge_error"] = str(e)

    # Phase C: Equal-weight softmax composition (simulating router)
    logger.info("\n--- PHASE C: Equal-Weight Composition (1/5 each) ---")
    try:
        peft_model.add_weighted_adapter(
            adapters=DOMAINS,
            weights=[0.2] * len(DOMAINS),
            adapter_name="equal_mix",
            combination_type="linear",
        )
        peft_model.set_adapter("equal_mix")

        results["equal_mix_ppl"] = {}
        for domain in DOMAINS:
            if domain not in domain_texts:
                continue
            ppl = evaluate_perplexity(peft_model, tokenizer, domain_texts[domain], f"equal_{domain}")
            results["equal_mix_ppl"][domain] = ppl
            logger.info(f"  [{domain.upper()}] Equal-mix PPL: {ppl:.2f}")

    except Exception as e:
        logger.error(f"Equal-weight composition failed: {e}")

    cleanup_model(peft_model)
    cleanup_model(model)

    output = RESULTS_DIR / "phase4_interference.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 4 results saved to {output}")

    # Print summary
    print("\n" + "=" * 75)
    print("INTERFERENCE TEST SUMMARY")
    print("=" * 75)
    print(f"{'DOMAIN':<10} | {'SINGLE PPL':<12} | {'MERGED PPL':<12} | {'DEGRADATION':<15} | STATUS")
    print("-" * 75)
    for domain in DOMAINS:
        if domain in results["single_adapter_ppl"] and domain in results.get("merged_ppl", {}):
            s = results["single_adapter_ppl"][domain]
            m = results["merged_ppl"][domain]
            d = results["degradation"][domain]
            flag = "✅ PASS" if abs(d) < 5.0 else "❌ FAIL"
            print(f"{domain.upper():<10} | {s:<12.2f} | {m:<12.2f} | {d:+12.2f}%  | {flag}")
    print("=" * 75)

    return results


# ─── Utility ───────────────────────────────────────────────────────────────────

def print_results_table(title: str, results: Dict):
    """Print a formatted results table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for name, result in results.items():
        score = result.get("score", 0)
        metric = result.get("metric_name", "?")
        n = result.get("num_examples", 0)
        print(f"  {name:<40s}: {score:.4f} ({metric}, n={n})")
    print(f"{'=' * 60}\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRI-MoE Full Evaluation Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        help="Which phase to run: 1, 2, 4, or 'all'")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples per benchmark")
    args = parser.parse_args()

    (PROJECT_ROOT / "logs" / "lori_moe").mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.phase in ("1", "all"):
        all_results["phase1"] = run_phase1(args.max_samples)

    if args.phase in ("2", "all"):
        all_results["phase2"] = run_phase2(args.max_samples)

    if args.phase in ("3", "all"):
        all_results["phase3"] = run_phase3(args.max_samples)

    if args.phase in ("4", "all"):
        all_results["phase4"] = run_phase4(min(args.max_samples, 100))

    # Save combined
    combined_output = RESULTS_DIR / "all_results.json"
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {combined_output}")


if __name__ == "__main__":
    main()
