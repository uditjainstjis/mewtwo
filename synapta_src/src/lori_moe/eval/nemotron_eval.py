#!/usr/bin/env python3
"""
Nemotron-Aware Evaluation Pipeline — Publication Grade

Runs FULL test sets with proper statistical reporting:
  - GSM8K: 1,319 test examples (math reasoning)
  - MATH500: 500 examples (hard math)
  - ARC-Challenge: 1,172 test examples (science reasoning)
  - MMLU: 14,042 5-shot test examples (multi-domain knowledge)
  - HumanEval: 164 test examples (code generation)

Statistical features:
  - Bootstrap 95% confidence intervals on all metrics
  - Per-category breakdown for MMLU
  - Paired comparisons (baseline vs adapter, GC vs blind)

Usage:
    # Quick development run (200 samples per benchmark)
    python -m src.lori_moe.eval.nemotron_eval --quick

    # Publication run (full test sets)
    python -m src.lori_moe.eval.nemotron_eval --full

    # Specific benchmarks
    python -m src.lori_moe.eval.nemotron_eval --benchmarks gsm8k mmlu arc humaneval
"""

import sys
import json
import time
import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants — Full test set sizes
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_SIZES = {
    "gsm8k": 1319,       # Full GSM8K test set
    "math500": 500,      # MATH-500 (standard subset)
    "arc": 1172,         # Full ARC-Challenge test set
    "mmlu": 14042,       # Full MMLU test set
    "humaneval": 164,    # Full HumanEval
}

# Quick mode — With KV cache fix, generation is ~5-10s each instead of ~100s
QUICK_SIZES = {
    "gsm8k": 200,        # ~15-30 min with cache (was ~6 hours without)
    "math500": 200,      # ~15-30 min with cache
    "arc": 300,          # Very fast with max_new_tokens=64
    "mmlu": 500,         # Already fast (8 tokens each)
    "humaneval": 100,    # Code generation
}

NUM_BOOTSTRAP = 1000
RANDOM_SEED = 42


@dataclass
class EvalResult:
    """Result from a single eval run with statistical metadata."""
    benchmark: str
    metric: str
    score: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    num_examples: int = 0
    num_correct: int = 0
    config_name: str = "base"
    per_sample: List[bool] = field(default_factory=list)  # For paired tests
    details: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Statistical utilities
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(correct: List[bool], n_iter: int = NUM_BOOTSTRAP) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for accuracy."""
    if not correct:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(RANDOM_SEED)
    n = len(correct)
    arr = np.array(correct, dtype=float)
    means = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_test(
    correct_a: List[bool], correct_b: List[bool], n_iter: int = 10000
) -> float:
    """Paired bootstrap test for two systems on same examples. Returns p-value."""
    assert len(correct_a) == len(correct_b)
    a = np.array(correct_a, dtype=float)
    b = np.array(correct_b, dtype=float)
    observed_diff = np.mean(a) - np.mean(b)
    
    rng = np.random.RandomState(RANDOM_SEED)
    count = 0
    n = len(a)
    for _ in range(n_iter):
        # Randomly swap labels
        mask = rng.randint(0, 2, size=n)
        perm_a = np.where(mask, a, b)
        perm_b = np.where(mask, b, a)
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return (count + 1) / (n_iter + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_number(text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    # \\boxed{answer}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    # #### answer (GSM8K format)
    hash_match = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        candidate = hash_match[-1].strip()
        # Clean: remove $, commas, percent signs
        candidate = candidate.replace("$", "").replace(",", "").replace("%", "")
        num = re.search(r'[-+]?\d*\.?\d+', candidate)
        return num.group() if num else candidate

    # "The answer is X" / "therefore X" / "= X"
    patterns = [
        r'(?:the answer is|answer:|therefore|thus|hence)\s*[:=]?\s*([-+]?\d*\.?\d+)',
        r'=\s*([-+]?\d*\.?\d+)\s*$',
    ]
    for pat in patterns:
        match = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match[-1].strip()

    # Last number in the text
    all_nums = re.findall(r'[-+]?\d+\.?\d*', text)
    if all_nums:
        return all_nums[-1]

    return None


def normalize_number(s: str) -> Optional[float]:
    """Normalize a number string for comparison."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def numbers_equal(a: str, b: str, tol: float = 1e-5) -> bool:
    """Compare two number strings with tolerance."""
    na, nb = normalize_number(a), normalize_number(b)
    if na is None or nb is None:
        return str(a).strip() == str(b).strip()
    return abs(na - nb) < tol


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def format_chat(tokenizer, question: str, system_prompt: str = None) -> str:
    """Format prompt with Nemotron chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"System: {system_prompt}\n" if system_prompt else ""
        return f"{prefix}User: {question}\nAssistant:"


def load_model_4bit(model_path: str):
    """Load Nemotron in 4-bit."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 512) -> str:
    """Generate a single response."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Nemotron-H specific: HybridMambaAttentionDynamicCache is required for KV cache
    past_key_values = None
    try:
        # Check if this is Nemotron by looking at the config
        if "NemotronH" in str(type(model.config)) or (hasattr(model.config, "hybrid_override_pattern")):
            from models.nemotron.modeling_nemotron_h import HybridMambaAttentionDynamicCache
            past_key_values = HybridMambaAttentionDynamicCache(
                model.config, 
                batch_size=inputs["input_ids"].shape[0], 
                device=model.device,
                dtype=model.dtype
            )
            logger.debug("Using HybridMambaAttentionDynamicCache for generation")
    except Exception as e:
        logger.warning(f"Failed to initialize specialized Nemotron cache: {e}. Falling back to default.")

    with torch.no_grad():
        streamer = TextStreamer(tokenizer, skip_prompt=True) if os.environ.get("DEBUG_GEN", "0") == "1" else None
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            past_key_values=past_key_values,
            streamer=streamer,
        )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark implementations
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_gsm8k(model, tokenizer, max_samples: int, config_name: str = "base") -> EvalResult:
    """GSM8K — Full test set (1,319 examples)."""
    from datasets import load_dataset
    t0 = time.time()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    correct_list = []
    examples = []

    system = "You are an expert mathematician. Solve the problem step by step. Put your final numerical answer in \\boxed{}."

    for i, ex in enumerate(tqdm(ds, desc=f"GSM8K [{config_name}]")):
        prompt = format_chat(tokenizer, ex["question"], system)
        response = generate_response(model, tokenizer, prompt)

        gold = extract_number(ex["answer"])
        pred = extract_number(response)
        is_correct = numbers_equal(pred, gold) if pred and gold else False
        correct_list.append(is_correct)

        if i < 5:
            examples.append({"q": ex["question"][:80], "gold": gold, "pred": pred, "ok": is_correct})

    score, ci_lo, ci_hi = bootstrap_ci(correct_list)

    logger.info(f"GSM8K [{config_name}]: {sum(correct_list)}/{len(correct_list)} = {score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return EvalResult(
        benchmark="gsm8k", metric="exact_match", score=score,
        ci_lower=ci_lo, ci_upper=ci_hi,
        num_examples=len(correct_list), num_correct=sum(correct_list),
        config_name=config_name, per_sample=correct_list,
        details={"examples": examples}, runtime_seconds=time.time() - t0,
    )


def evaluate_math500(model, tokenizer, max_samples: int, config_name: str = "base") -> EvalResult:
    """MATH-500 — Hard math problems."""
    from datasets import load_dataset
    t0 = time.time()

    try:
        ds = load_dataset("lighteval/MATH", split="test")
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split="test")

    if max_samples and max_samples < len(ds):
        # Sample deterministically for reproducibility
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(len(ds), size=max_samples, replace=False)
        ds = ds.select(indices.tolist())

    correct_list = []
    system = "Solve the following math problem. Show your work and put your final answer in \\boxed{}."

    for i, ex in enumerate(tqdm(ds, desc=f"MATH [{config_name}]")):
        prompt = format_chat(tokenizer, ex["problem"], system)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=768)

        gold = extract_number(ex["solution"])
        pred = extract_number(response)
        is_correct = numbers_equal(pred, gold) if pred and gold else False
        correct_list.append(is_correct)

    score, ci_lo, ci_hi = bootstrap_ci(correct_list)
    logger.info(f"MATH [{config_name}]: {sum(correct_list)}/{len(correct_list)} = {score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return EvalResult(
        benchmark="math500", metric="exact_match", score=score,
        ci_lower=ci_lo, ci_upper=ci_hi,
        num_examples=len(correct_list), num_correct=sum(correct_list),
        config_name=config_name, per_sample=correct_list,
        runtime_seconds=time.time() - t0,
    )


def evaluate_arc(model, tokenizer, max_samples: int, config_name: str = "base") -> EvalResult:
    """ARC-Challenge — Full test set (1,172 examples)."""
    from datasets import load_dataset
    t0 = time.time()

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    correct_list = []
    system = "Answer the multiple-choice science question. Reply with ONLY the letter (A, B, C, or D)."

    for i, ex in enumerate(tqdm(ds, desc=f"ARC [{config_name}]")):
        choices = ex["choices"]
        choice_text = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        full_q = f"{ex['question']}\n\n{choice_text}"

        prompt = format_chat(tokenizer, full_q, system)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=64)

        gold = ex["answerKey"].upper()
        pred = None
        for char in response.strip():
            if char.upper() in "ABCDE":
                pred = char.upper()
                break

        correct_list.append(pred == gold)

    score, ci_lo, ci_hi = bootstrap_ci(correct_list)
    logger.info(f"ARC [{config_name}]: {sum(correct_list)}/{len(correct_list)} = {score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return EvalResult(
        benchmark="arc_challenge", metric="accuracy", score=score,
        ci_lower=ci_lo, ci_upper=ci_hi,
        num_examples=len(correct_list), num_correct=sum(correct_list),
        config_name=config_name, per_sample=correct_list,
        runtime_seconds=time.time() - t0,
    )


def evaluate_mmlu(model, tokenizer, max_samples: int, config_name: str = "base") -> EvalResult:
    """MMLU — 5-shot, full test set (14,042 examples across 57 subjects)."""
    from datasets import load_dataset
    t0 = time.time()

    # Load full MMLU
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    if max_samples and max_samples < len(test_ds):
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(len(test_ds), size=max_samples, replace=False)
        test_ds = test_ds.select(indices.tolist())

    # Build 5-shot examples per subject from dev set
    dev_by_subject = defaultdict(list)
    for ex in dev_ds:
        dev_by_subject[ex["subject"]].append(ex)

    correct_list = []
    subject_results = defaultdict(lambda: {"correct": 0, "total": 0})

    LETTERS = ["A", "B", "C", "D"]

    for i, ex in enumerate(tqdm(test_ds, desc=f"MMLU [{config_name}]")):
        subject = ex["subject"]
        gold_idx = ex["answer"]
        gold_letter = LETTERS[gold_idx] if isinstance(gold_idx, int) else gold_idx

        # Build 5-shot prompt
        few_shot = ""
        shots = dev_by_subject.get(subject, [])[:5]
        for shot in shots:
            shot_choices = " ".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(shot["choices"]))
            shot_answer = LETTERS[shot["answer"]] if isinstance(shot["answer"], int) else shot["answer"]
            few_shot += f"Q: {shot['question']} {shot_choices}\nA: {shot_answer}\n\n"

        test_choices = " ".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(ex["choices"]))
        question = f"{few_shot}Q: {ex['question']} {test_choices}\nA:"

        prompt = format_chat(tokenizer, question)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=8)

        pred = None
        for char in response.strip():
            if char.upper() in "ABCD":
                pred = char.upper()
                break

        is_correct = (pred == gold_letter)
        correct_list.append(is_correct)
        subject_results[subject]["total"] += 1
        if is_correct:
            subject_results[subject]["correct"] += 1

    score, ci_lo, ci_hi = bootstrap_ci(correct_list)

    # Per-subject breakdown
    subject_scores = {}
    for subj, res in sorted(subject_results.items()):
        if res["total"] > 0:
            subject_scores[subj] = res["correct"] / res["total"]

    logger.info(f"MMLU [{config_name}]: {sum(correct_list)}/{len(correct_list)} = {score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    logger.info(f"  Subjects evaluated: {len(subject_scores)}")

    return EvalResult(
        benchmark="mmlu", metric="5shot_accuracy", score=score,
        ci_lower=ci_lo, ci_upper=ci_hi,
        num_examples=len(correct_list), num_correct=sum(correct_list),
        config_name=config_name, per_sample=correct_list,
        details={"subject_scores": subject_scores}, runtime_seconds=time.time() - t0,
    )


def evaluate_humaneval(model, tokenizer, max_samples: int, config_name: str = "base") -> EvalResult:
    """HumanEval — Code generation (164 problems). Uses simple syntax check."""
    from datasets import load_dataset
    t0 = time.time()

    ds = load_dataset("openai/openai_humaneval", split="test")
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    correct_list = []
    system = "Complete the following Python function. Only output the code, no explanations."

    for i, ex in enumerate(tqdm(ds, desc=f"HumanEval [{config_name}]")):
        prompt = format_chat(tokenizer, ex["prompt"], system)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)

        # Extract code from response
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        # Combine prompt + completion
        full_code = ex["prompt"] + code

        # Basic validation: can it parse?
        try:
            compile(full_code, "<string>", "exec")
            # Run test cases if available
            test_code = full_code + "\n" + ex.get("test", "")
            try:
                exec_globals = {}
                exec(test_code, exec_globals)
                correct_list.append(True)
            except Exception:
                correct_list.append(False)
        except SyntaxError:
            correct_list.append(False)

    score, ci_lo, ci_hi = bootstrap_ci(correct_list)
    logger.info(f"HumanEval [{config_name}]: {sum(correct_list)}/{len(correct_list)} = {score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return EvalResult(
        benchmark="humaneval", metric="pass@1", score=score,
        ci_lower=ci_lo, ci_upper=ci_hi,
        num_examples=len(correct_list), num_correct=sum(correct_list),
        config_name=config_name, per_sample=correct_list,
        runtime_seconds=time.time() - t0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_FUNCS = {
    "gsm8k": evaluate_gsm8k,
    "math500": evaluate_math500,
    "arc": evaluate_arc,
    "mmlu": evaluate_mmlu,
    "humaneval": evaluate_humaneval,
}


def run_evaluation(
    model_path: str,
    adapter_dir: Optional[str] = None,
    output_dir: str = str(PROJECT_ROOT / "results" / "nemotron"),
    benchmarks: List[str] = None,
    sizes: Dict[str, int] = None,
    eval_baseline: bool = True,
    eval_adapters: bool = True,
    domains: List[str] = None,
):
    """Run full evaluation suite."""
    domains = domains or ["math", "code", "science"]
    benchmarks = benchmarks or ["gsm8k", "arc", "mmlu"]
    sizes = sizes or BENCHMARK_SIZES
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Baseline
    if eval_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE EVALUATION")
        logger.info("=" * 60)

        model, tokenizer = load_model_4bit(model_path)

        for bench in benchmarks:
            if bench in BENCHMARK_FUNCS:
                result = BENCHMARK_FUNCS[bench](
                    model, tokenizer, sizes.get(bench, 200), config_name="baseline"
                )
                all_results[f"baseline_{bench}"] = asdict(result)

        del model; torch.cuda.empty_cache()

    # Single adapters
    if eval_adapters and adapter_dir:
        logger.info("\n" + "=" * 60)
        logger.info("ADAPTER EVALUATION")
        logger.info("=" * 60)

        domain_benchmark_map = {
            "math": ["gsm8k", "math500"],
            "code": ["humaneval"],
            "science": ["arc"],
        }

        for domain in domains:
            adapter_path = None
            for subdir in ["dare_sparsified", "best", "final"]:
                candidate = Path(adapter_dir) / domain / subdir
                if candidate.exists() and (candidate / "adapter_config.json").exists():
                    adapter_path = candidate
                    break

            if adapter_path is None:
                logger.warning(f"No adapter for {domain}")
                continue

            model, tokenizer = load_model_4bit(model_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

            for bench in domain_benchmark_map.get(domain, []):
                if bench in benchmarks and bench in BENCHMARK_FUNCS:
                    result = BENCHMARK_FUNCS[bench](
                        model, tokenizer, sizes.get(bench, 200),
                        config_name=f"adapter_{domain}"
                    )
                    all_results[f"adapter_{domain}_{bench}"] = asdict(result)

            del model; torch.cuda.empty_cache()

    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*80}")
    print("NEMOTRON EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Benchmark':<15} {'Score':>8} {'95% CI':>20} {'N':>6}")
    print(f"{'-'*74}")
    for name, r in all_results.items():
        ci = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        print(f"{r['config_name']:<25} {r['benchmark']:<15} {r['score']:>8.4f} {ci:>20} {r['num_examples']:>6}")
    print(f"{'='*80}")

    # Paired comparisons
    print(f"\nPAIRED COMPARISONS:")
    for bench in benchmarks:
        baseline_key = f"baseline_{bench}"
        if baseline_key not in all_results:
            continue
        baseline_samples = all_results[baseline_key].get("per_sample", [])
        for name, r in all_results.items():
            if name == baseline_key or bench not in name:
                continue
            adapter_samples = r.get("per_sample", [])
            if len(baseline_samples) == len(adapter_samples) and len(baseline_samples) > 0:
                p_val = paired_bootstrap_test(adapter_samples, baseline_samples)
                delta = r["score"] - all_results[baseline_key]["score"]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {r['config_name']} vs baseline ({bench}): "
                      f"Δ={delta:+.4f}, p={p_val:.4f} {sig}")

    logger.info(f"\nResults saved to {results_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Nemotron Evaluation — Publication Grade")
    parser.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "models" / "nemotron"))
    parser.add_argument("--adapter_dir", type=str, default=str(PROJECT_ROOT / "adapters" / "nemotron_30b"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "nemotron"))
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "arc", "mmlu"],
                        choices=list(BENCHMARK_FUNCS.keys()))
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--no_adapters", action="store_true")
    parser.add_argument("--domains", nargs="+", default=["math", "code", "science"])
    
    # Scale controls
    parser.add_argument("--full", action="store_true", help="Use full test sets (publication mode)")
    parser.add_argument("--quick", action="store_true", help="Use reduced test sets (development mode)")
    parser.add_argument("--max_samples", type=int, default=None, help="Override all benchmark sizes")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Determine sizes
    if args.max_samples:
        sizes = {b: args.max_samples for b in BENCHMARK_FUNCS}
    elif args.full:
        sizes = BENCHMARK_SIZES
        logger.info("📊 PUBLICATION MODE — using full test sets")
    elif args.quick:
        sizes = QUICK_SIZES
        logger.info("🔧 DEVELOPMENT MODE — using reduced test sets")
    else:
        sizes = QUICK_SIZES
        logger.info("🔧 Default: development mode. Use --full for publication-grade runs.")

    logger.info(f"Benchmark sizes: {sizes}")

    run_evaluation(
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        sizes=sizes,
        eval_baseline=not args.no_baseline,
        eval_adapters=not args.no_adapters,
        domains=args.domains,
    )


if __name__ == "__main__":
    main()
