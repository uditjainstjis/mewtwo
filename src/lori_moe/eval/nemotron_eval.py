#!/usr/bin/env python3
"""
Nemotron-Aware Evaluation Pipeline

Wraps the existing evaluation code with Nemotron chat template support
and adds GC-LoRI ablation evaluation.

This handles the key issue from the plan: the existing evaluation code
uses plain prompts, but Nemotron needs its specific chat template.

Usage:
    python -m src.lori_moe.eval.nemotron_eval \
        --model_path ./models/nemotron \
        --adapter_dir ./checkpoints/nemotron_lori/adapters \
        --output_dir ./results/nemotron \
        --max_samples 200
"""

import sys
import json
import time
import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from a single eval run."""
    benchmark: str
    metric: str
    score: float
    num_examples: int
    config_name: str
    details: dict = field(default_factory=dict)


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical or boxed answer from model response."""
    # \\boxed{answer}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()

    # #### answer (GSM8K)
    hash_match = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        return hash_match[-1].strip()

    # "The answer is X"
    ans_match = re.findall(
        r'(?:the answer is|answer:|therefore|thus|=)\s*(.+?)(?:\.|,|\n|$)',
        text, re.IGNORECASE
    )
    if ans_match:
        # Clean up: extract just the number
        candidate = ans_match[-1].strip()
        num_match = re.search(r'[-+]?\d*\.?\d+', candidate)
        if num_match:
            return num_match.group()
        return candidate

    return None


def format_prompt_with_template(
    tokenizer, question: str, system_prompt: str = None
) -> str:
    """Format a prompt using the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text
    except Exception:
        # Fallback
        prefix = f"System: {system_prompt}\n" if system_prompt else ""
        return f"{prefix}User: {question}\nAssistant:"


def load_model_4bit(model_path: str, device: str = "cuda"):
    """Load Nemotron in 4-bit with tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def evaluate_gsm8k(
    model,
    tokenizer,
    max_samples: int = 200,
    max_new_tokens: int = 512,
    config_name: str = "base",
) -> EvalResult:
    """Evaluate on GSM8K with Nemotron chat template."""
    from datasets import load_dataset

    logger.info(f"Evaluating GSM8K ({max_samples} samples, config={config_name})")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    examples = []

    system_prompt = (
        "You are a precise mathematics expert. Solve problems step-by-step, "
        "showing all work clearly. Put your final numerical answer in \\boxed{}."
    )

    for i, example in enumerate(tqdm(ds, desc=f"GSM8K [{config_name}]")):
        question = example["question"]
        gold = example["answer"]
        gold_num = extract_answer(gold)

        prompt = format_prompt_with_template(
            tokenizer, question, system_prompt=system_prompt
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred_num = extract_answer(response)

        is_correct = (
            pred_num is not None
            and gold_num is not None
            and str(pred_num).strip() == str(gold_num).strip()
        )
        if is_correct:
            correct += 1
        total += 1

        if i < 3:
            logger.info(f"  Example {i}: gold={gold_num}, pred={pred_num}, correct={is_correct}")
            examples.append({
                "question": question[:100],
                "gold": gold_num,
                "pred": pred_num,
                "correct": is_correct,
            })

    accuracy = correct / max(total, 1)
    logger.info(f"GSM8K [{config_name}]: {correct}/{total} = {accuracy:.4f}")

    return EvalResult(
        benchmark="gsm8k",
        metric="exact_match",
        score=accuracy,
        num_examples=total,
        config_name=config_name,
        details={"examples": examples},
    )


def evaluate_arc(
    model,
    tokenizer,
    max_samples: int = 200,
    max_new_tokens: int = 128,
    config_name: str = "base",
) -> EvalResult:
    """Evaluate on ARC-Challenge with Nemotron chat template."""
    from datasets import load_dataset

    logger.info(f"Evaluating ARC-Challenge ({max_samples} samples, config={config_name})")

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    system_prompt = (
        "You are a science expert. Answer the following multiple-choice question. "
        "Reply with ONLY the letter of the correct answer (A, B, C, or D)."
    )

    for i, example in enumerate(tqdm(ds, desc=f"ARC [{config_name}]")):
        question = example["question"]
        choices = example["choices"]
        gold = example["answerKey"]

        # Format choices
        choice_text = "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        full_question = f"{question}\n\n{choice_text}"

        prompt = format_prompt_with_template(
            tokenizer, full_question, system_prompt=system_prompt
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Extract answer letter
        pred = None
        for char in response:
            if char.upper() in "ABCDE":
                pred = char.upper()
                break

        if pred == gold.upper():
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info(f"ARC [{config_name}]: {correct}/{total} = {accuracy:.4f}")

    return EvalResult(
        benchmark="arc_challenge",
        metric="accuracy",
        score=accuracy,
        num_examples=total,
        config_name=config_name,
    )


def run_nemotron_evaluation(
    model_path: str,
    adapter_dir: Optional[str] = None,
    output_dir: str = str(PROJECT_ROOT / "results" / "nemotron"),
    max_samples: int = 200,
    eval_baseline: bool = True,
    eval_adapters: bool = True,
    domains: List[str] = None,
):
    """
    Run complete Nemotron evaluation suite.

    Steps:
    1. Baseline (no adapters)
    2. Single adapter per domain
    3. Save comparison table
    """
    domains = domains or ["math", "code", "science"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── 1. Baseline ──
    if eval_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE EVALUATION (No Adapters)")
        logger.info("=" * 60)

        model, tokenizer = load_model_4bit(model_path)

        base_gsm8k = evaluate_gsm8k(model, tokenizer, max_samples, config_name="baseline")
        all_results["baseline_gsm8k"] = asdict(base_gsm8k)

        base_arc = evaluate_arc(model, tokenizer, max_samples, config_name="baseline")
        all_results["baseline_arc"] = asdict(base_arc)

        del model
        torch.cuda.empty_cache()

    # ── 2. Single Adapters ──
    if eval_adapters and adapter_dir:
        logger.info("\n" + "=" * 60)
        logger.info("SINGLE ADAPTER EVALUATION")
        logger.info("=" * 60)

        for domain in domains:
            # Find adapter
            adapter_path = None
            for subdir in ["dare_sparsified", "best", "final"]:
                candidate = Path(adapter_dir) / domain / subdir
                if candidate.exists() and (candidate / "adapter_config.json").exists():
                    adapter_path = candidate
                    break

            if adapter_path is None:
                logger.warning(f"No adapter found for '{domain}'")
                continue

            logger.info(f"\nLoading adapter: {domain} from {adapter_path}")
            model, tokenizer = load_model_4bit(model_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

            if domain == "math":
                result = evaluate_gsm8k(model, tokenizer, max_samples, config_name=f"single_{domain}")
                all_results[f"single_{domain}_gsm8k"] = asdict(result)

            if domain == "science":
                result = evaluate_arc(model, tokenizer, max_samples, config_name=f"single_{domain}")
                all_results[f"single_{domain}_arc"] = asdict(result)

            del model
            torch.cuda.empty_cache()

    # ── Save Results ──
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("NEMOTRON EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Benchmark':<15} {'Score':>10}")
    print(f"{'-'*55}")
    for name, result in all_results.items():
        print(f"{result['config_name']:<30} {result['benchmark']:<15} {result['score']:>10.4f}")
    print(f"{'='*60}")

    logger.info(f"Results saved to {results_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Nemotron Evaluation Suite")
    parser.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "models" / "nemotron"))
    parser.add_argument("--adapter_dir", type=str, default=str(PROJECT_ROOT / "checkpoints" / "nemotron_lori" / "adapters"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "nemotron"))
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument("--no_adapters", action="store_true")
    parser.add_argument("--domains", nargs="+", default=["math", "code", "science"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_nemotron_evaluation(
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        eval_baseline=not args.no_baseline,
        eval_adapters=not args.no_adapters,
        domains=args.domains,
    )


if __name__ == "__main__":
    main()
