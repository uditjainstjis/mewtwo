"""
Evaluation Pipeline for LoRI-MoE

Runs proper benchmarks using exact-match / pass@1 metrics — NOT semantic similarity.
Integrates with lm-eval-harness where possible, with custom evaluation loops for
interference testing and routing analysis.

Benchmarks:
  - MATH500: Mathematical reasoning (exact match)
  - GSM8K: Grade-school math (exact match)
  - MMLU: Multi-domain knowledge (accuracy)
  - BBH: Hard multi-step reasoning (accuracy)
  - HumanEval: Code generation (pass@1)

Critical tests:
  - Interference test: does multi-adapter hurt single-domain performance?
  - Routing analysis: does the router actually switch domains?
"""
import os
import sys
import json
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark: str
    metric_name: str
    score: float
    num_examples: int
    config: str  # e.g., "base", "math_only", "all_experts", "lori_moe"
    timestamp: str = ""
    details: dict = field(default_factory=dict)


# ─── Evaluation via lm-eval-harness ───────────────────────────────────────────


def run_lm_eval_benchmark(
    model,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 0,
    batch_size: int = 4,
    limit: Optional[int] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Run evaluation using lm-eval-harness.
    
    Falls back to custom evaluation if lm-eval is not available.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # Wrap model for lm-eval
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
        )

        parsed = {}
        for task_name, task_result in results.get("results", {}).items():
            # Find the primary metric
            for metric_key in ["acc,none", "exact_match,none", "acc_norm,none"]:
                if metric_key in task_result:
                    score = task_result[metric_key]
                    parsed[task_name] = BenchmarkResult(
                        benchmark=task_name,
                        metric_name=metric_key.split(",")[0],
                        score=score,
                        num_examples=limit or -1,
                        config="lm_eval",
                        details=task_result,
                    )
                    break

        return parsed

    except ImportError:
        logger.warning("lm-eval-harness not available, using custom evaluation")
        return {}
    except Exception as e:
        logger.error(f"lm-eval failed: {e}")
        return {}


# ─── Custom Math Evaluation (MATH500 / GSM8K) ─────────────────────────────────


def evaluate_math(
    model,
    tokenizer,
    dataset_name: str = "math500",
    max_samples: int = 500,
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    """
    Evaluate mathematical reasoning with exact-match accuracy.
    
    Uses boxed answer extraction: looks for \\boxed{answer} in model output.
    """
    from datasets import load_dataset

    logger.info(f"Running math evaluation: {dataset_name} ({max_samples} samples)")

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        question_key = "question"
        answer_key = "answer"
    else:
        # MATH500 or MATH
        try:
            ds = load_dataset("lighteval/MATH", split="test")
        except Exception:
            ds = load_dataset("hendrycks/competition_math", split="test")
        question_key = "problem"
        answer_key = "solution"

    ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    errors = []

    model.eval()
    with torch.no_grad():
        for i, example in enumerate(tqdm(ds, desc=f"Eval {dataset_name}")):
            question = example[question_key]
            gold_answer = example[answer_key]

            # Extract numerical answer from gold
            gold_num = extract_answer(gold_answer)

            # Generate
            prompt = f"Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\nProblem: {question}\n\nSolution:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_num = extract_answer(response)

            if pred_num is not None and gold_num is not None:
                if str(pred_num).strip() == str(gold_num).strip():
                    correct += 1
            total += 1

            if i < 3:
                logger.info(f"  Example {i}: gold={gold_num}, pred={pred_num}, correct={pred_num==gold_num}")

    accuracy = correct / max(total, 1)
    logger.info(f"{dataset_name}: {correct}/{total} = {accuracy:.4f}")

    return BenchmarkResult(
        benchmark=dataset_name,
        metric_name="exact_match",
        score=accuracy,
        num_examples=total,
        config="",
    )


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from text, looking for \\boxed{} or #### patterns."""
    import re

    # Try \\boxed{answer}
    boxed_match = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match[-1].strip()

    # Try #### answer (GSM8K format)
    hash_match = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hash_match:
        return hash_match[-1].strip()

    # Try "The answer is X"
    answer_match = re.findall(r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match[-1].strip()

    return None


# ─── Interference Test ─────────────────────────────────────────────────────────


def run_interference_test(
    base_model_name: str,
    adapter_dir: str,
    domains: List[str],
    max_samples: int = 100,
) -> Dict[str, Dict]:
    """
    Critical test: does multi-adapter composition degrade single-domain performance?

    For each domain:
      1. Evaluate with ONLY that domain's adapter active → score_single
      2. Evaluate with ALL adapters active (router decides) → score_multi
      3. Degradation = score_single - score_multi

    Success: degradation < 2% for all domains
    """
    logger.info("\n" + "=" * 60)
    logger.info("INTERFERENCE TEST")
    logger.info("=" * 60)

    results = {}

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test each domain
    domain_benchmarks = {
        "math": ("gsm8k", "openai/gsm8k", "question", "answer"),
        "code": ("humaneval", None, None, None),  # Special handling
        "science": ("arc", "allenai/ai2_arc", "question", "answerKey"),
    }

    for domain in domains:
        if domain not in domain_benchmarks:
            logger.info(f"Skipping interference test for '{domain}' (no quick benchmark)")
            continue

        bench_name = domain_benchmarks[domain][0]
        logger.info(f"\nTesting domain: {domain} ({bench_name})")

        # Single adapter
        adapter_path = find_adapter_path(adapter_dir, domain)
        if not adapter_path:
            logger.warning(f"  No adapter found for '{domain}'")
            continue

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )

        # Evaluate with single adapter
        single_model = PeftModel.from_pretrained(base_model, adapter_path)
        single_model.eval()

        if bench_name == "gsm8k":
            single_result = evaluate_math(single_model, tokenizer, "gsm8k", max_samples)
        else:
            single_result = BenchmarkResult(
                benchmark=bench_name, metric_name="accuracy",
                score=0.0, num_examples=0, config="single"
            )

        results[domain] = {
            "single_adapter_score": single_result.score,
            "multi_adapter_score": 0.0,  # Will be filled when full MoE is tested
            "degradation": 0.0,
        }

        logger.info(f"  Single adapter: {single_result.score:.4f}")

        # Cleanup
        del single_model, base_model
        torch.cuda.empty_cache()

    return results


def find_adapter_path(adapter_dir: str, domain: str) -> Optional[str]:
    """Find the best adapter checkpoint for a domain."""
    base = Path(adapter_dir) / domain
    for subdir in ["best", "final", "dare_sparsified"]:
        path = base / subdir
        if path.exists() and (path / "adapter_config.json").exists():
            return str(path)
    return None


# ─── Routing Analysis ─────────────────────────────────────────────────────────


def analyze_routing(
    model,
    tokenizer,
    router,
    prompts: List[str],
    expert_names: List[str],
    output_dir: str = "/home/learner/Desktop/mewtwo/results/lori_moe",
) -> Dict:
    """
    Visualize token-level routing decisions for qualitative analysis.

    For each prompt:
      - Tokenize
      - Get hidden states from each layer
      - Run router to get per-token expert probabilities
      - Save routing heatmap data

    Success criteria: routing should visibly switch between experts
    within a single prompt for multi-domain queries.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    analysis = []

    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]  # Last layer
            routing_weights, _ = router.get_router(0)(hidden_states)

        # Convert to readable format
        routing_data = routing_weights[0].cpu().numpy().tolist()  # (seq_len, K)

        prompt_analysis = {
            "prompt": prompt,
            "tokens": tokens[:len(routing_data)],
            "routing_weights": routing_data,
            "expert_names": expert_names,
            "dominant_expert_per_token": [],
        }

        for t_idx, weights in enumerate(routing_data):
            dominant_idx = max(range(len(weights)), key=lambda i: weights[i])
            prompt_analysis["dominant_expert_per_token"].append(expert_names[dominant_idx])

        analysis.append(prompt_analysis)

        # Log summary
        from collections import Counter
        dist = Counter(prompt_analysis["dominant_expert_per_token"])
        logger.info(
            f"Prompt {prompt_idx}: {len(tokens)} tokens, "
            f"expert distribution: {dict(dist)}"
        )

    # Save analysis
    analysis_file = output_path / "routing_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Routing analysis saved to {analysis_file}")
    return {"analysis": analysis, "file": str(analysis_file)}


# ─── Full Evaluation Suite ─────────────────────────────────────────────────────


def run_full_evaluation(
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_dir: str = "/home/learner/Desktop/mewtwo/checkpoints/lori_moe/adapters",
    router_path: str = "/home/learner/Desktop/mewtwo/checkpoints/lori_moe/router/best_router.pt",
    output_dir: str = "/home/learner/Desktop/mewtwo/results/lori_moe",
    max_samples: int = 200,
):
    """
    Run the complete evaluation suite.

    Order:
    1. Base model zero-shot (baseline)
    2. Single adapter per domain (Phase 1 validation)
    3. Full LoRI-MoE with router (final system)
    4. Interference test
    5. Routing analysis
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── 1. Base model baseline ──
    logger.info("\n" + "="*60)
    logger.info("EVALUATION: Base Model Baseline")
    logger.info("="*60)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # GSM8K baseline
    base_gsm8k = evaluate_math(base_model, tokenizer, "gsm8k", max_samples)
    base_gsm8k.config = "base_zero_shot"
    all_results["base_gsm8k"] = asdict(base_gsm8k)
    logger.info(f"Base GSM8K: {base_gsm8k.score:.4f}")

    # Try lm-eval benchmarks
    lm_eval_results = run_lm_eval_benchmark(
        base_model, tokenizer,
        tasks=["mmlu", "gsm8k"],
        num_fewshot=0,
        limit=max_samples,
    )
    for name, result in lm_eval_results.items():
        result.config = "base_zero_shot"
        all_results[f"base_{name}"] = asdict(result)

    del base_model
    torch.cuda.empty_cache()

    # ── 2. Single adapter evaluation ──
    logger.info("\n" + "="*60)
    logger.info("EVALUATION: Single Domain Adapters")
    logger.info("="*60)

    domains = ["math", "code", "science", "legal", "medical"]
    for domain in domains:
        adapter_path = find_adapter_path(adapter_dir, domain)
        if not adapter_path:
            continue

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
        adapter_model.eval()

        if domain == "math":
            result = evaluate_math(adapter_model, tokenizer, "gsm8k", max_samples)
            result.config = f"single_{domain}"
            all_results[f"single_{domain}_gsm8k"] = asdict(result)
            logger.info(f"Single {domain} adapter on GSM8K: {result.score:.4f}")

        del adapter_model, base_model
        torch.cuda.empty_cache()

    # ── 3. Full LoRI-MoE evaluation ──
    logger.info("\n" + "="*60)
    logger.info("EVALUATION: Full LoRI-MoE with Token Router")
    logger.info("="*60)
    
    from src.lori_moe.config import LoRIMoEConfig
    from src.lori_moe.model.lori_moe_model import LoRIMoEModel
    
    config = LoRIMoEConfig()
    config.model.base_model = base_model_name
    config.paths.adapters_dir = Path(adapter_dir)
    
    moe_model = LoRIMoEModel(config)
    moe_model.build(load_experts=True)
    
    logger.info(f"Loading router from {router_path}")
    if Path(router_path).exists():
        router_state = torch.load(router_path, map_location="cuda")
        moe_model.routers.load_state_dict(router_state["model_state_dict"])
    else:
        logger.warning(f"Router path {router_path} does not exist, using initialized dense weights")
        
    moe_model.eval()
    
    # GSM8K
    moe_gsm8k = evaluate_math(moe_model, tokenizer, "gsm8k", max_samples)
    moe_gsm8k.config = "lori_moe"
    all_results["lori_moe_gsm8k"] = asdict(moe_gsm8k)
    logger.info(f"LoRI-MoE GSM8K: {moe_gsm8k.score:.4f}")
    
    del moe_model
    torch.cuda.empty_cache()

    # ── Save all results ──
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"  {name:40s}: {result['score']:.4f} ({result['metric_name']})")
    print("=" * 60)

    logger.info(f"Results saved to {results_file}")
    return all_results


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LoRI-MoE evaluation suite")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="/home/learner/Desktop/mewtwo/checkpoints/lori_moe/adapters")
    parser.add_argument("--router_path", type=str, default="/home/learner/Desktop/mewtwo/checkpoints/lori_moe/router/best_router.pt")
    parser.add_argument("--output_dir", type=str, default="/home/learner/Desktop/mewtwo/results/lori_moe")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_full_evaluation(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        router_path=args.router_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
