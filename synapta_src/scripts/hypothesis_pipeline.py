#!/usr/bin/env python3
"""
MEWTWO Exhaustive Hypothesis Testing Pipeline
=============================================
Runs 15 experiments autonomously. Zero retraining. Pure evaluation.

Phase 1: Baselines (3 tests)      — raw Nemotron on GSM8K, HumanEval, ARC
Phase 2: Single Adapters (9 tests) — each adapter × each benchmark
Phase 3: Composition (3 tests)     — merged adapter on all benchmarks
Phase 4: Auto-Analysis             — compute deltas, PASS/FAIL verdicts

Usage:
    nohup .venv/bin/python scripts/hypothesis_pipeline.py 2>&1 | tee logs/nemotron/hypothesis.log &
"""
import gc
import json
import re
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
MERGED_ADAPTER = PROJECT / "adapters" / "submission"
RESULTS_DIR = PROJECT / "results" / "nemotron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(PROJECT / "logs" / "nemotron" / "hypothesis.log")),
    ],
)
log = logging.getLogger("hypothesis")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ─── Result storage ─────────────────────────────────────────────

ALL_RESULTS = []


@dataclass
class TestResult:
    test_id: str
    phase: int
    config: str       # "baseline", "math", "code", "science", "merged"
    benchmark: str     # "gsm8k", "humaneval", "arc"
    score: float
    num_correct: int
    num_total: int
    time_seconds: float
    details: dict = None


# ─── Model management ───────────────────────────────────────────

_cached_tokenizer = None
_cached_model = None
_cached_config = None


def get_tokenizer():
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        if _cached_tokenizer.pad_token is None:
            _cached_tokenizer.pad_token = _cached_tokenizer.eos_token
    return _cached_tokenizer


def load_model(adapter_path: Optional[str] = None, config_name: str = "baseline"):
    """Load model, reusing base if possible. Returns (model, config_name)."""
    global _cached_model, _cached_config

    # If we need a different config, unload everything
    if _cached_config != "base" and _cached_config is not None:
        del _cached_model
        _cached_model = None
        _cached_config = None
        gc.collect()
        torch.cuda.empty_cache()

    # Load base model if not cached
    if _cached_model is None:
        log.info(f"Loading base model from {MODEL_PATH}...")
        _cached_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
        )
        _cached_config = "base"
        vram = torch.cuda.memory_allocated() / 1e9
        log.info(f"Base model loaded. VRAM: {vram:.1f} GB")

    if adapter_path is None:
        return _cached_model

    # Load adapter on top
    log.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(_cached_model, adapter_path, is_trainable=False)
    model.eval()
    return model


def unload_adapter(model):
    """Remove adapter, keep base model cached."""
    global _cached_model
    if isinstance(model, PeftModel):
        model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def find_adapter(domain: str) -> Optional[str]:
    """Find best adapter checkpoint."""
    for sub in ["best", "dare_sparsified", "final"]:
        p = ADAPTER_BASE / domain / sub
        if (p / "adapter_config.json").exists():
            return str(p)
    return None


# ─── Answer extraction ───────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """Extract answer from model output."""
    # \boxed{answer}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    # #### answer (GSM8K)
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hashes:
        # Clean: remove commas from numbers
        ans = hashes[-1].strip().replace(",", "")
        return ans
    # "The answer is X"
    ans_match = re.findall(r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if ans_match:
        return ans_match[-1].strip()
    return None


def normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    # Try to parse as float for numeric comparison
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return s.lower().strip()


# ─── Chat template helper ────────────────────────────────────────

def format_prompt(tokenizer, system: str, user: str) -> str:
    """Format using the model's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


# ─── Benchmark: GSM8K ────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, max_samples=100, label="") -> TestResult:
    """Evaluate on GSM8K with proper chat template."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    system = "You are a precise mathematics expert. Solve problems step-by-step. Put your final numerical answer after ####."
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(tqdm(ds, desc=f"GSM8K [{label}]")):
            question = ex["question"]
            gold = extract_answer(ex["answer"])
            gold_norm = normalize_number(gold)

            prompt = format_prompt(tokenizer, system, question)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = extract_answer(response)
            pred_norm = normalize_number(pred)

            if pred_norm and gold_norm and pred_norm == gold_norm:
                correct += 1
            total += 1

            if i < 3:
                log.info(f"  [{label}] GSM8K #{i}: gold={gold_norm}, pred={pred_norm}, match={pred_norm==gold_norm}")

    score = correct / max(total, 1)
    log.info(f"GSM8K [{label}]: {correct}/{total} = {score:.4f}")
    return TestResult(
        test_id="", phase=0, config=label, benchmark="gsm8k",
        score=score, num_correct=correct, num_total=total, time_seconds=0,
    )


# ─── Benchmark: HumanEval ────────────────────────────────────────

def eval_humaneval(model, tokenizer, max_samples=100, label="") -> TestResult:
    """Evaluate on HumanEval with proper chat template and code execution."""
    from datasets import load_dataset
    import tempfile, subprocess

    ds = load_dataset("openai/openai_humaneval", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    system = "You are an expert Python programmer. Complete the given function. Output ONLY the Python code, no explanations."
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(tqdm(ds, desc=f"HumanEval [{label}]")):
            prompt_code = ex["prompt"]
            test_code = ex["test"]
            entry_point = ex["entry_point"]

            user_msg = f"Complete this Python function:\n\n```python\n{prompt_code}\n```\n\nWrite ONLY the complete function implementation."
            prompt = format_prompt(tokenizer, system, user_msg)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Extract code from response
            code = response
            # Try to find code block
            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)

            # If response starts with the function continuation, prepend the prompt
            if not code.strip().startswith("def "):
                code = prompt_code + code

            # Build test harness
            full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

            # Execute safely
            passed = False
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
                    f.write(full_code)
                    f.flush()
                    result = subprocess.run(
                        [sys.executable, f.name],
                        capture_output=True, timeout=10, text=True,
                    )
                    passed = result.returncode == 0
            except (subprocess.TimeoutExpired, Exception):
                passed = False

            if passed:
                correct += 1
            total += 1

            if i < 3:
                log.info(f"  [{label}] HumanEval #{i}: passed={passed}")

    score = correct / max(total, 1)
    log.info(f"HumanEval [{label}]: {correct}/{total} = {score:.4f}")
    return TestResult(
        test_id="", phase=0, config=label, benchmark="humaneval",
        score=score, num_correct=correct, num_total=total, time_seconds=0,
    )


# ─── Benchmark: ARC-Challenge ────────────────────────────────────

def eval_arc(model, tokenizer, max_samples=100, label="") -> TestResult:
    """Evaluate on ARC-Challenge with proper MCQ formatting."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))

    system = "You are a science expert. Answer the multiple choice question. Reply with ONLY the letter (A, B, C, or D)."
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, ex in enumerate(tqdm(ds, desc=f"ARC [{label}]")):
            question = ex["question"]
            choices = ex["choices"]
            gold = ex["answerKey"].strip().upper()

            # Format choices
            choice_text = ""
            for label_c, text in zip(choices["label"], choices["text"]):
                choice_text += f"\n{label_c}. {text}"

            user_msg = f"{question}\n{choice_text}\n\nAnswer:"
            prompt = format_prompt(tokenizer, system, user_msg)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Extract letter answer
            pred = ""
            # Look for standalone letter
            letter_match = re.search(r'\b([A-D])\b', response[:20])
            if letter_match:
                pred = letter_match.group(1).upper()

            if pred == gold:
                correct += 1
            total += 1

            if i < 3:
                log.info(f"  [{label}] ARC #{i}: gold={gold}, pred={pred}, match={pred==gold}")

    score = correct / max(total, 1)
    log.info(f"ARC [{label}]: {correct}/{total} = {score:.4f}")
    return TestResult(
        test_id="", phase=0, config=label, benchmark="arc",
        score=score, num_correct=correct, num_total=total, time_seconds=0,
    )


# ─── Test runner ──────────────────────────────────────────────────

def run_test(test_id: str, phase: int, config: str, benchmark: str,
             adapter_path: Optional[str] = None):
    """Run a single test and record results."""
    log.info(f"\n{'='*60}")
    log.info(f"TEST {test_id}: [{config}] on [{benchmark}]")
    log.info(f"{'='*60}")

    tokenizer = get_tokenizer()
    t0 = time.time()

    model = load_model(adapter_path, config)

    bench_fn = {"gsm8k": eval_gsm8k, "humaneval": eval_humaneval, "arc": eval_arc}
    samples = {"gsm8k": 100, "humaneval": 100, "arc": 100}

    result = bench_fn[benchmark](model, tokenizer, max_samples=samples[benchmark], label=config)
    result.test_id = test_id
    result.phase = phase
    result.time_seconds = time.time() - t0

    ALL_RESULTS.append(result)

    # Unload adapter if one was loaded
    if adapter_path:
        unload_adapter(model)

    # Save incrementally
    save_results()

    log.info(f"TEST {test_id} DONE: {result.score:.4f} ({result.num_correct}/{result.num_total}) in {result.time_seconds:.0f}s\n")
    return result


def save_results():
    """Save all results incrementally."""
    data = [asdict(r) for r in ALL_RESULTS]
    with open(RESULTS_DIR / "hypothesis_matrix.json", "w") as f:
        json.dump(data, f, indent=2)


# ─── Phase 4: Analysis ───────────────────────────────────────────

def run_analysis():
    """Generate the final results table and hypothesis verdicts."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: AUTO-ANALYSIS")
    log.info("=" * 70)

    # Build lookup
    scores = {}
    for r in ALL_RESULTS:
        scores[(r.config, r.benchmark)] = r.score

    # Results matrix
    benchmarks = ["gsm8k", "humaneval", "arc"]
    configs = ["baseline", "math", "code", "science", "merged"]

    log.info("\n" + "─" * 70)
    log.info(f"{'Config':<15} {'GSM8K':>10} {'HumanEval':>12} {'ARC':>10}")
    log.info("─" * 70)
    for config in configs:
        vals = []
        for bench in benchmarks:
            score = scores.get((config, bench))
            if score is not None:
                vals.append(f"{score:.1%}")
            else:
                vals.append("—")
        log.info(f"{config:<15} {vals[0]:>10} {vals[1]:>12} {vals[2]:>10}")
    log.info("─" * 70)

    # Deltas
    log.info("\nDELTAS vs BASELINE:")
    log.info("─" * 70)
    for config in ["math", "code", "science", "merged"]:
        deltas = []
        for bench in benchmarks:
            base = scores.get(("baseline", bench))
            adapter = scores.get((config, bench))
            if base is not None and adapter is not None:
                d = adapter - base
                sign = "+" if d >= 0 else ""
                deltas.append(f"{sign}{d:.1%}")
            else:
                deltas.append("—")
        log.info(f"{config:<15} {deltas[0]:>10} {deltas[1]:>12} {deltas[2]:>10}")
    log.info("─" * 70)

    # Hypothesis verdicts
    log.info("\nHYPOTHESIS VERDICTS:")
    log.info("=" * 70)

    # H1: Adapters help their own domain
    h1_results = []
    for config, bench in [("math", "gsm8k"), ("code", "humaneval"), ("science", "arc")]:
        base = scores.get(("baseline", bench), 0)
        adapt = scores.get((config, bench), 0)
        delta = adapt - base
        h1_results.append((config, bench, delta))
        log.info(f"  H1 [{config}→{bench}]: Δ = {delta:+.1%} {'✅ PASS' if delta >= 0.03 else '❌ FAIL'}")

    h1_pass = any(d >= 0.03 for _, _, d in h1_results)
    log.info(f"  H1 Overall: {'✅ PASS' if h1_pass else '❌ FAIL'}")

    # H2: Cross-domain transfer
    cross_tests = [
        ("math", "humaneval"), ("math", "arc"),
        ("code", "gsm8k"), ("code", "arc"),
        ("science", "gsm8k"), ("science", "humaneval"),
    ]
    h2_results = []
    for config, bench in cross_tests:
        base = scores.get(("baseline", bench), 0)
        adapt = scores.get((config, bench), 0)
        delta = adapt - base
        h2_results.append((config, bench, delta))

    h2_pass = any(d >= 0.03 for _, _, d in h2_results)
    best_cross = max(h2_results, key=lambda x: x[2])
    log.info(f"  H2 Best cross-domain: {best_cross[0]}→{best_cross[1]}: Δ = {best_cross[2]:+.1%}")
    log.info(f"  H2 Overall: {'✅ PASS' if h2_pass else '❌ FAIL'}")

    # H3: Composition beats individuals
    h3_results = []
    for bench in benchmarks:
        merged = scores.get(("merged", bench), 0)
        best_single = max(
            scores.get(("math", bench), 0),
            scores.get(("code", bench), 0),
            scores.get(("science", bench), 0),
        )
        delta = merged - best_single
        h3_results.append((bench, delta))
        log.info(f"  H3 [{bench}]: merged={merged:.1%}, best_single={best_single:.1%}, Δ = {delta:+.1%}")

    h3_pass = any(d > 0 for _, d in h3_results)
    log.info(f"  H3 Overall: {'✅ PASS' if h3_pass else '❌ FAIL'}")

    # H4: Format was the bottleneck
    old_baseline_gsm8k = 0.82
    old_baseline_arc = 0.00
    new_baseline_gsm8k = scores.get(("baseline", "gsm8k"), 0)
    new_baseline_arc = scores.get(("baseline", "arc"), 0)
    h4_gsm8k_delta = new_baseline_gsm8k - old_baseline_gsm8k
    h4_arc_delta = new_baseline_arc - old_baseline_arc
    log.info(f"  H4 [GSM8K]: old_baseline={old_baseline_gsm8k:.1%}, new={new_baseline_gsm8k:.1%}, Δ = {h4_gsm8k_delta:+.1%}")
    log.info(f"  H4 [ARC]:   old_baseline={old_baseline_arc:.1%}, new={new_baseline_arc:.1%}, Δ = {h4_arc_delta:+.1%}")
    h4_pass = abs(h4_gsm8k_delta) >= 0.05 or abs(h4_arc_delta) >= 0.05
    log.info(f"  H4 Overall: {'✅ PASS' if h4_pass else '❌ FAIL'}")

    log.info("\n" + "=" * 70)
    summary = f"H1={'PASS' if h1_pass else 'FAIL'} H2={'PASS' if h2_pass else 'FAIL'} H3={'PASS' if h3_pass else 'FAIL'} H4={'PASS' if h4_pass else 'FAIL'}"
    log.info(f"VERDICT: {summary}")
    log.info("=" * 70)

    # Save full analysis
    analysis = {
        "scores": {f"{c}_{b}": s for (c, b), s in scores.items()},
        "h1_pass": h1_pass, "h1_details": [(c, b, d) for c, b, d in h1_results],
        "h2_pass": h2_pass, "h2_best": best_cross,
        "h3_pass": h3_pass, "h3_details": [(b, d) for b, d in h3_results],
        "h4_pass": h4_pass,
        "summary": summary,
    }
    with open(RESULTS_DIR / "hypothesis_verdicts.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    log.info(f"\nResults: {RESULTS_DIR / 'hypothesis_matrix.json'}")
    log.info(f"Verdicts: {RESULTS_DIR / 'hypothesis_verdicts.json'}")


# ─── Main pipeline ────────────────────────────────────────────────

def main():
    start = time.time()

    log.info("=" * 70)
    log.info("MEWTWO EXHAUSTIVE HYPOTHESIS TESTING PIPELINE")
    log.info(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    adapters = {
        "math": find_adapter("math"),
        "code": find_adapter("code"),
        "science": find_adapter("science"),
        "merged": str(MERGED_ADAPTER) if MERGED_ADAPTER.exists() else None,
    }
    log.info("Adapters found:")
    for k, v in adapters.items():
        log.info(f"  {k}: {v or 'NOT FOUND'}")

    # ─── Phase 1: Baselines ───
    log.info("\n" + "█" * 70)
    log.info("PHASE 1: COMPLETE BASELINE MATRIX")
    log.info("█" * 70)

    run_test("1.1", 1, "baseline", "gsm8k")
    run_test("1.2", 1, "baseline", "humaneval")
    run_test("1.3", 1, "baseline", "arc")

    # ─── Phase 2: Single Adapters ───
    log.info("\n" + "█" * 70)
    log.info("PHASE 2: SINGLE-ADAPTER MATRIX (9 TESTS)")
    log.info("█" * 70)

    for domain in ["math", "code", "science"]:
        path = adapters[domain]
        if not path:
            log.warning(f"Skipping {domain} — no adapter found")
            continue
        for bench in ["gsm8k", "humaneval", "arc"]:
            test_num = {"math": {"gsm8k": "2.1", "humaneval": "2.2", "arc": "2.3"},
                        "code": {"gsm8k": "2.4", "humaneval": "2.5", "arc": "2.6"},
                        "science": {"gsm8k": "2.7", "humaneval": "2.8", "arc": "2.9"}}
            run_test(test_num[domain][bench], 2, domain, bench, path)

    # ─── Phase 3: Composition ───
    log.info("\n" + "█" * 70)
    log.info("PHASE 3: COMPOSITION TESTS")
    log.info("█" * 70)

    if adapters["merged"]:
        run_test("3.1", 3, "merged", "gsm8k", adapters["merged"])
        run_test("3.2", 3, "merged", "humaneval", adapters["merged"])
        run_test("3.3", 3, "merged", "arc", adapters["merged"])
    else:
        log.warning("Merged adapter not found, skipping Phase 3")

    # ─── Phase 4: Analysis ───
    run_analysis()

    elapsed = (time.time() - start) / 60
    log.info(f"\nPIPELINE COMPLETE. Total time: {elapsed:.0f} minutes")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
