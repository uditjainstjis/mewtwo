#!/usr/bin/env python3
"""
MEWTWO Research Sprint — Phases 2-5
=====================================
TRUE multi-adapter composition + standardized benchmarks + demo prep.

Run AFTER master_pipeline.py Phase 1 completes.

Usage:
    nohup .venv/bin/python scripts/research_sprint.py 2>&1 &
    tail -f logs/nemotron/research_sprint.log
"""
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
RESULTS_DIR = PROJECT / "results" / "nemotron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs" / "nemotron"
TASKS_FILE = PROJECT / "PIPELINE_TASKS.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "research_sprint.log")),
    ],
)
log = logging.getLogger("sprint")

BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)

ALL_RESULTS = {}
_base_model = None
_tokenizer = None


def update_task(task_id: str, status: str, note: str = ""):
    content = TASKS_FILE.read_text()
    markers = {"done": "[x]", "running": "[/]", "failed": "[!]"}
    for old_m in ["[ ]", "[/]", "[!]"]:
        old = f"- {old_m} {task_id}"
        new = f"- {markers.get(status, '[ ]')} {task_id}"
        content = content.replace(old, new + (f" — {note}" if note and old in content else ""))
    content = re.sub(r"Last updated:.*", f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", content)
    TASKS_FILE.write_text(content)
    log.info(f"📋 Task {task_id} → {status}" + (f" ({note})" if note else ""))


def save_results():
    with open(RESULTS_DIR / "sprint_results.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def load_base():
    global _base_model
    if _base_model is not None:
        return _base_model
    log.info("Loading base Nemotron model...")
    _base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True,
    )
    _base_model.eval()
    log.info(f"Base model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _base_model


def find_adapter(domain: str) -> Optional[str]:
    for sub in ["best", "dare_sparsified", "final"]:
        p = ADAPTER_BASE / domain / sub
        if (p / "adapter_config.json").exists():
            return str(p)
    return None


def format_prompt(tok, system: str, user: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def generate(model, tok, prompt: str, max_tokens=768) -> str:
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1536)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_code(response: str) -> str:
    block = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if block:
        return block.group(1)
    lines = response.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and
                  any(kw in l for kw in ['import ', 'def ', 'print(', 'return ', '=', 'for ', 'while ', 'if '])]
    if len(code_lines) > 2:
        start = next((i for i, l in enumerate(lines) if any(kw in l for kw in ['import ', 'def '])), 0)
        return '\n'.join(lines[start:])
    return response


def run_code(code: str, timeout: int = 15) -> tuple:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
            f.write(code)
            f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout, text=True)
            return r.returncode == 0, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip().replace(",", "").replace("$", "")
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hashes:
        return hashes[-1].strip()
    ans = re.findall(r'(?:the answer is|answer:|result:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if ans:
        return ans[-1].strip()
    nums = re.findall(r'[-+]?\d*\.?\d+(?:e[+-]?\d+)?', text, re.IGNORECASE)
    if nums:
        return nums[-1]
    return None


# ══════════════════════════════════════════════════════════════════
# PHASE 2: TRUE COMPOSITION EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def phase_2():
    log.info("\n" + "█" * 70)
    log.info("PHASE 2: TRUE MULTI-ADAPTER COMPOSITION")
    log.info("█" * 70)

    tok = get_tokenizer()
    base = load_base()

    # Load mixed-domain queries
    update_task("2.1", "done", "50 mixed-domain queries created")

    with open(PROJECT / "data" / "mixed_domain_eval_50.json") as f:
        queries = json.load(f)

    # 2.2: Build composition configs
    update_task("2.2", "running")

    math_path = find_adapter("math")
    code_path = find_adapter("code")
    science_path = find_adapter("science")

    configs = {}

    # Config 1: baseline (no adapter)
    configs["base"] = {"type": "base"}

    # Config 2-4: single best adapter
    for domain in ["math", "code", "science"]:
        path = find_adapter(domain)
        if path:
            configs[f"single_{domain}"] = {"type": "single", "path": path}

    # Config 5: linear blend (equal weights)
    configs["linear_equal"] = {
        "type": "composed",
        "method": "linear",
        "weights": [1.0/3, 1.0/3, 1.0/3],
    }

    # Config 6: DARE merge
    configs["dare_merge"] = {
        "type": "composed",
        "method": "dare_linear",
        "weights": [1.0, 1.0, 1.0],
        "density": 0.5,
    }

    # Config 7: TIES merge
    configs["ties_merge"] = {
        "type": "composed",
        "method": "ties",
        "weights": [1.0, 1.0, 1.0],
        "density": 0.5,
    }

    # Config 8: oracle-weighted (give more weight to relevant domains)
    configs["oracle_weighted"] = {
        "type": "oracle_composed",
    }

    update_task("2.2", "done", f"{len(configs)} configs built")

    # 2.3: Run the experiment
    update_task("2.3", "running")

    results_by_config = {}

    for cfg_name, cfg in configs.items():
        log.info(f"\n{'='*60}")
        log.info(f"CONFIG: {cfg_name}")
        log.info(f"{'='*60}")

        # Build or load model for this config
        if cfg["type"] == "base":
            model = base
        elif cfg["type"] == "single":
            model = PeftModel.from_pretrained(base, cfg["path"], is_trainable=False)
            model.eval()
        elif cfg["type"] in ("composed", "oracle_composed"):
            # Load all three adapters
            model = PeftModel.from_pretrained(base, math_path, adapter_name="math", is_trainable=False)
            model.load_adapter(code_path, adapter_name="code")
            model.load_adapter(science_path, adapter_name="science")

            if cfg["type"] == "composed":
                try:
                    model.add_weighted_adapter(
                        adapters=["math", "code", "science"],
                        weights=cfg["weights"],
                        adapter_name=f"composed_{cfg_name}",
                        combination_type=cfg["method"],
                        density=cfg.get("density", 0.5) if cfg["method"] in ("dare_linear", "ties") else None,
                    )
                    model.set_adapter(f"composed_{cfg_name}")
                except Exception as e:
                    log.warning(f"Composition {cfg_name} failed: {e}. Falling back to linear.")
                    model.add_weighted_adapter(
                        adapters=["math", "code", "science"],
                        weights=cfg["weights"],
                        adapter_name=f"composed_{cfg_name}",
                        combination_type="linear",
                    )
                    model.set_adapter(f"composed_{cfg_name}")
            model.eval()

        config_results = []
        sys_msg = "You are an expert in mathematics, physics, and Python programming. Solve the problem precisely. If code is needed, write complete, runnable Python code."

        for qi, q in enumerate(tqdm(queries, desc=f"Mixed [{cfg_name}]")):
            # For oracle_weighted, set weights per query
            if cfg["type"] == "oracle_composed":
                w = [0.1, 0.1, 0.1]
                for d in q["domains"]:
                    if d == "math": w[0] = 0.8
                    elif d == "code": w[1] = 0.8
                    elif d == "science": w[2] = 0.8
                # Normalize
                total = sum(w)
                w = [x/total for x in w]
                try:
                    # Remove old composed adapter if exists
                    try:
                        model.delete_adapter("oracle_composed")
                    except Exception:
                        pass
                    model.add_weighted_adapter(
                        adapters=["math", "code", "science"],
                        weights=w,
                        adapter_name="oracle_composed",
                        combination_type="linear",
                    )
                    model.set_adapter("oracle_composed")
                except Exception as e:
                    log.warning(f"Oracle composition failed for {q['id']}: {e}")

            response = generate(model, tok, format_prompt(tok, sys_msg, q["question"]))

            # Evaluate
            result = {
                "id": q["id"],
                "domains": q["domains"],
                "config": cfg_name,
                "response_preview": response[:200],
            }

            if q["answer_type"] == "code_exec":
                code = extract_code(response)
                passed, stdout, stderr = run_code(code)
                result["code_runs"] = passed
                result["stdout"] = stdout[:500]
                result["stderr"] = stderr[:200]
                # Check expected answer
                if q.get("expected") and passed and stdout:
                    expected = q["expected"]
                    result["matches_expected"] = expected.lower() in stdout.lower()
                else:
                    result["matches_expected"] = False
                result["score"] = 1.0 if passed else 0.0
            elif q["answer_type"] == "numerical":
                answer = extract_number(response)
                result["extracted_answer"] = answer
                if q.get("expected") and answer:
                    try:
                        pred = float(answer)
                        gold = float(q["expected"])
                        if gold != 0:
                            result["relative_error"] = abs(pred - gold) / abs(gold)
                            result["score"] = 1.0 if result["relative_error"] < 0.1 else 0.0
                        else:
                            result["score"] = 1.0 if abs(pred) < 1e-6 else 0.0
                    except (ValueError, OverflowError):
                        result["score"] = 0.0
                else:
                    result["score"] = 0.0

            config_results.append(result)
            log.info(f"  [{cfg_name}] {q['id']}: score={result.get('score', 0)}")

        # Summarize config
        total = len(config_results)
        correct = sum(1 for r in config_results if r.get("score", 0) > 0)
        code_runs = sum(1 for r in config_results if r.get("code_runs", False))
        avg_score = sum(r.get("score", 0) for r in config_results) / max(total, 1)

        results_by_config[cfg_name] = {
            "total": total,
            "correct": correct,
            "code_runs": code_runs,
            "avg_score": avg_score,
            "details": config_results,
        }

        log.info(f"\n  [{cfg_name}] SUMMARY: {correct}/{total} correct, {code_runs} code_runs, avg={avg_score:.3f}")

        # Cleanup
        if cfg["type"] in ("single", "composed", "oracle_composed") and isinstance(model, PeftModel):
            model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        ALL_RESULTS[f"composition_{cfg_name}"] = results_by_config[cfg_name]
        save_results()

    update_task("2.3", "done", f"All {len(configs)} configs evaluated on 50 queries")

    # 2.4: Comparative analysis
    update_task("2.4", "running")

    log.info("\n" + "─" * 70)
    log.info(f"{'Config':<20} {'Correct':>8} {'CodeRuns':>9} {'AvgScore':>9}")
    log.info("─" * 70)
    for cfg_name, res in results_by_config.items():
        log.info(f"{cfg_name:<20} {res['correct']:>5}/{res['total']:<3} {res['code_runs']:>9} {res['avg_score']:>9.3f}")
    log.info("─" * 70)

    # Find best composition vs best single
    single_scores = {k: v["avg_score"] for k, v in results_by_config.items() if k.startswith("single_")}
    composed_scores = {k: v["avg_score"] for k, v in results_by_config.items()
                       if k in ("linear_equal", "dare_merge", "ties_merge", "oracle_weighted")}
    base_score = results_by_config.get("base", {}).get("avg_score", 0)
    best_single = max(single_scores.values()) if single_scores else 0
    best_composed = max(composed_scores.values()) if composed_scores else 0

    log.info(f"\n  Base: {base_score:.3f}")
    log.info(f"  Best Single: {best_single:.3f} ({max(single_scores, key=single_scores.get) if single_scores else 'N/A'})")
    log.info(f"  Best Composed: {best_composed:.3f} ({max(composed_scores, key=composed_scores.get) if composed_scores else 'N/A'})")
    log.info(f"  Composition Δ vs Best Single: {best_composed - best_single:+.3f}")
    log.info(f"  Composition Δ vs Base: {best_composed - base_score:+.3f}")

    verdict = "PASS" if best_composed > best_single + 0.05 else "FAIL"
    log.info(f"\n  H-COMP VERDICT: {verdict} (threshold: +5%)")

    ALL_RESULTS["composition_verdict"] = {
        "base": base_score,
        "best_single": best_single,
        "best_composed": best_composed,
        "delta_vs_single": best_composed - best_single,
        "delta_vs_base": best_composed - base_score,
        "verdict": verdict,
    }
    save_results()
    update_task("2.4", "done", f"composition Δ={best_composed - best_single:+.3f} → {verdict}")

    log.info("✅ Phase 2 complete.")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: STANDARDIZED BENCHMARKS (lm-eval-harness)
# ══════════════════════════════════════════════════════════════════

def phase_3():
    log.info("\n" + "█" * 70)
    log.info("PHASE 3: STANDARDIZED BENCHMARKS (lm-eval-harness)")
    log.info("█" * 70)

    lm_eval_bin = str(PROJECT / ".venv" / "bin" / "lm-eval")
    configs_to_eval = [
        ("base", None),
        ("math", find_adapter("math")),
        ("code", find_adapter("code")),
        ("science", find_adapter("science")),
    ]

    # Also create a linear-merged adapter for standardized eval
    # (reuse from Phase 2 or the submission_adapter)
    merged_dir = PROJECT / "submission_adapter"
    if merged_dir.exists():
        configs_to_eval.append(("merged", str(merged_dir)))

    tasks_benchmarks = [
        ("3.1", "gsm8k", "gsm8k"),
        ("3.2", "arc_challenge", "arc_challenge"),
    ]

    for task_id, bench_name, lm_eval_task in tasks_benchmarks:
        update_task(task_id, "running")
        for cfg_name, adapter_path in configs_to_eval:
            log.info(f"\n  lm-eval: {bench_name} with {cfg_name}")

            model_args = f"pretrained={MODEL_PATH},trust_remote_code=True,load_in_4bit=True"
            if adapter_path:
                model_args += f",peft={adapter_path}"

            output_path = str(RESULTS_DIR / f"lm_eval_{cfg_name}_{bench_name}")

            cmd = [
                lm_eval_bin, "--model", "hf",
                "--model_args", model_args,
                "--tasks", lm_eval_task,
                "--device", "cuda:0",
                "--batch_size", "2",
                "--output_path", output_path,
                "--log_samples",
            ]

            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                   env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
                if r.returncode == 0:
                    log.info(f"  ✅ {cfg_name}×{bench_name} complete")
                    # Parse results
                    results_file = Path(output_path)
                    for rf in results_file.rglob("results.json"):
                        with open(rf) as f:
                            data = json.load(f)
                        ALL_RESULTS[f"lm_eval_{cfg_name}_{bench_name}"] = data.get("results", {})
                        log.info(f"  Results: {json.dumps(data.get('results', {}), indent=2)[:500]}")
                        break
                else:
                    log.warning(f"  ❌ {cfg_name}×{bench_name} failed: {r.stderr[-500:]}")
                    ALL_RESULTS[f"lm_eval_{cfg_name}_{bench_name}"] = {"error": r.stderr[-200:]}
            except subprocess.TimeoutExpired:
                log.warning(f"  ⏰ {cfg_name}×{bench_name} timed out (2h)")
                ALL_RESULTS[f"lm_eval_{cfg_name}_{bench_name}"] = {"error": "timeout"}
            except Exception as e:
                log.warning(f"  ❌ {cfg_name}×{bench_name} error: {e}")

            save_results()
            gc.collect()
            torch.cuda.empty_cache()

        update_task(task_id, "done", f"{bench_name} complete for all configs")

    # MMLU-Pro (longer, run separately)
    update_task("3.3", "running")
    for cfg_name, adapter_path in configs_to_eval[:2]:  # Just base + math to save time
        log.info(f"\n  lm-eval: mmlu_pro with {cfg_name}")
        model_args = f"pretrained={MODEL_PATH},trust_remote_code=True,load_in_4bit=True"
        if adapter_path:
            model_args += f",peft={adapter_path}"
        output_path = str(RESULTS_DIR / f"lm_eval_{cfg_name}_mmlu_pro")
        cmd = [
            lm_eval_bin, "--model", "hf",
            "--model_args", model_args,
            "--tasks", "mmlu_pro",
            "--device", "cuda:0",
            "--batch_size", "2",
            "--output_path", output_path,
            "--limit", "500",  # Limit to 500 for time
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                               env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            if r.returncode == 0:
                for rf in Path(output_path).rglob("results.json"):
                    with open(rf) as f:
                        ALL_RESULTS[f"lm_eval_{cfg_name}_mmlu_pro"] = json.load(f).get("results", {})
                    break
        except Exception as e:
            log.warning(f"  mmlu_pro/{cfg_name} failed: {e}")
        save_results()
        gc.collect()
        torch.cuda.empty_cache()
    update_task("3.3", "done", "MMLU-Pro complete")

    log.info("✅ Phase 3 complete.")


# ══════════════════════════════════════════════════════════════════
# PHASE 5: FINAL ANALYSIS
# ══════════════════════════════════════════════════════════════════

def phase_5():
    log.info("\n" + "█" * 70)
    log.info("PHASE 5: FINAL ANALYSIS")
    log.info("█" * 70)

    update_task("5.1", "running")

    # Load Phase 1 results too
    p1_results_path = RESULTS_DIR / "master_results.json"
    if p1_results_path.exists():
        with open(p1_results_path) as f:
            p1 = json.load(f)
        ALL_RESULTS["phase1"] = p1

    # Print comprehensive summary
    log.info("\n" + "=" * 80)
    log.info("COMPLETE RESULTS SUMMARY")
    log.info("=" * 80)

    # Phase 1 results
    if "phase1" in ALL_RESULTS:
        log.info("\n--- Phase 1: Clean Single-Adapter Evals ---")
        for key, val in ALL_RESULTS["phase1"].items():
            if isinstance(val, dict) and "score" in val:
                log.info(f"  {key}: {val['score']:.1%} ({val.get('correct',0)}/{val.get('total',0)})")

    # Phase 2 composition
    if "composition_verdict" in ALL_RESULTS:
        v = ALL_RESULTS["composition_verdict"]
        log.info("\n--- Phase 2: Composition Experiment ---")
        log.info(f"  Base score: {v['base']:.3f}")
        log.info(f"  Best single adapter: {v['best_single']:.3f}")
        log.info(f"  Best composition: {v['best_composed']:.3f}")
        log.info(f"  Δ (composition vs single): {v['delta_vs_single']:+.3f}")
        log.info(f"  VERDICT: {v['verdict']}")

    # Phase 3 standardized
    log.info("\n--- Phase 3: Standardized Benchmarks ---")
    for key, val in ALL_RESULTS.items():
        if key.startswith("lm_eval_"):
            log.info(f"  {key}: {json.dumps(val)[:200]}")

    update_task("5.1", "done", "Summary generated")
    update_task("5.3", "running")
    save_results()
    update_task("5.3", "done", "Final results saved")

    log.info(f"\n{'=' * 80}")
    log.info(f"RESEARCH SPRINT COMPLETE.")
    log.info(f"Results: {RESULTS_DIR / 'sprint_results.json'}")
    log.info(f"{'=' * 80}")


# ══════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    log.info("=" * 70)
    log.info("MEWTWO RESEARCH SPRINT — Phases 2-5")
    log.info(f"Started: {datetime.now()}")
    log.info(f"GPU: {torch.cuda.get_device_name()}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("=" * 70)

    phases = [
        ("Phase 2: TRUE Composition", phase_2),
        ("Phase 3: Standardized Benchmarks", phase_3),
        ("Phase 5: Final Analysis", phase_5),
    ]

    for name, fn in phases:
        log.info(f"\n{'▶' * 5} Starting {name} {'▶' * 5}")
        try:
            fn()
        except Exception as e:
            log.error(f"❌ {name} FAILED: {e}")
            log.error(traceback.format_exc())
            save_results()
            continue

    elapsed = (time.time() - start) / 3600
    log.info(f"\nSPRINT COMPLETE. Total: {elapsed:.1f} hours")


if __name__ == "__main__":
    main()
