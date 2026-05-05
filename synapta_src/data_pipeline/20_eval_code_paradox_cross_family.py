#!/usr/bin/env python3
"""Cross-family Code Paradox at n=200, fixing missing_artifacts.md item 4.

Goal: replicate the asymmetric positive cross-domain transfer (code-trained adapter
boosts MATH; math-trained adapter boosts HumanEval) on Qwen-3.5-0.8B and
Nemotron-Mini-4B-Instruct at n=200, in addition to the existing in-domain regression
finding (already verified at n=200 Qwen-3.5-0.8B in code_paradox_qwen_n200_summary.json).

Bench: MATH-500 (n=200, accuracy = exact-match boxed answer extraction).
Bench: HumanEval (n=164, pass@1 with v2 corrected extraction).

Adapter sources:
  - Qwen-3.5-0.8B: adapters/small_models_zoo/from_hf_kaggle/qwen3.5_0.8b/{math,code}/best/
    (or nearest existing path)
  - Nemotron-Mini-4B: adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_{math_SFT,code_SFT}_rank128/

Output: results/code_paradox_cross_family_n200.json
"""
import os, sys, json, time, datetime, re, subprocess
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
OUT_DIR = PROJECT / "results"
LOG_FILE = PROJECT / "logs" / "code_paradox_cross_family.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

N_MATH = 200
N_HUMANEVAL = 164
MAX_NEW = 512


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# Configurations to run: (config_name, base_hf_id, math_adapter_path, code_adapter_path)
# We use Qwen2.5-0.5B as a third small base since Qwen3.5-0.8B local file is incomplete.
CONFIGS = [
    {
        "name": "qwen3.5_0.8b",
        "base": "Qwen/Qwen3.5-0.8B",  # ~1.6 GB safetensors, will auto-download
        "math_path": str(PROJECT / "adapters/lori_moe/qwen3.5_0.8b/math/best"),
        "code_path": str(PROJECT / "adapters/lori_moe/qwen3.5_0.8b/code/best"),
    },
    {
        "name": "nemotron_mini_4b",
        "base": "nvidia/Nemotron-Mini-4B-Instruct",  # ~8 GB, will auto-download
        "math_path": str(PROJECT / "adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_math_SFT_rank128"),
        "code_path": str(PROJECT / "adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_code_SFT_rank128"),
    },
]


def load_math500(n):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{n}]")
    return [{"problem": r["problem"], "answer": r["answer"]} for r in ds]


def load_humaneval(n):
    ds = load_dataset("openai_humaneval", split=f"test[:{n}]")
    return [{"task_id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in ds]


def extract_boxed(s):
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*([^\n]+)", s)
    if m:
        return m.group(1).strip()
    return ""


def math_correct(pred, gold):
    p = extract_boxed(pred) or pred.strip().split("\n")[-1].strip()
    g = str(gold).strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-3
    except Exception:
        return False


def humaneval_check(prompt, completion, test, entry_point):
    """Run pytest-style check on the generated code."""
    code_block = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n?```", completion)
    body = code_block.group(1) if code_block else completion
    full = prompt + body
    if not body.strip():
        return False
    try:
        ns = {}
        exec(full, ns)
        exec(test, ns)
        ns[f"check"](ns[entry_point])
        return True
    except Exception:
        return False


def gen(model, tok, prompt_text):
    inp = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_config(cfg, math_set, humaneval_set):
    log(f"\n=== Config: {cfg['name']} ===")
    log(f"  base: {cfg['base']}")
    log(f"  math adapter: {cfg['math_path']}")
    log(f"  code adapter: {cfg['code_path']}")

    log("Loading tokenizer + base...")
    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Load adapters
    if not Path(cfg["math_path"] + "/adapter_config.json").exists():
        log(f"  WARN math adapter missing config: {cfg['math_path']}")
        return {"error": "math adapter missing"}
    if not Path(cfg["code_path"] + "/adapter_config.json").exists():
        log(f"  WARN code adapter missing config: {cfg['code_path']}")
        return {"error": "code adapter missing"}

    model = PeftModel.from_pretrained(base_model, cfg["math_path"], adapter_name="math",
                                       is_trainable=False)
    model.load_adapter(cfg["code_path"], adapter_name="code", is_trainable=False)
    model.eval()
    log(f"Adapters loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    SYSTEM_MATH = "Solve the following problem step-by-step. Put the final answer in \\boxed{}."
    SYSTEM_CODE = "Complete the following Python function. Provide ONLY the function body inside ```python``` block."

    results = {}

    # ── MATH-500 ──
    log(f"\n[{cfg['name']}] MATH-500 evaluation (n={len(math_set)})")
    for mode in ["base", "math", "code"]:
        log(f"  -- mode: {mode} --")
        n_correct = 0
        t_start = time.time()
        for i, q in enumerate(math_set):
            user = f"Problem: {q['problem']}\n\nSolution:"
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_MATH}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            try:
                if mode == "base":
                    with model.disable_adapter():
                        pred = gen(model, tok, prompt)
                else:
                    model.set_adapter(mode)
                    pred = gen(model, tok, prompt)
            except Exception as e:
                log(f"    ERROR row {i}: {type(e).__name__}: {str(e)[:120]}")
                pred = ""
            if math_correct(pred, q["answer"]):
                n_correct += 1
            if (i + 1) % 25 == 0:
                eta = ((time.time() - t_start) / (i + 1)) * (len(math_set) - i - 1) / 60
                log(f"    [{i+1}/{len(math_set)}] acc={100*n_correct/(i+1):.1f}% (ETA {eta:.1f}m)")
            torch.cuda.empty_cache()
        results[f"math500_{mode}"] = {"correct": n_correct, "total": len(math_set),
                                       "acc": round(n_correct / len(math_set), 4)}
        log(f"  RESULT: math500/{mode} = {n_correct}/{len(math_set)} = {100*n_correct/len(math_set):.1f}%")

    # ── HumanEval ──
    log(f"\n[{cfg['name']}] HumanEval (n={len(humaneval_set)})")
    for mode in ["base", "math", "code"]:
        log(f"  -- mode: {mode} --")
        n_correct = 0
        t_start = time.time()
        for i, q in enumerate(humaneval_set):
            user = f"{q['prompt']}"
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_CODE}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            try:
                if mode == "base":
                    with model.disable_adapter():
                        pred = gen(model, tok, prompt)
                else:
                    model.set_adapter(mode)
                    pred = gen(model, tok, prompt)
            except Exception as e:
                log(f"    ERROR row {i}: {type(e).__name__}: {str(e)[:120]}")
                pred = ""
            if humaneval_check(q["prompt"], pred, q["test"], q["entry_point"]):
                n_correct += 1
            if (i + 1) % 25 == 0:
                eta = ((time.time() - t_start) / (i + 1)) * (len(humaneval_set) - i - 1) / 60
                log(f"    [{i+1}/{len(humaneval_set)}] pass@1={100*n_correct/(i+1):.1f}% (ETA {eta:.1f}m)")
            torch.cuda.empty_cache()
        results[f"humaneval_{mode}"] = {"correct": n_correct, "total": len(humaneval_set),
                                         "pass_at_1": round(n_correct / len(humaneval_set), 4)}
        log(f"  RESULT: humaneval/{mode} = {n_correct}/{len(humaneval_set)} = {100*n_correct/len(humaneval_set):.1f}%")

    # Free for next config
    del model, base_model
    torch.cuda.empty_cache()
    return results


def main():
    log("=== Code Paradox cross-family n=200 eval ===")
    log(f"Configs to run: {[c['name'] for c in CONFIGS]}")

    log("Loading benchmarks...")
    math_set = load_math500(N_MATH)
    humaneval_set = load_humaneval(N_HUMANEVAL)
    log(f"  MATH-500: {len(math_set)}; HumanEval: {len(humaneval_set)}")

    all_results = {}
    for cfg in CONFIGS:
        all_results[cfg["name"]] = run_config(cfg, math_set, humaneval_set)
        with open(OUT_DIR / "code_paradox_cross_family_n200.json", "w") as f:
            json.dump(all_results, f, indent=2)

    log("\n=== FINAL ===")
    log(json.dumps(all_results, indent=2))
    log("=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
