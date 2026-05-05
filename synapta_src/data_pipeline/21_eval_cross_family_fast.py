#!/usr/bin/env python3
"""Streamlined fast eval: takes config name + benchmark as CLI args.

Improvements over 20_*:
  - Per-(config, benchmark, mode) JSON checkpoint after EACH mode finishes
  - Lower max_new_tokens (256 for HumanEval, 384 for MATH-500)
  - max_input_tokens reduced to 768 (HumanEval prompts are short, stops OOM-by-context)
  - Skip-if-done: looks at existing checkpoint and skips already-completed cells
  - Single config per process so two processes can run in parallel on RTX 5090

Usage:
    python 21_eval_cross_family_fast.py --config qwen3.5_0.8b --benchmark humaneval
    python 21_eval_cross_family_fast.py --config nemotron_mini_4b --benchmark math500
"""
import argparse, os, sys, json, time, datetime, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

PROJECT = Path("/home/learner/Desktop/mewtwo")
RESULTS_FILE = PROJECT / "results" / "code_paradox_cross_family_n200.json"
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "qwen3.5_0.8b": {
        "base": "Qwen/Qwen3.5-0.8B",
        "math_path": str(PROJECT / "adapters/lori_moe/qwen3.5_0.8b/math/best"),
        "code_path": str(PROJECT / "adapters/lori_moe/qwen3.5_0.8b/code/best"),
    },
    "nemotron_mini_4b": {
        "base": "nvidia/Nemotron-Mini-4B-Instruct",
        "math_path": str(PROJECT / "adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_math_SFT_rank128"),
        "code_path": str(PROJECT / "adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_code_SFT_rank128"),
    },
}

N_MATH = 200
N_HUMANEVAL = 164


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_results():
    if RESULTS_FILE.exists():
        try:
            return json.load(open(RESULTS_FILE))
        except Exception:
            return {}
    return {}


def save_results(d):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(d, f, indent=2)


def extract_boxed(s):
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m: return m.group(1).strip()
    m = re.search(r"####\s*([^\n]+)", s)
    if m: return m.group(1).strip()
    return ""


def math_correct(pred, gold):
    p = extract_boxed(pred) or pred.strip().split("\n")[-1].strip()
    g = str(gold).strip()
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-3
    except Exception: return False


def humaneval_check(prompt, completion, test, entry_point):
    code_block = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n?```", completion)
    body = code_block.group(1) if code_block else completion
    full = prompt + body
    if not body.strip(): return False
    try:
        ns = {}
        exec(full, ns)
        exec(test, ns)
        ns["check"](ns[entry_point])
        return True
    except Exception:
        return False


def load_math500(n):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{n}]")
    return [{"problem": r["problem"], "answer": r["answer"]} for r in ds]


def load_humaneval(n):
    ds = load_dataset("openai_humaneval", split=f"test[:{n}]")
    return [{"task_id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in ds]


def gen(model, tok, prompt_text, max_new):
    inp = tok(prompt_text, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def gen_batch(model, tok, prompt_texts, max_new):
    """Batched generation with left-padding. Greedy (do_sample=False) is deterministic
    regardless of batch size, and left-padding is the standard for causal LMs to ensure
    output equivalence to single-prompt calls."""
    saved_side = tok.padding_side
    tok.padding_side = "left"
    inp = tok(prompt_texts, return_tensors="pt", truncation=True, max_length=768,
              padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
    tok.padding_side = saved_side
    L = inp["input_ids"].shape[1]
    return [tok.decode(out[i][L:], skip_special_tokens=True).strip() for i in range(out.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--benchmark", required=True, choices=["math500", "humaneval"])
    ap.add_argument("--modes", nargs="+", default=["base", "math", "code"])
    ap.add_argument("--batch", type=int, default=8, help="batch size; greedy decoding makes outputs identical to bs=1 with left-padding")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    log_path = LOG_DIR / f"cross_family_{args.config}_{args.benchmark}.log"
    def log(msg):
        line = f"[{now_str()}] {msg}"
        with open(log_path, "a") as f: f.write(line + "\n")
        print(line, flush=True)

    log(f"=== Cross-family {args.config} / {args.benchmark} ===")

    # Resume check
    all_results = load_results()
    cfg_results = all_results.get(args.config, {})
    modes_to_run = []
    for mode in args.modes:
        key = f"{args.benchmark}_{mode}"
        if key in cfg_results:
            log(f"  SKIP {key} (already done: {cfg_results[key]})")
        else:
            modes_to_run.append(mode)
    if not modes_to_run:
        log("All modes already done. Exiting.")
        return

    log(f"Modes to run: {modes_to_run}")
    log(f"Loading benchmark...")
    if args.benchmark == "math500":
        items = load_math500(N_MATH)
        max_new = 384
        sys_msg = "Solve the following problem step-by-step. Put the final answer in \\boxed{}."
    else:
        items = load_humaneval(N_HUMANEVAL)
        max_new = 256
        sys_msg = "Complete the Python function. Provide ONLY the function body inside ```python``` block."
    log(f"  Loaded {len(items)}")

    log(f"Loading tokenizer + base: {cfg['base']}")
    tok = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    model = PeftModel.from_pretrained(base_model, cfg["math_path"], adapter_name="math",
                                       is_trainable=False)
    model.load_adapter(cfg["code_path"], adapter_name="code", is_trainable=False)
    model.eval()
    log(f"Adapters loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    BATCH = args.batch
    for mode in modes_to_run:
        log(f"\n=== mode: {mode} (batch={BATCH}) ===")

        # Build all prompts up front
        prompt_texts = []
        for q in items:
            if args.benchmark == "math500":
                user = f"Problem: {q['problem']}\n\nSolution:"
            else:
                user = q["prompt"]
            try:
                pt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                pt = tok.apply_chat_template(
                    [{"role": "user", "content": sys_msg + "\n\n" + user}],
                    tokenize=False, add_generation_prompt=True)
            prompt_texts.append(pt)

        # Set adapter once for the full pass (avoid per-row swap overhead)
        if mode != "base":
            model.set_adapter(mode)

        n_correct = 0
        t_start = time.time()
        i = 0
        while i < len(items):
            chunk = prompt_texts[i:i + BATCH]
            try:
                if mode == "base":
                    with model.disable_adapter():
                        preds = gen_batch(model, tok, chunk, max_new)
                else:
                    preds = gen_batch(model, tok, chunk, max_new)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  OOM on batch starting row {i}; falling back to bs=1 for this chunk")
                preds = []
                for pt in chunk:
                    try:
                        if mode == "base":
                            with model.disable_adapter():
                                preds.append(gen(model, tok, pt, max_new))
                        else:
                            preds.append(gen(model, tok, pt, max_new))
                    except Exception as e:
                        log(f"    sub-error: {type(e).__name__}")
                        preds.append("")
            except Exception as e:
                log(f"  ERROR batch row {i}: {type(e).__name__}: {str(e)[:120]}")
                preds = ["" for _ in chunk]

            for j, pred in enumerate(preds):
                q = items[i + j]
                if args.benchmark == "math500":
                    if math_correct(pred, q["answer"]):
                        n_correct += 1
                else:
                    if humaneval_check(q["prompt"], pred, q["test"], q["entry_point"]):
                        n_correct += 1

            i += BATCH
            done = min(i, len(items))
            if done % max(BATCH, 25) == 0 or done >= len(items):
                eta = ((time.time() - t_start) / done) * (len(items) - done) / 60
                log(f"    [{done}/{len(items)}] acc={100*n_correct/done:.1f}% (ETA {eta:.1f}m)")
            torch.cuda.empty_cache()

        score = round(n_correct / len(items), 4)
        elapsed_min = (time.time() - t_start) / 60
        log(f"  RESULT: {args.benchmark}/{mode} = {n_correct}/{len(items)} = {100*score:.1f}% ({elapsed_min:.1f}m)")

        # Checkpoint immediately after each mode
        all_results = load_results()
        if args.config not in all_results:
            all_results[args.config] = {}
        all_results[args.config][f"{args.benchmark}_{mode}"] = {
            "correct": n_correct, "total": len(items), "acc": score,
            "elapsed_min": round(elapsed_min, 1),
        }
        save_results(all_results)
        log(f"  Saved checkpoint to {RESULTS_FILE}")

    log(f"\n=== {args.config}/{args.benchmark} DONE ===")


if __name__ == "__main__":
    main()
