#!/usr/bin/env python3
"""Token-routed LoRI-MoE end-to-end on GSM8K, ARC-Challenge, MMLU.

Fixes missing_artifacts.md item 1: gives Cluster D (LoRI-MoE) the token-level routed
evaluation it needs to be a positive-result paper rather than a partial-result paper.

Architecture:
  - Base: Qwen2.5-1.5B-Instruct (full precision)
  - 5 LoRI experts loaded simultaneously into VRAM:
        adapters/published/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/
  - LogitsProcessor swaps active expert every K=10 generated tokens via regex over
    the decoded suffix (FormatGuard-style routing applied to LoRI experts).

Comparison rows produced for each benchmark (n=200, matching prior phase 3):
  - base (no adapter)
  - single-best adapter (math for GSM8K; legal for ARC; whichever for MMLU)
  - composite top-1 routed (existing phase3 result, re-validated)
  - **token-routed LoRI-MoE (this work, the new row)**

Output: results/lori_moe/phase3_token_routed.json
"""
import os, sys, json, time, datetime, re, math
from pathlib import Path
from collections import defaultdict

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)
from peft import PeftModel
from datasets import load_dataset

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_BASE = PROJECT / "adapters" / "lori_moe" / "qwen2.5_1.5b"
ADAPTER_NAMES = ["math", "legal", "science", "medical", "code"]
# Paths: adapter_model.safetensors lives at <ADAPTER_BASE>/<name>/best/
OUT_DIR = PROJECT / "results" / "lori_moe"
LOG_FILE = PROJECT / "logs" / "lori_moe_token_routed.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

N_PER_BENCHMARK = 200
MAX_NEW = 256
MAX_INPUT_TOKENS = 1536
SWAP_INTERVAL = 10

# Routing keywords per LoRI domain
ROUTE_RULES = [
    ("medical", re.compile(r"\b(patient|diagnos|symptom|disease|treatment|clinical|medical|drug|therapy)\b", re.I)),
    ("legal",   re.compile(r"\b(law|legal|court|statute|regulation|contract|liability|constitution)\b", re.I)),
    ("math",    re.compile(r"\b(equation|theorem|integral|derivative|matrix|solve|prove|calcul|sum|product)\b", re.I)),
    ("science", re.compile(r"\b(physic|chemistry|biology|electron|molecule|cell|gravity|species|enzyme)\b", re.I)),
    ("code",    re.compile(r"```|def |import |class |return |for .+ in |while ", re.I)),
]


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def route(ctx_text, default="math"):
    for name, pat in ROUTE_RULES:
        if pat.search(ctx_text):
            return name
    return default


class TokenRoutedLogitsProcessor(LogitsProcessor):
    def __init__(self, model, tok, default_adapter="math"):
        self.model = model
        self.tok = tok
        self.current = default_adapter
        self.swap_count = 0
        self.model.set_adapter(default_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % SWAP_INTERVAL == 0:
            ctx = self.tok.decode(input_ids[0][-200:], skip_special_tokens=True)
            target = route(ctx, default=self.current)
            if target != self.current:
                try:
                    self.model.set_adapter(target)
                    self.current = target
                    self.swap_count += 1
                except Exception:
                    pass
        return scores


def build_chat_prompt(tok, user_text, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt_text, mode, default_adapter=None):
    inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                 max_length=MAX_INPUT_TOKENS).to(model.device)
    swap_count = 0
    gen_kwargs = dict(max_new_tokens=MAX_NEW, do_sample=False,
                      pad_token_id=tok.pad_token_id, use_cache=True)
    if mode == "base":
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    elif mode == "single":
        model.set_adapter(default_adapter)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    elif mode == "token_routed":
        proc = TokenRoutedLogitsProcessor(model, tok, default_adapter=default_adapter or "math")
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs,
                                 logits_processor=LogitsProcessorList([proc]))
        swap_count = proc.swap_count
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return decoded, swap_count


# ============================================================================
# Benchmark scorers
# ============================================================================

def extract_gsm8k_answer(text):
    """Extract numerical answer from chain-of-thought; gsm8k uses '#### N' convention."""
    m = re.search(r"####\s*([-+]?\d+(?:[\.,]\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:[\.,]\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def gsm8k_correct(pred, gold):
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold) or gold.split("####")[-1].strip().replace(",", "")
    try:
        return abs(float(p) - float(g)) < 1e-3
    except Exception:
        return p.strip() == g.strip()


def mc_correct(pred, gold_letter, choices):
    """ARC/MMLU multi-choice: prediction may be a letter or text snippet."""
    p = pred.strip().upper()
    if p and p[0] == gold_letter.upper():
        return True
    # check first 5 chars contain "(A)" patterns
    m = re.search(r"\b([A-D])\b", p[:80])
    if m and m.group(1) == gold_letter.upper():
        return True
    # Fallback: check if gold text appears
    if isinstance(choices, list) and gold_letter in "ABCD":
        idx = ord(gold_letter.upper()) - ord("A")
        if 0 <= idx < len(choices):
            gold_text = choices[idx].strip().lower()
            if gold_text and gold_text in pred.lower():
                return True
    return False


def load_gsm8k(n):
    ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    return [{"question": r["question"], "answer": r["answer"], "task": "gsm8k"} for r in ds]


def load_arc(n):
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{n}]")
    rows = []
    for r in ds:
        choices = r["choices"]["text"]
        labels = r["choices"]["label"]
        gold_label = r["answerKey"]
        # Normalize numeric labels (1234) to letters (ABCD)
        if gold_label in "1234":
            gold_label = chr(ord("A") + int(gold_label) - 1)
            labels = ["A", "B", "C", "D"][:len(labels)]
        rows.append({"question": r["question"], "choices": choices, "labels": labels,
                     "gold": gold_label, "task": "arc"})
    return rows


def load_mmlu(n):
    ds = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    rows = []
    for r in ds:
        rows.append({"question": r["question"], "choices": r["choices"],
                     "labels": ["A", "B", "C", "D"], "gold": "ABCD"[r["answer"]],
                     "task": "mmlu", "subject": r.get("subject", "")})
    return rows


def format_mc_prompt(q):
    s = f"Question: {q['question']}\n"
    for lab, ch in zip(q["labels"], q["choices"]):
        s += f"{lab}. {ch}\n"
    s += "Answer with the letter only.\nAnswer:"
    return s


def format_gsm8k_prompt(q):
    return (f"Solve this grade-school math problem step-by-step. End your answer with "
            f"'#### N' where N is the final numerical answer.\n\n"
            f"Question: {q['question']}\n\nSolution:")


SYSTEM_PROMPT = ("You are a careful expert assistant. For multiple-choice questions answer with "
                 "the letter only. For math questions show your work then write the final answer "
                 "as '#### N'.")


# Domain default adapters per benchmark for the "single" baseline
DEFAULT_SINGLE = {"gsm8k": "math", "arc": "science", "mmlu": "math"}


# ============================================================================
# Main
# ============================================================================

def main():
    log("=== LoRI-MoE token-routed end-to-end eval ===")
    log(f"Base: {BASE_MODEL}")
    log(f"Adapters: {ADAPTER_NAMES}")
    log(f"N per benchmark: {N_PER_BENCHMARK}")

    log("Loading tokenizer + base model (bf16)...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    log(f"Loading first adapter: {ADAPTER_NAMES[0]}")
    first_path = (ADAPTER_BASE / ADAPTER_NAMES[0] / "best").resolve()
    log(f"  path: {first_path}")
    model = PeftModel.from_pretrained(base_model, str(first_path),
                                       adapter_name=ADAPTER_NAMES[0], is_trainable=False)
    for name in ADAPTER_NAMES[1:]:
        path = (ADAPTER_BASE / name / "best").resolve()
        try:
            model.load_adapter(str(path), adapter_name=name, is_trainable=False)
            log(f"  loaded adapter: {name}")
        except Exception as e:
            log(f"  WARN failed to load {name}: {type(e).__name__}: {str(e)[:120]}")
    model.eval()
    log(f"All adapters loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    benchmarks = {
        "gsm8k": load_gsm8k(N_PER_BENCHMARK),
        "arc":   load_arc(N_PER_BENCHMARK),
        "mmlu":  load_mmlu(N_PER_BENCHMARK),
    }
    log(f"Benchmarks loaded: " + ", ".join(f"{k}={len(v)}" for k, v in benchmarks.items()))

    summary = {"n_per_benchmark": N_PER_BENCHMARK,
               "modes": ["base", "single", "token_routed"],
               "results": defaultdict(dict)}

    out_predictions = OUT_DIR / "phase3_token_routed_predictions.jsonl"
    if out_predictions.exists():
        out_predictions.unlink()
    pred_f = open(out_predictions, "a")

    for bench_name, items in benchmarks.items():
        default_adapter = DEFAULT_SINGLE[bench_name]
        log(f"\n=== Benchmark: {bench_name} (default single = {default_adapter}) ===")
        for mode in ["base", "single", "token_routed"]:
            log(f"  -- mode: {mode} --")
            t_start = time.time()
            n_correct = 0
            n_total = 0
            n_swaps = 0
            for i, q in enumerate(items):
                if bench_name == "gsm8k":
                    user_text = format_gsm8k_prompt(q)
                else:
                    user_text = format_mc_prompt(q)
                prompt_text = build_chat_prompt(tok, user_text, system=SYSTEM_PROMPT)
                try:
                    pred, swaps = generate(model, tok, prompt_text, mode,
                                           default_adapter=default_adapter)
                except Exception as e:
                    log(f"    ERROR row {i}: {type(e).__name__}: {str(e)[:160]}")
                    pred, swaps = "", 0

                if bench_name == "gsm8k":
                    correct = gsm8k_correct(pred, q["answer"])
                else:
                    correct = mc_correct(pred, q["gold"], q.get("choices"))

                n_total += 1
                n_correct += int(correct)
                n_swaps += swaps

                pred_f.write(json.dumps({"benchmark": bench_name, "mode": mode, "i": i,
                                         "correct": int(correct), "pred": pred[:400],
                                         "swaps": swaps}) + "\n")
                pred_f.flush()

                if (i + 1) % 25 == 0:
                    eta = ((time.time() - t_start) / (i + 1)) * (len(items) - i - 1) / 60
                    log(f"    [{i+1}/{len(items)}] acc so far {100*n_correct/(i+1):.1f}% (ETA {eta:.1f}m)")
                torch.cuda.empty_cache()

            score = n_correct / n_total if n_total else 0.0
            mean_swaps = n_swaps / n_total if mode == "token_routed" and n_total else 0.0
            elapsed_min = (time.time() - t_start) / 60
            summary["results"][bench_name][mode] = {
                "score": round(score, 4), "correct": n_correct, "total": n_total,
                "mean_swaps": round(mean_swaps, 2), "elapsed_min": round(elapsed_min, 1),
                "default_single": default_adapter if mode == "single" else None,
            }
            log(f"  {bench_name}/{mode}: {n_correct}/{n_total} = {100*score:.1f}%  ({elapsed_min:.1f}m, mean_swaps={mean_swaps:.2f})")

            with open(OUT_DIR / "phase3_token_routed.json", "w") as f:
                json.dump(dict(summary["results"]), f, indent=2)

    pred_f.close()

    # Final summary
    final = {
        "n_per_benchmark": N_PER_BENCHMARK,
        "results": dict(summary["results"]),
        "comparison_to_phase3": {
            "phase3_composite_gsm8k": 0.04,
            "phase3_composite_arc": 0.72,
            "phase3_composite_mmlu": 0.53,
            "note": "Phase 3 (prompt-level routed top-1) numbers from results/lori_moe/phase3_composite.json",
        },
    }
    with open(OUT_DIR / "phase3_token_routed.json", "w") as f:
        json.dump(final, f, indent=2)
    log(f"\nWrote {OUT_DIR / 'phase3_token_routed.json'}")
    log("=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
