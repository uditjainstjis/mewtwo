#!/usr/bin/env python3
"""Run Synapta-Nemotron-30B + bfsi_extract on IndiaFinBench (test split, 324 Q).

Compares to Gemini 2.5 Flash 89.7% (top), Qwen3-32B 85.5%, etc.
Outputs per-task-type breakdown matching their published table.
"""
import os, sys, json, time, datetime, re, math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_PATH = str(PROJECT / "adapters" / "nemotron_30b" / "bfsi_extract" / "best")
DATA_PATH = PROJECT / "external_benchmarks" / "IndiaFinBench" / "data" / "test-00000-of-00001.parquet"
OUT_DIR = PROJECT / "results" / "indiafinbench_eval"
LOG_FILE = PROJECT / "logs" / "indiafinbench_eval.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_NEW = 200
MAX_INPUT_TOKENS = 1536


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def get_hybrid_cache(base_model, model, batch_size=1):
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, "HybridMambaAttentionDynamicCache")
    return HybridMambaAttentionDynamicCache(
        base_model.config, batch_size=batch_size,
        dtype=torch.bfloat16, device=model.device,
    )


def substring_match(gold: str, pred: str) -> int:
    return int(gold.lower().strip() in pred.lower().strip())


STOPWORDS = set("a an the of and or but in on at to from by for with as is are was were be been being have has had do does did this that these those it its their them they we i you".split())


def token_f1(gold: str, pred: str) -> float:
    g = [t for t in re.findall(r"\w+", gold.lower()) if t not in STOPWORDS]
    p = [t for t in re.findall(r"\w+", pred.lower()) if t not in STOPWORDS]
    if not g or not p:
        return 0.0
    common = set(g) & set(p)
    if not common:
        return 0.0
    prec = len(common) / len(set(p))
    rec = len(common) / len(set(g))
    return 2 * prec * rec / (prec + rec)


def normalized_match(gold: str, pred: str) -> int:
    """More forgiving: case-insensitive, punctuation-stripped, gold appears in pred."""
    g = re.sub(r'[^\w\s]', '', gold.lower()).strip()
    p = re.sub(r'[^\w\s]', '', pred.lower()).strip()
    return int(g in p) if g else 0


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    halfw = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return (mid - halfw, mid + halfw)


def main():
    log("=== Synapta-Nemotron-30B+bfsi_extract on IndiaFinBench (test split, 324 Q) ===")

    log(f"Loading benchmark from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    log(f"Loaded {len(df)} questions across task types: {df['task_type'].value_counts().to_dict()}")

    log("Loading tokenizer + Nemotron-30B 4-bit base...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    log(f"Loading bfsi_extract adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="bfsi_extract", is_trainable=False)
    model.set_adapter("bfsi_extract")
    model.eval()
    log(f"Adapter wrapped. VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    SYSTEM_MSG = (
        "You are a senior banking and financial regulation expert in India. "
        "Read the provided regulatory context carefully and answer the question "
        "precisely with the specific number, term, rule, or section citation. "
        "Quote directly from the regulation when possible."
    )

    out_file = OUT_DIR / "predictions.jsonl"
    if out_file.exists():
        out_file.unlink()

    results = []
    t_start = time.time()
    for i, row in df.iterrows():
        t0 = time.time()
        try:
            user = (f"REGULATORY CONTEXT:\n{row['context']}\n\n"
                    f"QUESTION: {row['question']}\n\nANSWER:")
            msgs = [{"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user}]
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                         max_length=MAX_INPUT_TOKENS).to(model.device)
            cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                )
            decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            gold = str(row["reference_answer"])
            row_result = {
                "id": row["id"],
                "task_type": row["task_type"],
                "difficulty": row["difficulty"],
                "source": row["source"],
                "question": row["question"],
                "gold": gold,
                "prediction": decoded,
                "substring_match": substring_match(gold, decoded),
                "normalized_match": normalized_match(gold, decoded),
                "token_f1": token_f1(gold, decoded),
                "elapsed_s": round(time.time() - t0, 2),
            }
            results.append(row_result)
            with open(out_file, "a") as f:
                f.write(json.dumps(row_result) + "\n")

            if (i + 1) % 10 == 0:
                eta_min = ((time.time() - t_start) / (i + 1)) * (len(df) - i - 1) / 60
                log(f"  [{i+1}/{len(df)}] {row['task_type'][:5]}: sub={row_result['substring_match']} norm={row_result['normalized_match']} f1={row_result['token_f1']:.2f} ({row_result['elapsed_s']:.1f}s) ETA={eta_min:.0f}m")

            torch.cuda.empty_cache()
        except Exception as e:
            log(f"  ERROR on {row['id']}: {type(e).__name__}: {str(e)[:200]}")

    log(f"\n=== EVAL COMPLETE in {(time.time() - t_start) / 60:.1f} min ===\n")

    # Compute per-task-type summary
    summary = {"overall": {}, "by_task": {}, "by_source": {}, "by_difficulty": {}}
    for metric in ["substring_match", "normalized_match", "token_f1"]:
        if metric == "token_f1":
            summary["overall"][metric] = {"mean": round(mean([r[metric] for r in results]), 4), "n": len(results)}
        else:
            k = sum(r[metric] for r in results)
            n = len(results)
            lo, hi = wilson(k, n)
            summary["overall"][metric] = {"correct": k, "total": n, "rate": round(k / n, 4), "wilson_95_lo": round(lo, 4), "wilson_95_hi": round(hi, 4)}

    for task in df['task_type'].unique():
        rows = [r for r in results if r['task_type'] == task]
        if not rows: continue
        summary["by_task"][task] = {
            "n": len(rows),
            "substring_match_rate": round(sum(r["substring_match"] for r in rows) / len(rows), 4),
            "normalized_match_rate": round(sum(r["normalized_match"] for r in rows) / len(rows), 4),
            "token_f1_mean": round(mean([r["token_f1"] for r in rows]), 4),
        }

    for src in df['source'].unique():
        rows = [r for r in results if r['source'] == src]
        if not rows: continue
        summary["by_source"][src] = {
            "n": len(rows),
            "substring_match_rate": round(sum(r["substring_match"] for r in rows) / len(rows), 4),
            "normalized_match_rate": round(sum(r["normalized_match"] for r in rows) / len(rows), 4),
        }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote summary to {summary_path}")

    # Print headline
    log("\n=== SYNAPTA on IndiaFinBench (test, n=324) ===")
    log(f"Substring match: {summary['overall']['substring_match']['rate']*100:.1f}% [{summary['overall']['substring_match']['wilson_95_lo']*100:.1f}, {summary['overall']['substring_match']['wilson_95_hi']*100:.1f}]")
    log(f"Normalized match: {summary['overall']['normalized_match']['rate']*100:.1f}% [{summary['overall']['normalized_match']['wilson_95_lo']*100:.1f}, {summary['overall']['normalized_match']['wilson_95_hi']*100:.1f}]")
    log(f"Token F1 (mean): {summary['overall']['token_f1']['mean']:.3f}")
    log("")
    log(f"{'Task':<30} {'n':<5} {'Sub %':<8} {'Norm %':<8} {'F1':<6}")
    for task, s in summary["by_task"].items():
        log(f"{task:<30} {s['n']:<5} {s['substring_match_rate']*100:<8.1f} {s['normalized_match_rate']*100:<8.1f} {s['token_f1_mean']:<6.3f}")
    log("")
    log("Compare to published top scores (their accuracy metric):")
    log("  Gemini 2.5 Flash: 89.7% overall (REG 93.1, NUM 84.8, CON 88.7, TMP 88.5)")
    log("  Qwen3-32B:        85.5% (REG 85.1, NUM 77.2, CON 90.3, TMP 92.3)")
    log("  LLaMA-3.3-70B:    83.7% (REG 86.2, NUM 75.0, CON 95.2, TMP 79.5)")


if __name__ == "__main__":
    main()
