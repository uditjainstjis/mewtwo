#!/usr/bin/env python3
"""Run Synapta-Nemotron-30B + bfsi_extract on Synapta Indian BFSI Benchmark v1.

60 hand-curated questions across RBI/SEBI; scoring methods: substring (30) and token_f1_threshold_0.5 (30).
Generates predictions.jsonl with {benchmark_id, prediction} for use with the gated scoring.py.
Then scores both base + adapter for paired McNemar comparison.
"""
import os, sys, json, time, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1"))

import scoring as score_lib

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_PATH = str(PROJECT / "adapters" / "nemotron_30b" / "bfsi_extract" / "best")
BENCH_PATH = PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1" / "questions.jsonl"
OUT_DIR = PROJECT / "results" / "benchmark_v1_eval"
LOG_FILE = PROJECT / "logs" / "benchmark_v1_eval.log"
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


SYSTEM_MSG = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible."
)


def run_predictions(model, base_model, tok, questions, mode_label):
    log(f"=== Running predictions in mode: {mode_label} ===")
    out_file = OUT_DIR / f"predictions_{mode_label}.jsonl"
    if out_file.exists():
        out_file.unlink()
    t_start = time.time()
    rows = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            user = (f"REGULATORY CONTEXT:\n{q['context']}\n\n"
                    f"QUESTION: {q['question']}\n\nANSWER:")
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
        except Exception as e:
            log(f"  ERROR on {q['benchmark_id']}: {type(e).__name__}: {str(e)[:200]}")
            decoded = ""
        row = {"benchmark_id": q["benchmark_id"], "prediction": decoded}
        rows.append(row)
        with open(out_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (i + 1) % 5 == 0:
            eta = ((time.time() - t_start) / (i + 1)) * (len(questions) - i - 1) / 60
            log(f"  [{i+1}/{len(questions)}] {q['regulator']} t{q['tier']} ({time.time()-t0:.1f}s) ETA={eta:.1f}m")
        torch.cuda.empty_cache()
    log(f"  predictions done in {(time.time()-t_start)/60:.1f}m")
    return rows


def main():
    log("=== Synapta-Nemotron-30B benchmark_v1 eval (paired base vs adapter) ===")
    questions = [json.loads(l) for l in BENCH_PATH.open()]
    log(f"Loaded {len(questions)} questions; methods: "
        f"{sum(1 for q in questions if q['scoring_method']=='substring')} substring, "
        f"{sum(1 for q in questions if q['scoring_method']=='token_f1_threshold_0.5')} f1>=0.5")

    log("Loading tokenizer + Nemotron-30B 4-bit base...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ---- BASE pass (no adapter) ----
    base_preds = run_predictions(base_model, base_model, tok, questions, "base")

    # ---- ADAPTER pass ----
    log(f"Loading bfsi_extract adapter from {ADAPTER_PATH}")
    adapter_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="bfsi_extract", is_trainable=False)
    adapter_model.set_adapter("bfsi_extract")
    adapter_model.eval()
    log(f"Adapter wrapped. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    adapter_preds = run_predictions(adapter_model, base_model, tok, questions, "bfsi_extract")

    # ---- Score both with the gated scoring.py ----
    log("\n=== SCORING ===")
    base_scored = score_lib.evaluate(questions, base_preds)
    adapter_scored = score_lib.evaluate(questions, adapter_preds)

    # Paired McNemar — note: scoring.evaluate() returns rows[].correct
    a_vec = [r["correct"] for r in base_scored["rows"]]
    b_vec = [r["correct"] for r in adapter_scored["rows"]]
    mcnemar_p, method = score_lib.mcnemar_paired(a_vec, b_vec)

    # Save full summary
    summary = {
        "n": len(questions),
        "base": {
            "correct": base_scored["primary_correct"],
            "total": base_scored["n"],
            "rate": base_scored["primary_score"],
            "wilson_95_ci": list(base_scored["primary_95_ci"]),
            "substring_rate": base_scored["substring_rate"],
            "exact_match_rate": base_scored["exact_match_rate"],
            "token_f1_mean": base_scored["token_f1_mean"],
        },
        "bfsi_extract": {
            "correct": adapter_scored["primary_correct"],
            "total": adapter_scored["n"],
            "rate": adapter_scored["primary_score"],
            "wilson_95_ci": list(adapter_scored["primary_95_ci"]),
            "substring_rate": adapter_scored["substring_rate"],
            "exact_match_rate": adapter_scored["exact_match_rate"],
            "token_f1_mean": adapter_scored["token_f1_mean"],
        },
        "lift_pp": (adapter_scored["primary_score"] - base_scored["primary_score"]) * 100,
        "mcnemar_p": mcnemar_p,
        "mcnemar_method": method,
        "by_method": {},
        "by_regulator": {},
        "by_tier": {},
    }

    # Per-method / regulator / tier
    for q, base_r, ad_r in zip(questions, base_scored["rows"], adapter_scored["rows"]):
        for key, val in [("by_method", q["scoring_method"]), ("by_regulator", q["regulator"]), ("by_tier", q["tier"])]:
            d = summary[key].setdefault(str(val), {"n": 0, "base": 0, "adapter": 0})
            d["n"] += 1
            d["base"] += int(base_r["correct"])
            d["adapter"] += int(ad_r["correct"])

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nWrote {summary_path}")

    log(f"\n=== HEADLINE ===")
    log(f"  base:         {summary['base']['correct']}/{summary['base']['total']} = {summary['base']['rate']*100:.1f}% [{summary['base']['wilson_95_ci'][0]*100:.1f}, {summary['base']['wilson_95_ci'][1]*100:.1f}]")
    log(f"  bfsi_extract: {summary['bfsi_extract']['correct']}/{summary['bfsi_extract']['total']} = {summary['bfsi_extract']['rate']*100:.1f}% [{summary['bfsi_extract']['wilson_95_ci'][0]*100:.1f}, {summary['bfsi_extract']['wilson_95_ci'][1]*100:.1f}]")
    log(f"  Lift: +{summary['lift_pp']:.1f} pp  ·  McNemar p = {mcnemar_p:.3e}  ({method})")
    log("")
    log("By scoring method:")
    for m, d in summary["by_method"].items():
        log(f"  {m:<28} n={d['n']:<3} base={100*d['base']/d['n']:5.1f}% adapter={100*d['adapter']/d['n']:5.1f}%")
    log("By regulator:")
    for r, d in summary["by_regulator"].items():
        log(f"  {r:<6} n={d['n']:<3} base={100*d['base']/d['n']:5.1f}% adapter={100*d['adapter']/d['n']:5.1f}%")
    log("=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
