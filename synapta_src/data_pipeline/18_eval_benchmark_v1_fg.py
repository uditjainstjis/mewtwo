#!/usr/bin/env python3
"""Format Guard eval on Synapta Indian BFSI Benchmark v1.

Loads bfsi_extract + math + code + science adapters; FG routes via BFSI-term detector
swapping every 10 tokens. Compares to bfsi_extract direct (already done by script 17).
Outputs predictions_format_guard.jsonl + appended summary keys.
"""
import os, sys, json, time, datetime, re
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          LogitsProcessor, LogitsProcessorList)
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1"))

import scoring as score_lib

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
BENCH_PATH = PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1" / "questions.jsonl"
OUT_DIR = PROJECT / "results" / "benchmark_v1_eval"
LOG_FILE = PROJECT / "logs" / "benchmark_v1_fg_eval.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_NEW = 200
MAX_INPUT_TOKENS = 1536
SWAP_INTERVAL = 10

SYSTEM_MSG = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible."
)
BFSI_TERMS = ["rbi", "sebi", "irdai", "nbfc", "kyc", "amc", "regulation", "circular",
              "master direction", "section ", "para ", "rs.", "lakh", "crore"]


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def code_fallback_router(text):
    return "math" if re.search(r'```(?:python)?|def |import |class ', text.lower()) else "code"


def bfsi_aware_router(text):
    t = text.lower()
    return "bfsi_extract" if any(term in t for term in BFSI_TERMS) else code_fallback_router(text)


class FormatGuardWithBFSI(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "bfsi_extract"
        self.swap_count = 0
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % SWAP_INTERVAL == 0:
            ctx = self.tok.decode(input_ids[0][-200:], skip_special_tokens=True)
            target = "math" if (ctx.count("```") % 2 == 1) else bfsi_aware_router(ctx)
            if target != self.current_adapter:
                try:
                    self.model.set_adapter(target)
                    self.current_adapter = target
                    self.swap_count += 1
                except Exception:
                    pass
        return scores


def get_hybrid_cache(base_model, model, batch_size=1):
    mod = sys.modules[base_model.__class__.__module__]
    Cache = getattr(mod, "HybridMambaAttentionDynamicCache")
    return Cache(base_model.config, batch_size=batch_size,
                 dtype=torch.bfloat16, device=model.device)


def main():
    log("=== Synapta benchmark_v1 Format Guard eval (n=60) ===")
    questions = [json.loads(l) for l in BENCH_PATH.open()]

    log("Loading tokenizer + Nemotron-30B 4-bit...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Load all 4 adapters
    log("Loading bfsi_extract first...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_BASE / "bfsi_extract" / "best"),
                                       adapter_name="bfsi_extract", is_trainable=False)
    for name in ["math", "code", "science"]:
        # adapters live under <name>/best (not <name>/) — same convention as bfsi_extract
        path = ADAPTER_BASE / name / "best"
        if path.exists():
            try:
                model.load_adapter(str(path), adapter_name=name, is_trainable=False)
                log(f"  loaded adapter: {name}")
            except Exception as e:
                log(f"  WARN failed to load {name}: {type(e).__name__}: {str(e)[:120]}")
        else:
            log(f"  WARN adapter path missing: {path}")
    model.eval()
    log(f"All adapters wrapped. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    out_file = OUT_DIR / "predictions_format_guard.jsonl"
    if out_file.exists():
        out_file.unlink()
    rows = []
    t_start = time.time()
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
            proc = FormatGuardWithBFSI(model, tok)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                    logits_processor=LogitsProcessorList([proc]),
                )
            decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            swaps = proc.swap_count
        except Exception as e:
            log(f"  ERROR on {q['benchmark_id']}: {type(e).__name__}: {str(e)[:200]}")
            decoded = ""
            swaps = 0
        row = {"benchmark_id": q["benchmark_id"], "prediction": decoded, "fg_swaps": swaps}
        rows.append(row)
        with open(out_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (i + 1) % 5 == 0:
            eta = ((time.time() - t_start) / (i + 1)) * (len(questions) - i - 1) / 60
            log(f"  [{i+1}/{len(questions)}] {q['regulator']} t{q['tier']} swaps={swaps} ({time.time()-t0:.1f}s) ETA={eta:.1f}m")
        torch.cuda.empty_cache()

    log(f"\n=== SCORING FG mode ===")
    fg_scored = score_lib.evaluate(questions, rows)
    fg_summary = {
        "format_guard": {
            "correct": fg_scored["primary_correct"],
            "total": fg_scored["n"],
            "rate": fg_scored["primary_score"],
            "wilson_95_ci": list(fg_scored["primary_95_ci"]),
            "substring_rate": fg_scored["substring_rate"],
            "exact_match_rate": fg_scored["exact_match_rate"],
            "token_f1_mean": fg_scored["token_f1_mean"],
            "mean_swaps_per_question": sum(r["fg_swaps"] for r in rows) / len(rows),
        }
    }

    # Load existing summary and merge
    summary_path = OUT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        existing.update(fg_summary)
        # Compute paired McNemar against bfsi_extract
        adapter_pred_path = OUT_DIR / "predictions_bfsi_extract.jsonl"
        if adapter_pred_path.exists():
            adapter_preds = [json.loads(l) for l in adapter_pred_path.open()]
            adapter_scored = score_lib.evaluate(questions, adapter_preds)
            a_vec = [r["correct"] for r in adapter_scored["rows"]]
            b_vec = [r["correct"] for r in fg_scored["rows"]]
            mc_p, mc_method = score_lib.mcnemar_paired(a_vec, b_vec)
            existing["fg_vs_adapter_mcnemar"] = {"p": mc_p, "method": mc_method,
                                                  "lift_pp": (fg_scored["primary_score"] - adapter_scored["primary_score"]) * 100}
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)
        log(f"Updated {summary_path}")

    log(f"\n=== HEADLINE (FG) ===")
    log(f"  format_guard: {fg_scored['primary_correct']}/{fg_scored['n']} = {fg_scored['primary_score']*100:.1f}% [{fg_scored['primary_95_ci'][0]*100:.1f}, {fg_scored['primary_95_ci'][1]*100:.1f}]")
    log(f"  substring rate: {fg_scored['substring_rate']*100:.1f}%, mean F1 {fg_scored['token_f1_mean']:.3f}")
    log(f"  mean adapter swaps per question: {sum(r['fg_swaps'] for r in rows) / len(rows):.1f}")
    if 'fg_vs_adapter_mcnemar' in existing:
        log(f"  vs bfsi_extract direct: lift={existing['fg_vs_adapter_mcnemar']['lift_pp']:+.1f} pp, McNemar p={existing['fg_vs_adapter_mcnemar']['p']:.3e}")
    log("=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
