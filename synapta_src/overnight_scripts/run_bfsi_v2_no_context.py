#!/usr/bin/env python3
"""BFSI v2 — No-context recall benchmark for Nemotron-30B.

Tests whether the model has RBI/SEBI/IRDAI knowledge from pretraining.
Unlike v1 (context-injected, both modes hit 100%), v2 asks the model to
recall the rule. This SHOULD differentiate base vs Format Guard.

Run AFTER the main swarm finishes (releases Nemotron from MBPP).
Output: results/bfsi_swarm_extras/bfsi_v2_results.jsonl
        results/bfsi_swarm_extras/bfsi_v2_summary.json
"""
import os, sys, json, re, time, datetime
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data" / "rbi_circulars"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

from questions_v2_no_context import QUESTIONS_V2

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
OUT_DIR = PROJECT / "results" / "bfsi_swarm_extras"
LOG_FILE = PROJECT / "logs" / "swarm_8h" / "extras" / "bfsi_v2.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_NEW = 200


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def heuristic_router(decoded_text):
    text = decoded_text.lower()
    if re.search(r'```(?:python)?|def |import |class |    \w+', text):
        return "math"
    if re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', text):
        return "code"
    return "code"


class FormatGuardProcessor(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.swap_count = 0
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-200:], skip_special_tokens=True)
            ticks = ctx.count("```")
            target = "math" if (ticks % 2 == 1) else heuristic_router(ctx)
            if target != self.current_adapter:
                self.model.set_adapter(target)
                self.current_adapter = target
                self.swap_count += 1
        return scores


def get_hybrid_cache(base_model, model, batch_size=1):
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, "HybridMambaAttentionDynamicCache")
    return HybridMambaAttentionDynamicCache(
        base_model.config, batch_size=batch_size,
        dtype=torch.bfloat16, device=model.device,
    )


def generate(model, base_model, tok, prompt_text, mode):
    inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])
    swap_count = 0
    if mode == "base":
        with model.disable_adapter():
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache)
    else:
        proc = FormatGuardProcessor(model, tok)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                                 logits_processor=LogitsProcessorList([proc]))
        swap_count = proc.swap_count
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded, swap_count


def main():
    log(f"=== BFSI v2 (no-context recall) ===")
    log(f"Loading Nemotron-30B + adapters...")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()

    def adapter_path(name):
        b = ADAPTER_BASE / name / "best"
        f = ADAPTER_BASE / name / "final"
        return str(b) if b.exists() else str(f)

    model = PeftModel.from_pretrained(base_model, adapter_path("math"), adapter_name="math", is_trainable=False)
    model.load_adapter(adapter_path("code"), adapter_name="code")
    model.load_adapter(adapter_path("science"), adapter_name="science")
    model.eval()
    log(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    sys_msg = "You are a senior banking and financial regulation expert in India. Answer each question precisely with the specific number, term, or rule. Be concise."

    out_file = OUT_DIR / "bfsi_v2_results.jsonl"
    if out_file.exists():
        out_file.unlink()

    results = {"base": [], "format_guard": []}
    for mode in ("base", "format_guard"):
        log(f"\n--- Mode: {mode} ---")
        for i, q in enumerate(QUESTIONS_V2):
            t0 = time.time()
            try:
                user_prompt = f"QUESTION: {q['question']}\n\nANSWER:"
                msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_prompt}]
                prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                output, swap_count = generate(model, base_model, tok, prompt_text, mode)

                output_lower = output.lower()
                gold = q["gold_answer"].lower()
                alts = [a.lower() for a in q.get("alternatives", [])]
                scoring = q.get("scoring", "contains")
                if scoring == "multi_term":
                    matches = sum(1 for a in alts if a in output_lower)
                    passed = matches >= q.get("must_match_count", 2)
                else:
                    passed = (gold in output_lower) or any(a in output_lower for a in alts)

                row = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "id": q["id"], "source": q["source"], "mode": mode,
                    "question": q["question"], "gold": q["gold_answer"],
                    "raw_output": output, "passed": passed,
                    "swap_count": swap_count, "elapsed_s": round(time.time() - t0, 2),
                }
                results[mode].append(row)
                with open(out_file, "a") as f:
                    f.write(json.dumps(row) + "\n")
                log(f"  [{mode}] {i+1}/{len(QUESTIONS_V2)} {q['id']}: passed={passed}")
            except Exception as e:
                log(f"  [{mode}] {q['id']} EXC: {e}")
                results[mode].append({"id": q["id"], "mode": mode, "error": str(e)[:200]})
            torch.cuda.empty_cache()

    summary = {}
    for mode, rows in results.items():
        passed = sum(1 for r in rows if r.get("passed"))
        n = len(rows)
        summary[mode] = {"passed": passed, "total": n, "rate": passed / max(n, 1)}
    with open(OUT_DIR / "bfsi_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n=== SUMMARY ===")
    log(f"base: {summary['base']['passed']}/{summary['base']['total']} = {summary['base']['rate']:.1%}")
    log(f"format_guard: {summary['format_guard']['passed']}/{summary['format_guard']['total']} = {summary['format_guard']['rate']:.1%}")
    log(f"DELTA: {(summary['format_guard']['rate'] - summary['base']['rate'])*100:+.1f} pp")


if __name__ == "__main__":
    main()
