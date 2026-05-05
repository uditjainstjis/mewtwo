#!/usr/bin/env python3
"""Held-out eval of the BFSI extract LoRA on Nemotron-30B (700 QAs, 26 unseen PDFs).
Modes: base, bfsi_extract_only, format_guard_with_bfsi (FG routes to bfsi_extract on
BFSI terms, code-router fallback, swap every 10 tokens).
Scoring: substring (case-insensitive), exact match, SQuAD token F1.
Stats: Wilson 95% CI + McNemar (statsmodels, scipy fallback).
Outputs: results/bfsi_eval/eval_results.jsonl + summary.json. Resumes; ~3h ETA."""
import sys, json, re, time, math, datetime
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

MODEL_PATH    = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE  = PROJECT / "adapters" / "nemotron_30b"
EVAL_CLEAN    = PROJECT / "data" / "rbi_corpus" / "qa" / "eval_clean.jsonl"
EVAL_FALLBACK = PROJECT / "data" / "rbi_corpus" / "qa" / "eval.jsonl"
OUT_DIR       = PROJECT / "results" / "bfsi_eval"
LOG_FILE      = PROJECT / "logs" / "bfsi_eval" / "eval.log"
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
STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
             "was", "were", "be", "been", "by", "with", "as", "at", "from", "that",
             "this", "it", "its"}


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
                self.model.set_adapter(target)
                self.current_adapter = target
                self.swap_count += 1
        return scores


def get_hybrid_cache(base_model, model, batch_size=1):
    mod = sys.modules[base_model.__class__.__module__]
    Cache = getattr(mod, "HybridMambaAttentionDynamicCache")
    return Cache(base_model.config, batch_size=batch_size,
                 dtype=torch.bfloat16, device=model.device)


def generate(model, base_model, tok, prompt_text, mode):
    inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                 max_length=MAX_INPUT_TOKENS).to(model.device)
    cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])
    swap_count = 0
    gen_kwargs = dict(max_new_tokens=MAX_NEW, do_sample=False,
                      pad_token_id=tok.pad_token_id, use_cache=True,
                      past_key_values=cache)
    if mode == "base":
        with model.disable_adapter(), torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    elif mode == "bfsi_extract_only":
        model.set_adapter("bfsi_extract")
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
    elif mode == "format_guard_with_bfsi":
        proc = FormatGuardWithBFSI(model, tok)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs,
                                 logits_processor=LogitsProcessorList([proc]))
        swap_count = proc.swap_count
    else:
        raise ValueError(f"Unknown mode: {mode}")
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip(), swap_count


def normalise_tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def substring_match(gold, pred):
    return 1 if gold.strip().lower() in pred.lower() else 0


def exact_match(gold, pred):
    return 1 if gold.strip().lower() == pred.strip().lower() else 0


def token_f1(gold, pred):
    g, p = normalise_tokens(gold), normalise_tokens(pred)
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g or not p:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(g) & Counter(p)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return prec, rec, 2 * prec * rec / (prec + rec)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_test(a_correct, b_correct):
    """Paired McNemar's test on per-item 0/1 outcomes.

    a is the 'control' baseline, b is the 'treatment'.
    b01 = a wrong & b right; b10 = a right & b wrong.
    Prefers statsmodels; falls back to scipy.binomtest.
    """
    assert len(a_correct) == len(b_correct)
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[0, b01], [b10, 0]], exact=(b01 + b10) < 25, correction=True)
        return {"b01": b01, "b10": b10, "statistic": float(res.statistic),
                "p_value": float(res.pvalue), "engine": "statsmodels"}
    except Exception:
        try:
            from scipy.stats import binomtest
            n = b01 + b10
            if n == 0:
                return {"b01": 0, "b10": 0, "statistic": None, "p_value": 1.0,
                        "engine": "scipy_binomtest"}
            p = binomtest(min(b01, b10), n, p=0.5).pvalue
            return {"b01": b01, "b10": b10, "statistic": None,
                    "p_value": float(p), "engine": "scipy_binomtest"}
        except Exception as e:
            return {"b01": b01, "b10": b10, "statistic": None,
                    "p_value": None, "engine": f"unavailable:{e}"}


def load_eval():
    src = EVAL_CLEAN if EVAL_CLEAN.exists() else EVAL_FALLBACK
    if not src.exists():
        raise FileNotFoundError(f"Neither {EVAL_CLEAN} nor {EVAL_FALLBACK} exists")
    rows = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log(f"Loaded {len(rows)} eval rows from {src.name}")
    return rows, src.name


def adapter_path(name):
    b = ADAPTER_BASE / name / "best"
    f = ADAPTER_BASE / name / "final"
    return str(b) if b.exists() else str(f)


def adapter_exists(name):
    return (ADAPTER_BASE / name / "best").exists() or (ADAPTER_BASE / name / "final").exists()


def aggregate(out_file, eval_rows, eval_name):
    by_mode = defaultdict(list)
    with open(out_file) as f:
        for line in f:
            try:
                r = json.loads(line)
                by_mode[r["mode"]].append(r)
            except Exception:
                pass

    summary = {"eval_source": eval_name, "n_questions": len(eval_rows),
               "modes": {}, "mcnemar": {}, "by_tier": {}, "by_regulator": {}}

    for mode, rows in by_mode.items():
        n = len(rows)
        sub_k = sum(r["substring_match"] for r in rows)
        em_k = sum(r["exact_match"] for r in rows)
        f1_mean = sum(r["token_f1"] for r in rows) / max(n, 1)
        ci_lo, ci_hi = wilson_ci(sub_k, n)
        summary["modes"][mode] = {
            "n": n, "substring_match_rate": sub_k / max(n, 1),
            "substring_match_ci95": [ci_lo, ci_hi],
            "exact_match_rate": em_k / max(n, 1),
            "token_f1_mean": f1_mean,
        }
        tier_b, reg_b = defaultdict(list), defaultdict(list)
        for r in rows:
            tier_b[str(r.get("tier"))].append(r["substring_match"])
            reg_b[str(r.get("regulator"))].append(r["substring_match"])
        summary["by_tier"][mode] = {t: {"n": len(v), "rate": sum(v) / max(len(v), 1)}
                                    for t, v in tier_b.items()}
        summary["by_regulator"][mode] = {r: {"n": len(v), "rate": sum(v) / max(len(v), 1)}
                                         for r, v in reg_b.items()}

    def aligned(a, b):
        if a not in by_mode or b not in by_mode:
            return None, None
        am = {r["qa_id"]: r["substring_match"] for r in by_mode[a]}
        bm = {r["qa_id"]: r["substring_match"] for r in by_mode[b]}
        common = sorted(set(am) & set(bm))
        return [am[k] for k in common], [bm[k] for k in common]

    pairs = [("bfsi_extract_only", "base"),
             ("format_guard_with_bfsi", "base"),
             ("format_guard_with_bfsi", "bfsi_extract_only")]
    for a, b in pairs:
        av, bv = aligned(a, b)
        if av is None or len(av) == 0:
            summary["mcnemar"][f"{a}_vs_{b}"] = {"skipped": True}
            continue
        # control = b, treatment = a
        summary["mcnemar"][f"{a}_vs_{b}"] = {"n_paired": len(av), **mcnemar_test(bv, av)}
    return summary


def print_report(summary):
    bar = "=" * 78
    print(f"\n{bar}\nBFSI EXTRACT HELD-OUT EVAL SUMMARY\n{bar}")
    print(f"Eval source: {summary['eval_source']}  n={summary['n_questions']}\n")
    hdr = f"{'Mode':<28} {'N':>5} {'Sub%':>7} {'CI95':>16} {'EM%':>6} {'F1':>6}"
    print(hdr); print("-" * len(hdr))
    for mode, s in summary["modes"].items():
        ci = s["substring_match_ci95"]
        print(f"{mode:<28} {s['n']:>5} {s['substring_match_rate']*100:>6.1f}% "
              f"[{ci[0]*100:>4.1f},{ci[1]*100:>4.1f}] "
              f"{s['exact_match_rate']*100:>5.1f}% {s['token_f1_mean']*100:>5.1f}")
    print("\nMcNemar paired tests (substring match):")
    for k, v in summary["mcnemar"].items():
        if v.get("skipped"):
            print(f"  {k}: skipped")
        else:
            print(f"  {k}: n={v['n_paired']} b01={v.get('b01')} b10={v.get('b10')} "
                  f"p={v.get('p_value')}")
    print("\nPer-tier substring-match rate:")
    for mode, tiers in summary["by_tier"].items():
        for t, st in sorted(tiers.items()):
            print(f"  {mode:<28} tier={t} n={st['n']} rate={st['rate']*100:.1f}%")
    print("\nPer-regulator substring-match rate:")
    for mode, regs in summary["by_regulator"].items():
        for r, sr in sorted(regs.items()):
            print(f"  {mode:<28} reg={r:<6} n={sr['n']} rate={sr['rate']*100:.1f}%")
    print(bar)


def main():
    log("=== BFSI extract held-out eval (Nemotron-30B 4-bit) ===")
    eval_rows, eval_name = load_eval()
    # Optional cap via env var, e.g. MAX_EVAL_QUESTIONS=200 — stratifies across regulator+tier
    import os as _os
    cap = _os.environ.get("MAX_EVAL_QUESTIONS")
    if cap and cap.isdigit() and int(cap) < len(eval_rows):
        n_cap = int(cap)
        from collections import defaultdict as _dd
        import random as _random
        _random.seed(42)
        # Stratify by (regulator, tier) for balanced subset
        buckets = _dd(list)
        for r in eval_rows:
            buckets[(r.get("regulator","?"), r.get("tier","?"))].append(r)
        per = max(1, n_cap // max(1, len(buckets)))
        sampled = []
        for k, rs in buckets.items():
            _random.shuffle(rs)
            sampled.extend(rs[:per])
        eval_rows = sampled[:n_cap]
        log(f"MAX_EVAL_QUESTIONS={n_cap} -- stratified subset, n={len(eval_rows)} across {len(buckets)} (regulator,tier) buckets")

    bfsi_ready  = adapter_exists("bfsi_extract")
    other_ready = all(adapter_exists(n) for n in ["math", "code", "science"])
    modes = ["base"]
    if bfsi_ready: modes.append("bfsi_extract_only")
    else: log("WARN: bfsi_extract adapter missing -- skipping bfsi_extract_only.")
    if bfsi_ready and other_ready: modes.append("format_guard_with_bfsi")
    else: log("WARN: missing adapters for FG -- skipping format_guard_with_bfsi.")
    log(f"Modes to run: {modes}")

    log("Loading tokenizer + base model in 4-bit...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()

    if bfsi_ready and other_ready:
        model = PeftModel.from_pretrained(base_model, adapter_path("math"),
                                          adapter_name="math", is_trainable=False)
        model.load_adapter(adapter_path("code"), adapter_name="code")
        model.load_adapter(adapter_path("science"), adapter_name="science")
        model.load_adapter(adapter_path("bfsi_extract"), adapter_name="bfsi_extract")
    elif bfsi_ready:
        model = PeftModel.from_pretrained(base_model, adapter_path("bfsi_extract"),
                                          adapter_name="bfsi_extract", is_trainable=False)
    else:
        model = base_model
    model.eval()
    log(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    out_file = OUT_DIR / "eval_results.jsonl"
    summary_file = OUT_DIR / "summary.json"

    done = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["qa_id"], r["mode"]))
                except Exception:
                    pass
        log(f"Resume: {len(done)} (qa_id, mode) rows already in {out_file.name}")

    total_jobs = len(modes) * len(eval_rows)
    job_idx = 0
    t_start = time.time()

    for mode in modes:
        log(f"\n--- Mode: {mode} ---")
        for q in eval_rows:
            job_idx += 1
            qid = q["qa_id"]
            if (qid, mode) in done:
                continue
            t0 = time.time()
            try:
                user = (f"REGULATORY CONTEXT:\n{q['context']}\n\n"
                        f"QUESTION: {q['question']}\n\nANSWER:")
                msgs = [{"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user}]
                prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True)
                output, swap_count = generate(model, base_model, tok, prompt_text, mode)

                gold = q["answer"]
                sub = substring_match(gold, output)
                em = exact_match(gold, output)
                prec, rec, f1 = token_f1(gold, output)

                row = {"ts": datetime.datetime.utcnow().isoformat(), "qa_id": qid,
                       "tier": q.get("tier"), "regulator": q.get("regulator"),
                       "source_pdf": q.get("source_pdf"), "mode": mode,
                       "question": q["question"], "gold": gold,
                       "raw_output": output, "generated_answer": output,
                       "substring_match": sub, "exact_match": em,
                       "token_precision": round(prec, 4), "token_recall": round(rec, 4),
                       "token_f1": round(f1, 4), "swap_count": swap_count,
                       "elapsed_s": round(time.time() - t0, 2)}
                with open(out_file, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if job_idx % 10 == 0 or job_idx <= 5:
                    elapsed = time.time() - t_start
                    rate = job_idx / max(elapsed, 1e-6)
                    eta_min = (total_jobs - job_idx) / max(rate, 1e-6) / 60
                    log(f"  [{mode}] {job_idx}/{total_jobs} {qid}: "
                        f"sub={sub} f1={f1:.2f} ({row['elapsed_s']}s) ETA={eta_min:.0f}m")
            except Exception as e:
                log(f"  [{mode}] {qid} EXC: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    log("\nAggregating stats from disk...")
    summary = aggregate(out_file, eval_rows, eval_name)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote summary: {summary_file}")
    print_report(summary)
    log("=== Eval complete ===")


if __name__ == "__main__":
    main()
