#!/usr/bin/env python3
"""Frontier-comparison eval: BFSI LoRA adapter vs GPT-4o / Claude 3.5 Sonnet on
the SAME 50 stratified held-out RBI/SEBI questions. Same prompt, same scoring
(substring / exact-match / token F1), per-call latency + cost telemetry.
Stratification prefers Tier-3 (heading-based extractive, lowest contamination
risk for closed APIs); falls back to Tier-2 to top up. Synapta numbers are
PULLED from results/bfsi_eval/eval_results.jsonl (mode=bfsi_extract_only).

Outputs:
  results/frontier_comparison/results.jsonl  (per-row, append-as-you-go)
  results/frontier_comparison/summary.json   (final summary)

Usage:
  export OPENAI_API_KEY=... ; export ANTHROPIC_API_KEY=...
  python synapta_src/data_pipeline/11_frontier_comparison.py [--n 50] \\
      [--skip-openai] [--skip-anthropic] [--cache] [--dry-run] [--no-cost-cap]
"""
import argparse, datetime, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
EVAL_CLEAN = PROJECT / "data" / "rbi_corpus" / "qa" / "eval_clean.jsonl"
BFSI_RESULTS = PROJECT / "results" / "bfsi_eval" / "eval_results.jsonl"
OUT_DIR = PROJECT / "results" / "frontier_comparison"
OUT_FILE = OUT_DIR / "results.jsonl"
SUMMARY_FILE = OUT_DIR / "summary.json"

SYSTEM_MSG = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible."
)
STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is",
             "are", "was", "were", "be", "been", "by", "with", "as", "at", "from",
             "that", "this", "it", "its"}

# Pricing per 1M tokens (input, output) USD — hardcoded per task spec.
MODEL_REGISTRY = {
    "gpt-4o":            {"provider": "openai",    "model_id": "gpt-4o",
                          "input_per_m": 5.00, "output_per_m": 15.00},
    "gpt-4o-mini":       {"provider": "openai",    "model_id": "gpt-4o-mini",
                          "input_per_m": 0.15, "output_per_m": 0.60},
    "claude-3-5-sonnet": {"provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022",
                          "input_per_m": 3.00, "output_per_m": 15.00},
}

REQUEST_TIMEOUT_S = 30
HARD_COST_CAP_USD = 10.0
EST_TOKENS_PER_Q = 5000
MAX_OUTPUT_TOKENS = 512


# ---- scoring (mirrors 08_eval_bfsi_extract.py) ----

def normalise_tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]

def substring_match(gold, pred):
    return 1 if gold.strip().lower() in pred.lower() else 0

def exact_match(gold, pred):
    return 1 if gold.strip().lower() == pred.strip().lower() else 0

def token_f1(gold, pred):
    g, p = normalise_tokens(gold), normalise_tokens(pred)
    if not g and not p: return 1.0
    if not g or not p:  return 0.0
    overlap = sum((Counter(g) & Counter(p)).values())
    if overlap == 0:    return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# ---- data + sampling ----

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_bfsi_baseline():
    if not BFSI_RESULTS.exists(): return {}
    out = {}
    for r in load_jsonl(BFSI_RESULTS):
        if r.get("mode") == "bfsi_extract_only":
            out[r["qa_id"]] = r
    return out

def stratified_sample(eval_rows, baseline_ids, n, seed=42):
    """Pick n with 50/50 RBI/SEBI split. Prefer Tier-3 (lowest contamination risk
    for frontier models); fall back to Tier-2 if intersection with baseline short."""
    rng = random.Random(seed)
    n_per_reg = n // 2
    picks, fill_log = [], {}
    for reg in ("RBI", "SEBI"):
        t3 = [r for r in eval_rows if r.get("regulator") == reg
              and r.get("tier") == 3 and r["qa_id"] in baseline_ids]
        rng.shuffle(t3); chosen = t3[:n_per_reg]
        if len(chosen) < n_per_reg:
            t2 = [r for r in eval_rows if r.get("regulator") == reg
                  and r.get("tier") == 2 and r["qa_id"] in baseline_ids]
            rng.shuffle(t2); chosen += t2[:n_per_reg - len(chosen)]
        picks.extend(chosen)
        fill_log[reg] = {"n": len(chosen),
                         "t3": sum(1 for x in chosen if x.get("tier") == 3),
                         "t2": sum(1 for x in chosen if x.get("tier") == 2)}
    return picks, fill_log


# ---- provider call wrappers ----

def build_user_prompt(qa):
    return f"REGULATORY CONTEXT:\n{qa['context']}\n\nQUESTION: {qa['question']}\n\nANSWER:"

def call_openai(client, model_id, system_msg, user_msg):
    resp = client.with_options(timeout=REQUEST_TIMEOUT_S).chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        max_tokens=MAX_OUTPUT_TOKENS, temperature=0)
    text = resp.choices[0].message.content or ""
    return text.strip(), resp.usage.prompt_tokens, resp.usage.completion_tokens

def call_anthropic(client, model_id, system_msg, user_msg):
    resp = client.with_options(timeout=REQUEST_TIMEOUT_S).messages.create(
        model=model_id, max_tokens=MAX_OUTPUT_TOKENS,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text.strip(), resp.usage.input_tokens, resp.usage.output_tokens

def estimate_cost(model_name, in_tok, out_tok):
    s = MODEL_REGISTRY[model_name]
    return (in_tok / 1e6) * s["input_per_m"] + (out_tok / 1e6) * s["output_per_m"]

def get_clients(skip_openai, skip_anthropic):
    """Lazy-import each SDK so a missing one is just a skipped model."""
    clients = {"openai": None, "anthropic": None}
    if not skip_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[warn] OPENAI_API_KEY not set -- skipping OpenAI models.")
        else:
            try:
                import openai
                clients["openai"] = openai.OpenAI()
            except ImportError:
                print("[warn] `openai` package not installed -- skipping OpenAI.")
            except Exception as e:
                print(f"[warn] OpenAI client init failed: {e}")
    if not skip_anthropic:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[warn] ANTHROPIC_API_KEY not set -- skipping Anthropic.")
        else:
            try:
                import anthropic
                clients["anthropic"] = anthropic.Anthropic()
            except ImportError:
                print("[warn] `anthropic` package not installed -- skipping Anthropic.")
            except Exception as e:
                print(f"[warn] Anthropic client init failed: {e}")
    return clients

def models_to_run(clients):
    out = []
    for name, spec in MODEL_REGISTRY.items():
        if clients.get(spec["provider"]) is not None:
            out.append(name)
    return out

def run_one(client, provider, model_id, system_msg, user_msg):
    """Returns (text, in_tok, out_tok, latency_ms, error_str_or_None)."""
    t0 = time.perf_counter()
    try:
        if provider == "openai":
            text, in_tok, out_tok = call_openai(client, model_id, system_msg, user_msg)
        elif provider == "anthropic":
            text, in_tok, out_tok = call_anthropic(client, model_id, system_msg, user_msg)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        return text, in_tok, out_tok, int((time.perf_counter() - t0) * 1000), None
    except Exception as e:
        return "", 0, 0, int((time.perf_counter() - t0) * 1000), f"{type(e).__name__}: {e}"


# ---- IO + aggregation ----

def load_cached_rows():
    if not OUT_FILE.exists(): return [], set()
    rows, done = [], set()
    for r in load_jsonl(OUT_FILE):
        rows.append(r); done.add((r["qa_id"], r["model"]))
    return rows, done

def append_row(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")

def synapta_row_from_baseline(b, qa):
    return {
        "ts": datetime.datetime.utcnow().isoformat(),
        "qa_id": qa["qa_id"], "tier": qa.get("tier"), "regulator": qa.get("regulator"),
        "model": "synapta_bfsi_extract", "provider": "local_lora",
        "model_id": "nemotron-30b + bfsi_extract LoRA",
        "question": qa["question"], "gold": qa["answer"],
        "generated_answer": b.get("generated_answer", ""),
        "substring_match": int(b.get("substring_match", 0)),
        "exact_match": int(b.get("exact_match", 0)),
        "token_f1": float(b.get("token_f1", 0.0)),
        "input_tokens": 0, "output_tokens": 0,
        "latency_ms": int(float(b.get("elapsed_s", 0.0)) * 1000),
        "cost_usd": 0.0001,  # nominal local-inference cost
        "error": None,
    }

def aggregate(rows, n_questions):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    summary = {"n_questions": n_questions, "models": {}}
    for model, mr in by_model.items():
        ok = [r for r in mr if r.get("error") is None]
        n_ok = len(ok) or 1
        n_correct = sum(r["substring_match"] for r in ok)
        total_cost = sum(r["cost_usd"] for r in ok)
        summary["models"][model] = {
            "n": len(mr), "n_ok": len(ok), "n_errors": len(mr) - len(ok),
            "substring":   round(sum(r["substring_match"] for r in ok) / n_ok, 4),
            "exact_match": round(sum(r["exact_match"] for r in ok) / n_ok, 4),
            "f1":          round(sum(r["token_f1"] for r in ok) / n_ok, 4),
            "ms_per_q":    round(sum(r["latency_ms"] for r in ok) / n_ok, 1),
            "cost_per_q_usd": round(total_cost / n_ok, 6),
            "total_cost_usd": round(total_cost, 4),
            "n_correct_substring": n_correct,
            "cost_per_correct_usd": round(total_cost / n_correct, 6) if n_correct else None,
        }
    return summary

def print_table(summary):
    bar = "=" * 96
    print(f"\n{bar}\nFRONTIER COMPARISON SUMMARY  (n_questions={summary['n_questions']})\n{bar}")
    hdr = (f"{'Model':<28} {'N':>4} {'Sub%':>7} {'EM%':>6} {'F1':>6} "
           f"{'ms/q':>7} {'$/q':>10} {'$/correct':>11}")
    print(hdr); print("-" * len(hdr))
    for model, s in summary["models"].items():
        cpc = f"${s['cost_per_correct_usd']:.4f}" if s["cost_per_correct_usd"] is not None else "n/a"
        print(f"{model:<28} {s['n']:>4} {s['substring']*100:>6.1f}% "
              f"{s['exact_match']*100:>5.1f}% {s['f1']*100:>5.1f} "
              f"{s['ms_per_q']:>6.0f}  ${s['cost_per_q_usd']:>8.5f} {cpc:>11}")
    print(bar)


# ---- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip-openai", action="store_true")
    ap.add_argument("--skip-anthropic", action="store_true")
    ap.add_argument("--cache", action="store_true",
                    help="reuse prior rows in results.jsonl (skip done qa_id+model)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one rendered prompt and exit; no API calls")
    ap.add_argument("--no-cost-cap", action="store_true",
                    help="disable the $10 hard cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EVAL_CLEAN.exists():
        sys.exit(f"FATAL: eval_clean.jsonl not found at {EVAL_CLEAN}")
    eval_rows = load_jsonl(EVAL_CLEAN)
    baseline = load_bfsi_baseline()
    if not baseline:
        print(f"[warn] no synapta baseline at {BFSI_RESULTS} -- comparison will lack our model.")

    picks, fill_log = stratified_sample(eval_rows, baseline, args.n, seed=args.seed)
    print(f"[info] stratified picks: {fill_log}  total={len(picks)}")
    if len(picks) < args.n:
        print(f"[warn] requested {args.n} but only {len(picks)} qa available with baseline coverage.")

    if args.dry_run:
        active = [m for m, s in MODEL_REGISTRY.items()
                  if not (args.skip_openai and s["provider"] == "openai")
                  and not (args.skip_anthropic and s["provider"] == "anthropic")]
        for qa in picks[:1]:
            user_msg = build_user_prompt(qa)
            print("\n--- DRY RUN: rendered prompt for first qa ---")
            print(f"qa_id={qa['qa_id']} regulator={qa.get('regulator')} tier={qa.get('tier')}")
            print(f"SYSTEM:\n{SYSTEM_MSG}\n\nUSER:\n{user_msg[:1500]}"
                  f"{' ...[truncated]' if len(user_msg) > 1500 else ''}")
            print(f"\nWould send to: {active}")
        print(f"\n[dry-run] would issue {len(picks)} questions x {len(active)} models = "
              f"{len(picks) * len(active)} API calls.")
        return

    clients = get_clients(args.skip_openai, args.skip_anthropic)
    api_models = models_to_run(clients)
    if not api_models:
        print("[warn] no API models active (no keys/SDKs). Will only emit synapta baseline.")

    # cost cap check
    est_cost = 0.0
    for m in api_models:
        s = MODEL_REGISTRY[m]
        in_t, out_t = EST_TOKENS_PER_Q * 0.8, EST_TOKENS_PER_Q * 0.2
        est_cost += len(picks) * ((in_t/1e6)*s["input_per_m"] + (out_t/1e6)*s["output_per_m"])
    print(f"[info] estimated total cost: ${est_cost:.2f}  (cap=${HARD_COST_CAP_USD:.2f})")
    if est_cost > HARD_COST_CAP_USD and not args.no_cost_cap:
        sys.exit(f"REFUSED: estimated cost ${est_cost:.2f} exceeds cap "
                 f"${HARD_COST_CAP_USD:.2f}. Re-run with --no-cost-cap to override.")

    cached_rows, done = (([], set()) if not args.cache else load_cached_rows())
    if args.cache:
        print(f"[info] cache: {len(done)} rows already on disk; will skip those.")

    # synapta baseline rows (always emit, no API cost)
    synapta_done = {qid for (qid, m) in done if m == "synapta_bfsi_extract"}
    for qa in picks:
        if qa["qa_id"] in synapta_done: continue
        b = baseline.get(qa["qa_id"])
        if b: append_row(synapta_row_from_baseline(b, qa))

    # frontier API calls
    for model_name in api_models:
        spec = MODEL_REGISTRY[model_name]
        client = clients[spec["provider"]]
        print(f"\n--- Running model: {model_name} ({spec['model_id']}) ---")
        for i, qa in enumerate(picks, 1):
            if (qa["qa_id"], model_name) in done: continue
            user_msg = build_user_prompt(qa)
            text, in_tok, out_tok, latency_ms, err = run_one(
                client, spec["provider"], spec["model_id"], SYSTEM_MSG, user_msg)
            cost = estimate_cost(model_name, in_tok, out_tok) if not err else 0.0
            sub = substring_match(qa["answer"], text) if not err else 0
            em  = exact_match(qa["answer"], text) if not err else 0
            f1  = token_f1(qa["answer"], text) if not err else 0.0
            row = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "qa_id": qa["qa_id"], "tier": qa.get("tier"), "regulator": qa.get("regulator"),
                "model": model_name, "provider": spec["provider"], "model_id": spec["model_id"],
                "question": qa["question"], "gold": qa["answer"], "generated_answer": text,
                "substring_match": sub, "exact_match": em, "token_f1": round(f1, 4),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "latency_ms": latency_ms, "cost_usd": round(cost, 6), "error": err,
            }
            append_row(row)
            if i % 5 == 0 or i <= 3 or err:
                tag = "ERR" if err else f"sub={sub} f1={f1:.2f}"
                print(f"  [{model_name}] {i}/{len(picks)} {qa['qa_id']}: "
                      f"{tag} ({latency_ms}ms, ${cost:.5f})")

    all_rows = load_cached_rows()[0]
    summary = aggregate(all_rows, len(picks))
    summary["stratification"] = fill_log
    summary["seed"] = args.seed
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[info] wrote summary: {SUMMARY_FILE}")
    print_table(summary)


if __name__ == "__main__":
    main()
