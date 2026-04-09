"""
Hypothesis tests on Qwen2.5-1.5B + expert adapters (no CoT routing).

**Metrics (aligned with older Mistral vs Synapta / real_benchmark style):**
- semantic_sim: cosine similarity vs reference (all-MiniLM-L6-v2), same family as mistral_comparison / real_benchmark
- perplexity: assign probability to reference continuation under current adapter routing (+ layer gate when applicable)
- latency_s: wall time for generation
- exact_match, token_f1: objective overlap vs reference (strict; long refs often yield EM=0)

**Methods:**
- weighted_merge: 0.5/0.5 both adapters, all layers (baseline)
- late_layer_injection: same merge, LoRA only from layer N/2 upward
- sequential_token_segments: adapter A for first 48 tokens, B for next 132
- late_last_quarter (optional --extra): LoRA only in last ~25% of layers
- sequential_reverse (optional --extra): B first, then A (same token budget)
- early_third_only (--more): LoRA only in first ~1/3 of layers (inverse of late)
- oracle_single_d1 / oracle_single_d2 (--more): single adapter at full weight (oracle domains)
- merge_high_clamp (--more): merged weights with global weight_cap=1.0 (vs default 0.5)

Usage:
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --limit 10
    # Use PYTHONUNBUFFERED=1 or `python -u` so logs appear immediately when piping/teeing.
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --extra
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --extra --more
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --extra --more \\
        --data data/multidomain_eval_external.json --output results/injection_external.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from md_dataset import prepare_md_items


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _norm(pred) == _norm(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p = _norm(pred).split()
    r = _norm(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    p_count, r_count = {}, {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in r:
        r_count[t] = r_count.get(t, 0) + 1
    overlap = sum(min(p_count[t], r_count.get(t, 0)) for t in p_count)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def build_weights_two_domains(registry: dict, d1: str, d2: str, w1: float, w2: float) -> dict:
    out = {d: 0.0 for d in registry}
    if d1 in out:
        out[d1] = w1
    if d2 in out:
        out[d2] = w2
    return out


_sem_model = None


def semantic_similarity(pred: str, ref: str) -> float:
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer

        _sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = _sem_model.encode([pred or "", ref or ""], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def perplexity_for_setting(
    engine, prompt: str, ref: str, weights: dict, gate_min: int, gate_max: int = -1
) -> float:
    engine.set_adapter_layer_gate(gate_min, gate_max)
    p = engine.compute_perplexity(prompt, ref, routing_weights=weights)
    engine.set_adapter_layer_gate(0, -1)
    return float(min(p, 99999.0))


def run():
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.chdir(BACKEND_DIR)
    from dynamic_mlx_inference import DynamicEngine

    registry = json.load(open("expert_registry.json"))
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
    n_layers = engine._num_layers or 28
    late_start = max(1, n_layers // 2)
    late_last_q = max(1, int(n_layers * 0.75))
    early_max = max(0, n_layers // 3 - 1)

    data_rel = getattr(run, "_data_path", None) or "data/multidomain_eval_v2.json"
    limit = getattr(run, "_limit", None)
    path, items = prepare_md_items(
        PROJECT_ROOT,
        data_rel,
        limit=int(limit) if limit else None,
        two_domain_only=True,
    )
    st = path.stat()
    print(
        f"Dataset: {path.resolve()}\n"
        f"  mtime_utc={datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()}  "
        f"bytes={st.st_size}  items={len(items)} (>=2 domains)\n"
        f"  first_id={items[0].get('id') if items else '—'}",
        flush=True,
    )

    methods = list(getattr(run, "_methods", None) or [])
    if not methods:
        methods = ["weighted_merge", "late_layer_injection", "sequential_token_segments"]
    results = []

    total = len(items) * len(methods)
    idx = 0
    for item in items:
        doms = item.get("required_adapters") or item.get("domains") or []
        d1, d2 = doms[0], doms[1]
        q = item["question"]
        ref = item["reference_answer"]
        prompt = f"<|im_start|>user\n{q}<|redacted_im_end|>\n<|im_start|>assistant\n"

        w_mix = build_weights_two_domains(registry, d1, d2, 0.5, 0.5)
        w_a = build_weights_two_domains(registry, d1, d2, 1.0, 0.0)
        w_b = build_weights_two_domains(registry, d1, d2, 0.0, 1.0)

        for method in methods:
            idx += 1
            t0 = time.time()
            if method == "weighted_merge":
                engine.set_adapter_layer_gate(0, -1)
                pred, _ = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)
            elif method == "late_layer_injection":
                engine.set_adapter_layer_gate(late_start, -1)
                pred, _ = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)
                engine.set_adapter_layer_gate(0, -1)
            elif method == "late_last_quarter":
                engine.set_adapter_layer_gate(late_last_q, -1)
                pred, _ = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)
                engine.set_adapter_layer_gate(0, -1)
            elif method == "early_third_only":
                engine.set_adapter_layer_gate(0, early_max)
                pred, _ = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)
                engine.set_adapter_layer_gate(0, -1)
            elif method == "oracle_single_d1":
                engine.set_adapter_layer_gate(0, -1)
                pred, _ = engine.generate(prompt, routing_weights=w_a, max_tokens=180)
            elif method == "oracle_single_d2":
                engine.set_adapter_layer_gate(0, -1)
                pred, _ = engine.generate(prompt, routing_weights=w_b, max_tokens=180)
            elif method == "merge_high_clamp":
                from dynamic_mlx_inference import set_global_clamp

                set_global_clamp(1.0)
                try:
                    engine.set_adapter_layer_gate(0, -1)
                    pred, _ = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)
                finally:
                    set_global_clamp(0.5)
            elif method == "sequential_token_segments":
                engine.set_adapter_layer_gate(0, -1)
                pred, _ = engine.generate_sequential_segments(
                    prompt,
                    [(w_a, 48), (w_b, 132)],
                    reset_weights_between=True,
                )
            elif method == "sequential_reverse":
                engine.set_adapter_layer_gate(0, -1)
                pred, _ = engine.generate_sequential_segments(
                    prompt,
                    [(w_b, 48), (w_a, 132)],
                    reset_weights_between=True,
                )
            else:
                raise ValueError(method)

            lat = time.time() - t0
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)
            sim = semantic_similarity(pred, ref)
            if method == "weighted_merge":
                ppl = perplexity_for_setting(engine, prompt, ref, w_mix, 0, -1)
            elif method == "late_layer_injection":
                ppl = perplexity_for_setting(engine, prompt, ref, w_mix, late_start, -1)
            elif method == "late_last_quarter":
                ppl = perplexity_for_setting(engine, prompt, ref, w_mix, late_last_q, -1)
            elif method == "early_third_only":
                ppl = perplexity_for_setting(engine, prompt, ref, w_mix, 0, early_max)
            elif method == "oracle_single_d1":
                ppl = perplexity_for_setting(engine, prompt, ref, w_a, 0, -1)
            elif method == "oracle_single_d2":
                ppl = perplexity_for_setting(engine, prompt, ref, w_b, 0, -1)
            elif method == "merge_high_clamp":
                from dynamic_mlx_inference import set_global_clamp

                set_global_clamp(1.0)
                try:
                    ppl = perplexity_for_setting(engine, prompt, ref, w_mix, 0, -1)
                finally:
                    set_global_clamp(0.5)
            else:
                # Sequential paths: PPL under mixture, all layers (proxy for comparability).
                ppl = perplexity_for_setting(engine, prompt, ref, w_mix, 0, -1)

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_path": (
                    str(path.relative_to(PROJECT_ROOT))
                    if str(path).startswith(str(PROJECT_ROOT))
                    else str(path)
                ),
                "item_id": item.get("id"),
                "domains": [d1, d2],
                "method": method,
                "semantic_sim": round(sim, 4),
                "perplexity": round(ppl, 2),
                "exact_match": em,
                "token_f1": f1,
                "latency_s": round(lat, 3),
                "late_start_layer": late_start if method == "late_layer_injection" else None,
                "late_last_quarter_start": late_last_q if method == "late_last_quarter" else None,
                "early_third_max_layer": early_max if method == "early_third_only" else None,
                "n_layers": n_layers,
                "prediction_text": pred,
                "prediction_preview": (pred or "")[:280],
            }
            results.append(row)
            print(
                f"[{idx}/{total}] {method:28s} | id={item.get('id')} | "
                f"Sim={sim:.3f} PPL={ppl:.1f} EM={em:.2f} F1={f1:.2f} | lat={lat:.2f}s",
                flush=True,
            )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = getattr(run, "_out_path", None) or (out_dir / "injection_hypotheses_eval.jsonl")
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved: {out_path}", flush=True)
    agg = {}
    for m in methods:
        rows = [r for r in results if r["method"] == m]
        if not rows:
            continue
        agg[m] = {
            "semantic_sim": float(np.mean([x["semantic_sim"] for x in rows])),
            "perplexity": float(np.mean([x["perplexity"] for x in rows])),
            "em": float(np.mean([x["exact_match"] for x in rows])),
            "f1": float(np.mean([x["token_f1"] for x in rows])),
            "lat": float(np.mean([x["latency_s"] for x in rows])),
        }
        print(
            f"  {m:28s} | Sim={agg[m]['semantic_sim']:.3f} | PPL={agg[m]['perplexity']:.1f} | "
            f"EM={agg[m]['em']:.3f} | F1={agg[m]['f1']:.3f} | Lat={agg[m]['lat']:.2f}s",
            flush=True,
        )

    base = agg.get("weighted_merge")
    if base:
        for m in methods:
            if m == "weighted_merge" or m not in agg:
                continue
            rel_f1 = (agg[m]["f1"] - base["f1"]) / max(1e-9, base["f1"] if base["f1"] > 0 else 1e-9)
            rel_em = (agg[m]["em"] - base["em"]) / max(1e-9, base["em"] if base["em"] > 0 else 1e-9)
            rel_sim = (agg[m]["semantic_sim"] - base["semantic_sim"]) / max(
                1e-9, base["semantic_sim"] if base["semantic_sim"] > 0 else 1e-9
            )
            print(f"\n vs weighted_merge [{m}]:", flush=True)
            print(
                f"   ΔSim: {rel_sim:+.2%}  ΔPPL: {agg[m]['perplexity'] - base['perplexity']:+.2f}",
                flush=True,
            )
            print(f"   ΔEM relative: {rel_em:+.2%}  ΔF1 relative: {rel_f1:+.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also run late_last_quarter and sequential_reverse hypotheses.",
    )
    parser.add_argument(
        "--more",
        action="store_true",
        help="Also run early_third_only, oracle_single_d1, oracle_single_d2, merge_high_clamp.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSONL path under project root (default: results/injection_hypotheses_eval.jsonl).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/multidomain_eval_v2.json",
        help="MD benchmark JSON (same schema as multidomain_eval_v2.json).",
    )
    args = parser.parse_args()
    if not args.real:
        print("Use --real to run GPU inference.")
        raise SystemExit(0)
    run._limit = args.limit if args.limit and args.limit > 0 else None
    core = ["weighted_merge", "late_layer_injection", "sequential_token_segments"]
    extra = ["late_last_quarter", "sequential_reverse"]
    more = ["early_third_only", "oracle_single_d1", "oracle_single_d2", "merge_high_clamp"]
    mth = list(core)
    if args.extra:
        mth += extra
    if args.more:
        mth += more
    run._methods = mth
    run._out_path = args.output.strip() or None
    run._data_path = args.data.strip() or "data/multidomain_eval_v2.json"
    run()
