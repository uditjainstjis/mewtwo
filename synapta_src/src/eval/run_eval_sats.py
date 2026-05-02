"""
Evaluate Speculative Adapter Trajectory Search (SATS) vs baselines on multidomain_eval_v2.

Usage:
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_sats.py --real
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_sats.py --real --data data/multidomain_eval_external.json --output sats_external.jsonl
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

from dynamic_mlx_inference import DynamicEngine
from orchestrator import Orchestrator
from agent_cluster import AdversarialAgentCluster
from trajectory_search import run_sats


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


def load_items():
    rel = getattr(run, "_data_path", None) or "data/multidomain_eval_v2.json"
    limit = getattr(run, "_limit", None)
    two_only = not getattr(run, "_all_items", False)
    _, items = prepare_md_items(
        PROJECT_ROOT,
        rel,
        limit=int(limit) if limit else None,
        two_domain_only=two_only,
    )
    return items


def run():
    os.chdir(BACKEND_DIR)
    registry = json.load(open("expert_registry.json"))
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)
    cluster = AdversarialAgentCluster(engine=engine, orchestrator=orchestrator)

    items = load_items()
    data_rel = getattr(run, "_data_path", None) or "data/multidomain_eval_v2.json"
    print(f"Dataset: {data_rel} | {len(items)} items (after limit/domain filter)\n")
    methods = getattr(run, "_methods", None) or ["standard_top2", "sats_router", "sats_oracle_domains"]
    results = []

    total = len(items) * len(methods)
    idx = 0
    for item in items:
        q = item["question"]
        ref = item["reference_answer"]
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

        for method in methods:
            idx += 1
            start = time.time()
            if method == "standard_top2":
                w, _, _ = orchestrator.route_top2(q)
                pred, _ = engine.generate(prompt, routing_weights=w, max_tokens=180)
                aux = {}
            elif method == "cluster_strict":
                cr = cluster.run(q)
                pred = cr.response
                aux = {"passed": cr.passed, "rounds_used": cr.rounds_used}
            elif method == "sats_router":
                r = run_sats(engine, orchestrator, item, use_oracle_domains=False)
                pred = r["best_text"]
                aux = {"best_name": r["best_name"], "best_scores": r["best_scores"]}
            elif method == "sats_oracle_domains":
                r = run_sats(engine, orchestrator, item, use_oracle_domains=True)
                pred = r["best_text"]
                aux = {"best_name": r["best_name"], "best_scores": r["best_scores"]}
            else:
                raise ValueError(method)

            latency = time.time() - start
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_path": data_rel,
                "item_id": item.get("id"),
                "method": method,
                "exact_match": em,
                "token_f1": f1,
                "latency_s": round(latency, 3),
                "prediction_preview": (pred or "")[:250],
                "aux": aux,
            }
            results.append(row)
            print(f"[{idx}/{total}] {method:18s} | EM={em:.2f} | F1={f1:.2f} | lat={latency:.2f}s")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_rel = getattr(run, "_out_path", None) or "sats_eval.jsonl"
    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\nAggregate:")
    agg = {}
    for method in methods:
        m = [r for r in results if r["method"] == method]
        agg[method] = {
            "em": float(np.mean([x["exact_match"] for x in m])),
            "f1": float(np.mean([x["token_f1"] for x in m])),
            "lat": float(np.mean([x["latency_s"] for x in m])),
        }
        print(f"  {method:18s} | EM={agg[method]['em']:.3f} | F1={agg[method]['f1']:.3f} | Lat={agg[method]['lat']:.2f}s")

    if "standard_top2" in agg:
        base = agg["standard_top2"]
        for k in methods:
            if k == "standard_top2":
                continue
            rel_em = (agg[k]["em"] - base["em"]) / max(1e-9, base["em"] if base["em"] > 0 else 1e-9)
            rel_f1 = (agg[k]["f1"] - base["f1"]) / max(1e-9, base["f1"] if base["f1"] > 0 else 1e-9)
            print(f"\nRelative vs standard_top2 for {k}:")
            print(f"  EM relative gain: {rel_em:+.2%}")
            print(f"  F1 relative gain: {rel_f1:+.2%}")
            print(f"  3x target reached (EM or F1): {('YES' if (rel_em >= 2.0 or rel_f1 >= 2.0) else 'NO')}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSONL filename under results/ or absolute path (default: sats_eval.jsonl).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/multidomain_eval_v2.json",
        help="MD benchmark JSON.",
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Include single-domain rows (default: only >=2 domains, matching injection eval).",
    )
    args = parser.parse_args()
    if not args.real:
        print("Use --real to run actual experiments.")
        raise SystemExit(0)
    run._limit = args.limit if args.limit and args.limit > 0 else None
    if args.methods.strip():
        run._methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run._out_path = args.output.strip() or None
    run._data_path = args.data.strip() or "data/multidomain_eval_v2.json"
    run._all_items = args.all_items
    run()

