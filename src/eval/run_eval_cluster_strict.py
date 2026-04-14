"""
Strict cluster evaluation against standard routing baseline.

Usage:
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_cluster_strict.py --real
    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_cluster_strict.py --real --limit 5 --data data/multidomain_eval_external.json \\
        --output cluster_external.jsonl
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


def run():
    os.chdir(BACKEND_DIR)
    registry = json.load(open("expert_registry.json"))
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry)
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)
    cluster = AdversarialAgentCluster(engine=engine, orchestrator=orchestrator)

    data_rel = getattr(run, "_data_path", None) or "data/multidomain_eval_v2.json"
    limit = getattr(run, "_limit", None)
    two_only = not getattr(run, "_all_items", False)
    _, items = prepare_md_items(
        PROJECT_ROOT,
        data_rel,
        limit=int(limit) if limit else None,
        two_domain_only=two_only,
    )
    print(f"Dataset: {data_rel} | {len(items)} items (after limit/domain filter)\n")
    methods = ["standard", "cluster_strict"]
    results = []

    total = len(items) * len(methods)
    idx = 0
    for item in items:
        query = item["question"]
        ref = item["reference_answer"]
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        for method in methods:
            idx += 1
            start = time.time()
            if method == "standard":
                weights, _ = orchestrator.route_top2(query)
                pred, _ = engine.generate(prompt, routing_weights=weights, max_tokens=150)
                passed = True
                veto_count = 0
                rounds_used = 1
                evidence_completeness = 0.0
                disagreement_entropy = 0.0
                reaudits = 0
                high_disagreement_rounds = 0
                composer_samples = 0.0
            else:
                cr = cluster.run(query)
                pred = cr.response
                passed = cr.passed
                veto_count = len(cr.veto_reasons)
                rounds_used = cr.rounds_used
                evidence_completeness = cr.metrics.get("evidence_completeness", 0.0)
                round_entropies = [rd.get("consensus_entropy", 0.0) for rd in cr.trace.get("rounds", [])]
                disagreement_entropy = float(np.mean(round_entropies)) if round_entropies else 0.0
                reaudits = sum(1 for rd in cr.trace.get("rounds", []) if rd.get("reaudit_triggered"))
                high_disagreement_rounds = sum(1 for rd in cr.trace.get("rounds", []) if rd.get("high_disagreement"))
                composer_samples = float(cr.metrics.get("composer_samples", 0.0))

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
                "passed": passed,
                "veto_count": veto_count,
                "rounds_used": rounds_used,
                "evidence_completeness": evidence_completeness,
                "disagreement_entropy": disagreement_entropy,
                "reaudit_count": reaudits,
                "high_disagreement_rounds": high_disagreement_rounds,
                "composer_samples": composer_samples,
                "prediction_preview": (pred or "")[:250],
            }
            results.append(row)
            print(
                f"[{idx}/{total}] {method:14s} | EM={em:.2f} | F1={f1:.2f} | "
                f"lat={latency:.2f}s | pass={passed} | veto={veto_count}"
            )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_rel = getattr(run, "_out_path", None) or "cluster_strict_eval.jsonl"
    out_path = Path(out_rel)
    if not out_path.is_absolute():
        out_path = out_dir / out_path
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\nAggregate:")
    aggregate = {}
    for method in methods:
        m = [r for r in results if r["method"] == method]
        em_avg = float(np.mean([x["exact_match"] for x in m]))
        f1_avg = float(np.mean([x["token_f1"] for x in m]))
        lat_avg = float(np.mean([x["latency_s"] for x in m]))
        pass_rate = float(np.mean([1.0 if x["passed"] else 0.0 for x in m]))
        evidence = float(np.mean([x["evidence_completeness"] for x in m]))
        aggregate[method] = {
            "exact_match": em_avg,
            "token_f1": f1_avg,
            "latency_s": lat_avg,
            "pass_rate": pass_rate,
            "evidence_completeness": evidence,
            "disagreement_entropy": float(np.mean([x.get("disagreement_entropy", 0.0) for x in m])),
            "reaudit_count": float(np.mean([x.get("reaudit_count", 0) for x in m])),
        }
        print(
            f"  {method:14s} | EM={em_avg:.3f} | "
            f"F1={f1_avg:.3f} | "
            f"Lat={lat_avg:.2f}s | "
            f"PassRate={pass_rate:.3f} | "
            f"Evidence={evidence:.3f} | "
            f"DisEntropy={aggregate[method]['disagreement_entropy']:.3f} | "
            f"Reaudit={aggregate[method]['reaudit_count']:.2f}"
        )
    if "standard" in aggregate and "cluster_strict" in aggregate:
        base = aggregate["standard"]
        strict = aggregate["cluster_strict"]
        rel_em = (strict["exact_match"] - base["exact_match"]) / max(1e-9, base["exact_match"] if base["exact_match"] > 0 else 1e-9)
        rel_f1 = (strict["token_f1"] - base["token_f1"]) / max(1e-9, base["token_f1"] if base["token_f1"] > 0 else 1e-9)
        print("\nRelative improvement vs standard:")
        print(f"  EM relative gain: {rel_em:+.2%}")
        print(f"  F1 relative gain: {rel_f1:+.2%}")
        print(f"  3x target reached (EM or F1): {('YES' if (rel_em >= 2.0 or rel_f1 >= 2.0) else 'NO')}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--data",
        type=str,
        default="data/multidomain_eval_v2.json",
        help="MD benchmark JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSONL under results/ or absolute path (default: cluster_strict_eval.jsonl).",
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Include single-domain rows (default: only >=2 domains).",
    )
    args = parser.parse_args()
    if not args.real:
        print("Use --real to run actual strict-cluster evaluation.")
        raise SystemExit(0)
    run._limit = args.limit if args.limit and args.limit > 0 else None
    run._data_path = args.data.strip() or "data/multidomain_eval_v2.json"
    run._out_path = args.output.strip() or None
    run._all_items = args.all_items
    run()
