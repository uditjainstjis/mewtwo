from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from hf_trained_router import HFTrainedRouter  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def gold_experts(row: dict) -> list[str]:
    return list(row.get("experts") or row.get("required_adapters") or row.get("domains") or [])


def overlap_score(pred: list[str], gold: list[str]) -> float:
    p = set(pred)
    g = set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--registry", default="backend/expert_registry.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-experts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    registry_path = PROJECT_ROOT / args.registry
    adapter_path = PROJECT_ROOT / args.adapter

    rows = load_rows(data_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    router = HFTrainedRouter(registry_path, args.model, adapter_path)
    per_row = []
    exact = 0
    partial = 0
    overlap_total = 0.0

    for row in rows:
        question = row["question"]
        gold = gold_experts(row)
        decision = router.route(question, max_experts=args.max_experts, max_tokens=args.max_tokens)
        pred = decision.experts
        ex = int(pred == gold)
        part = int(bool(set(pred) & set(gold)))
        ov = overlap_score(pred, gold)
        exact += ex
        partial += part
        overlap_total += ov
        per_row.append(
            {
                "id": row.get("id"),
                "question": question,
                "gold_experts": gold,
                "pred_experts": pred,
                "exact_match": ex,
                "partial_overlap": part,
                "overlap_f1": round(ov, 4),
                "router_thinking": decision.thinking,
                "router_raw_text": decision.raw_text,
                "latency_s": round(decision.latency_s, 4),
            }
        )

    summary = {
        "n": len(per_row),
        "exact_match_rate": round(exact / len(per_row), 4) if per_row else 0.0,
        "partial_overlap_rate": round(partial / len(per_row), 4) if per_row else 0.0,
        "mean_overlap_f1": round(overlap_total / len(per_row), 4) if per_row else 0.0,
        "mean_latency_s": round(sum(row["latency_s"] for row in per_row) / len(per_row), 4) if per_row else 0.0,
    }

    payload = {"summary": summary, "rows": per_row}
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
