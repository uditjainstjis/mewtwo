"""
Aggregate mean metrics from run_eval_injection_hypotheses.py JSONL output.

Usage:
  python3 src/eval/aggregate_injection_jsonl.py results/injection_hypotheses_eval_full_20260408.jsonl
  python3 src/eval/aggregate_injection_jsonl.py results/injection_hypotheses_eval_full_20260408.jsonl --by-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=str)
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Break out rows by dataset_path field.",
    )
    parser.add_argument(
        "--ppl-cap",
        type=float,
        default=0.0,
        help="If >0, cap each row's perplexity at this value before averaging (reduces outlier spikes).",
    )
    args = parser.parse_args()
    ppl_cap = float(args.ppl_cap or 0.0)
    path = Path(args.jsonl)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print("No rows.", file=sys.stderr)
        raise SystemExit(1)

    def ppl_vals(group: list[dict]) -> list[float]:
        raw = [float(x["perplexity"]) for x in group]
        if ppl_cap > 0:
            return [min(v, ppl_cap) for v in raw]
        return raw

    def agg(group: list[dict]) -> dict[str, float]:
        ppls = ppl_vals(group)
        return {
            "n": len(group),
            "semantic_sim": float(np.mean([x["semantic_sim"] for x in group])),
            "perplexity": float(np.mean(ppls)),
            "ppl_median": float(np.median([float(x["perplexity"]) for x in group])),
            "em": float(np.mean([x["exact_match"] for x in group])),
            "f1": float(np.mean([x["token_f1"] for x in group])),
            "latency_s": float(np.mean([x["latency_s"] for x in group])),
        }

    if args.by_dataset:
        by_ds: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            ds = r.get("dataset_path") or "(unknown)"
            by_ds[ds].append(r)
        for ds in sorted(by_ds.keys()):
            print(f"\n## dataset_path={ds}")
            by_method: dict[str, list[dict]] = defaultdict(list)
            for r in by_ds[ds]:
                by_method[r["method"]].append(r)
            for m in sorted(by_method.keys()):
                a = agg(by_method[m])
                cap_note = f" (PPL cap={ppl_cap})" if ppl_cap > 0 else ""
                print(
                    f"  {m:28s}  n={a['n']:4d}  Sim={a['semantic_sim']:.3f}  "
                    f"PPL={a['perplexity']:.1f}{cap_note}  PPL_med={a['ppl_median']:.1f}  "
                    f"EM={a['em']:.3f}  F1={a['f1']:.3f}  Lat={a['latency_s']:.2f}s"
                )
        return

    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print(f"# {path}  ({len(rows)} rows)\n")
    for m in sorted(by_method.keys()):
        a = agg(by_method[m])
        cap_note = f" (PPL cap={ppl_cap})" if ppl_cap > 0 else ""
        print(
            f"{m:28s}  n={a['n']:4d}  Sim={a['semantic_sim']:.3f}  "
            f"PPL={a['perplexity']:.1f}{cap_note}  PPL_med={a['ppl_median']:.1f}  "
            f"EM={a['em']:.3f}  F1={a['f1']:.3f}  Lat={a['latency_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
