"""
Mean numeric fields per `method` for generic eval JSONL (SATS, cluster_strict, etc.).

Usage:
  python3 src/eval/aggregate_eval_jsonl.py results/sats_eval.jsonl
  python3 src/eval/aggregate_eval_jsonl.py results/cluster_strict_eval.jsonl --skip timestamp
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
        "--skip",
        type=str,
        default="timestamp,item_id,prediction_preview",
        help="Comma-separated keys to exclude from averaging.",
    )
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    skip.add("method")

    path = Path(args.jsonl)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No rows.", file=sys.stderr)
        raise SystemExit(1)

    sample = rows[0]
    numeric_keys = []
    for k, v in sample.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            numeric_keys.append(k)

    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    print(f"# {path}  ({len(rows)} rows)\n")
    for method in sorted(by_m.keys()):
        group = by_m[method]
        parts = [f"{method:22s}  n={len(group):4d}"]
        for k in sorted(numeric_keys):
            vals = [x[k] for x in group if k in x and isinstance(x[k], (int, float))]
            if not vals:
                continue
            mu = float(np.mean(vals))
            parts.append(f"  {k}={mu:.4g}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
