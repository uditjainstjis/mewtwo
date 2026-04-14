#!/usr/bin/env python3
"""
Score predictions against a rubric-rich external MD dataset.

The goal is to provide a stronger accuracy-oriented layer than whole-answer semantic
similarity. This is still imperfect, but it is more defensible when the dataset includes
structured rubric fields.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_phrase(text: str, phrase: str) -> bool:
    return norm(phrase) in norm(text)


def score_item(pred: str, rubric: dict) -> dict[str, float]:
    pred_n = norm(pred)
    must_all = rubric.get("must_include_all") or []
    must_any = rubric.get("must_include_any") or []
    must_not = rubric.get("must_not_include") or []
    regex_targets = rubric.get("regex_targets") or []
    numeric_targets = rubric.get("numeric_targets") or []

    all_hits = sum(1 for p in must_all if contains_phrase(pred_n, p))
    any_hits = 0
    for group in must_any:
        if any(contains_phrase(pred_n, variant) for variant in group):
            any_hits += 1
    not_hits = sum(1 for p in must_not if contains_phrase(pred_n, p))
    regex_hits = sum(1 for pat in regex_targets if re.search(pat, pred, re.I))

    numeric_hits = 0
    for target in numeric_targets:
        value = str(target.get("value", "")).strip()
        tol = float(target.get("tolerance", 0.0) or 0.0)
        if not value:
            continue
        if tol == 0.0:
            numeric_hits += 1 if contains_phrase(pred, value) else 0
            continue
        try:
            desired = float(value)
        except ValueError:
            numeric_hits += 1 if contains_phrase(pred, value) else 0
            continue
        found = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", pred)]
        if any(abs(x - desired) <= tol for x in found):
            numeric_hits += 1

    denom_pos = len(must_all) + len(must_any) + len(regex_targets) + len(numeric_targets)
    coverage = (
        (all_hits + any_hits + regex_hits + numeric_hits) / denom_pos if denom_pos > 0 else 0.0
    )
    critical_error_rate = (not_hits / len(must_not)) if must_not else 0.0
    pass_flag = 1.0 if coverage >= 0.8 and critical_error_rate == 0.0 else 0.0

    return {
        "coverage": coverage,
        "critical_error_rate": critical_error_rate,
        "pass_rate": pass_flag,
        "must_all_hit_rate": (all_hits / len(must_all)) if must_all else 0.0,
        "must_any_hit_rate": (any_hits / len(must_any)) if must_any else 0.0,
        "regex_hit_rate": (regex_hits / len(regex_targets)) if regex_targets else 0.0,
        "numeric_hit_rate": (numeric_hits / len(numeric_targets)) if numeric_targets else 0.0,
    }


def load_dataset(path: Path) -> dict[str, dict]:
    with open(path) as f:
        items = json.load(f)
    return {str(item["id"]): item for item in items}


def load_predictions(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    raise ValueError(f"Unsupported predictions format: {path}")


def extract_prediction_text(row: dict) -> str:
    return (
        row.get("answer")
        or row.get("prediction_text")
        or row.get("response_text")
        or row.get("prediction_preview")
        or row.get("generated_text_preview")
        or ""
    )


def extract_method(row: dict) -> str:
    return row.get("method") or row.get("model") or "unknown"


def extract_item_id(row: dict) -> str:
    return str(row.get("item_id") or row.get("id") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions against a rubric-rich MD dataset.")
    parser.add_argument("dataset", type=str, help="External dataset JSON with rubric fields.")
    parser.add_argument("predictions", type=str, help="Predictions JSON or JSONL.")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    rows = load_predictions(Path(args.predictions))

    scored = []
    for row in rows:
        item_id = extract_item_id(row)
        if not item_id or item_id not in dataset:
            continue
        pred = extract_prediction_text(row)
        rubric = dataset[item_id].get("rubric") or {}
        metrics = score_item(pred, rubric)
        scored.append(
            {
                "item_id": item_id,
                "method": extract_method(row),
                **metrics,
            }
        )

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_method[row["method"]].append(row)

    print(f"# scored rows: {len(scored)}")
    for method in sorted(by_method):
        group = by_method[method]
        print(
            f"{method:28s} "
            f"n={len(group):4d} "
            f"pass={np.mean([x['pass_rate'] for x in group]):.3f} "
            f"coverage={np.mean([x['coverage'] for x in group]):.3f} "
            f"critical_err={np.mean([x['critical_error_rate'] for x in group]):.3f} "
            f"must_all={np.mean([x['must_all_hit_rate'] for x in group]):.3f} "
            f"must_any={np.mean([x['must_any_hit_rate'] for x in group]):.3f} "
            f"regex={np.mean([x['regex_hit_rate'] for x in group]):.3f} "
            f"numeric={np.mean([x['numeric_hit_rate'] for x in group]):.3f}"
        )


if __name__ == "__main__":
    main()
