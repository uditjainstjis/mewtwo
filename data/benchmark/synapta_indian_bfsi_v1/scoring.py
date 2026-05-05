"""Reference scorer for the Synapta Indian BFSI Benchmark v1.

Three metrics:
  - exact_match(gold, pred) -> bool
  - substring_match(gold, pred, alternatives) -> bool
  - token_f1(gold, pred, stopwords=None) -> float in [0,1]

Per-question scoring:
  Each question record carries a `scoring_method` field. The scorer evaluates
  using the indicated method. For "token_f1_threshold_0.5" the question is
  counted correct iff token-F1 >= 0.5.

Significance helpers:
  - wilson_95_ci(k, n) -> (lo, hi)   95% CI on a binomial proportion
  - mcnemar_paired(a_vec, b_vec) -> p-value for a paired McNemar test
    (uses statsmodels if available, else exact binomial fallback via scipy or
     pure-python; reports the method used in the result string)

CLI:
  python scoring.py --predictions preds.jsonl --benchmark questions.jsonl
  python scoring.py --predictions preds_a.jsonl --predictions preds_b.jsonl \
                    --benchmark questions.jsonl  # paired McNemar between A and B

Predictions JSONL schema (per line):
  {"benchmark_id": "<id>", "prediction": "<model output text>"}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path

DEFAULT_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for",
    "by", "with", "as", "is", "are", "was", "were", "be", "been", "being",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.replace("₹", "rs")          # ₹ -> rs
    text = re.sub(r"\bper\s*cent\b", "%", text)  # normalize "per cent" -> "%"
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(gold: str, pred: str) -> bool:
    return _normalize(gold) == _normalize(pred)


def substring_match(gold: str, pred: str, alternatives: list[str] | None = None) -> bool:
    """True if any of {gold} ∪ alternatives appears as a substring of pred."""
    p = _normalize(pred)
    candidates = [gold] + list(alternatives or [])
    for c in candidates:
        n = _normalize(c)
        if n and n in p:
            return True
    return False


def _toks(text: str, stopwords: set[str]) -> list[str]:
    return [t for t in _normalize(text).split() if t and t not in stopwords]


def token_f1(gold: str, pred: str, stopwords: set[str] | None = None) -> float:
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    g = Counter(_toks(gold, sw))
    p = Counter(_toks(pred, sw))
    if not g or not p:
        return 0.0
    overlap = sum((g & p).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def wilson_95_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_paired(a: list[bool], b: list[bool]) -> tuple[float, str]:
    """McNemar test on paired correctness vectors. Returns (p_value, method)."""
    assert len(a) == len(b)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    n = b10 + b01

    # Try statsmodels first
    try:
        from statsmodels.stats.contingency_tables import mcnemar  # type: ignore
        table = [[0, b01], [b10, 0]]
        res = mcnemar(table, exact=(n < 25))
        return float(res.pvalue), "statsmodels.mcnemar"
    except Exception:
        pass

    # Fallback: exact binomial via scipy
    try:
        from scipy.stats import binomtest  # type: ignore
        if n == 0:
            return 1.0, "binomtest_n0"
        return float(binomtest(min(b10, b01), n, p=0.5).pvalue), "scipy.binomtest"
    except Exception:
        pass

    # Pure-python fallback: exact two-sided binomial p-value
    if n == 0:
        return 1.0, "pure_python_n0"
    k = min(b10, b01)
    # P(X<=k | n, 0.5) under symmetric two-sided
    def comb(n, r):
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    one_sided = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * one_sided)), "pure_python_binomial"


def score_question(record: dict, prediction: str) -> dict:
    method = record.get("scoring_method", "substring")
    gold = record["gold_answer"]
    alts = record.get("alternative_answers", [])
    out = {
        "benchmark_id": record["benchmark_id"],
        "method": method,
        "exact_match": exact_match(gold, prediction),
        "substring_match": substring_match(gold, prediction, alts),
        "token_f1": token_f1(gold, prediction),
    }
    if method == "exact_match":
        out["correct"] = out["exact_match"]
    elif method == "token_f1_threshold_0.5":
        out["correct"] = out["token_f1"] >= 0.5
    else:  # substring (default)
        out["correct"] = out["substring_match"]
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def evaluate(benchmark: list[dict], preds: list[dict]) -> dict:
    pred_map = {p["benchmark_id"]: p["prediction"] for p in preds}
    rows = []
    for q in benchmark:
        bid = q["benchmark_id"]
        if bid not in pred_map:
            rows.append({"benchmark_id": bid, "missing": True, "correct": False,
                         "exact_match": False, "substring_match": False, "token_f1": 0.0})
            continue
        rows.append(score_question(q, pred_map[bid]))

    n = len(rows)
    em = sum(r["exact_match"] for r in rows)
    sub = sum(r["substring_match"] for r in rows)
    f1_mean = sum(r["token_f1"] for r in rows) / max(n, 1)
    correct = sum(r["correct"] for r in rows)
    lo, hi = wilson_95_ci(correct, n)
    return {
        "n": n,
        "primary_correct": correct,
        "primary_score": correct / max(n, 1),
        "primary_95_ci": (lo, hi),
        "exact_match_rate": em / max(n, 1),
        "substring_rate": sub / max(n, 1),
        "token_f1_mean": f1_mean,
        "rows": rows,
    }


def _print_scoreboard(name: str, summary: dict) -> None:
    lo, hi = summary["primary_95_ci"]
    print(f"\n=== {name} ===")
    print(f"  N                : {summary['n']}")
    print(f"  Primary correct  : {summary['primary_correct']} / {summary['n']} "
          f"= {summary['primary_score']:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  Exact match rate : {summary['exact_match_rate']:.3f}")
    print(f"  Substring rate   : {summary['substring_rate']:.3f}")
    print(f"  Token F1 mean    : {summary['token_f1_mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", required=True,
                    help="predictions JSONL; pass twice for paired comparison")
    ap.add_argument("--benchmark", required=True, help="questions.jsonl")
    args = ap.parse_args()

    bench = load_jsonl(Path(args.benchmark))
    summaries = []
    for pp in args.predictions:
        preds = load_jsonl(Path(pp))
        summary = evaluate(bench, preds)
        summaries.append((pp, summary))
        _print_scoreboard(pp, summary)

    if len(summaries) == 2:
        a = [r["correct"] for r in summaries[0][1]["rows"]]
        b = [r["correct"] for r in summaries[1][1]["rows"]]
        p, method = mcnemar_paired(a, b)
        print(f"\n=== Paired McNemar ({summaries[0][0]} vs {summaries[1][0]}) ===")
        print(f"  p-value : {p:.4g}  (method: {method})")


if __name__ == "__main__":
    main()
