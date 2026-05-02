import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATIONS_DIR = ROOT / "results" / "benchmark_generations"


def parse_args():
    parser = argparse.ArgumentParser(description="Paired benchmark comparison from generation JSONL files.")
    parser.add_argument("--left", required=True, type=Path, help="Left generation JSONL file.")
    parser.add_argument("--right", required=True, type=Path, help="Right generation JSONL file.")
    return parser.parse_args()


def key_for_row(row: dict):
    if "task_id" in row:
        return ("task_id", row["task_id"])
    if "question" in row:
        return ("question", row["question"])
    if "entry_point" in row:
        return ("entry_point", row["entry_point"])
    return ("raw", json.dumps(row, sort_keys=True))


def load_rows(path: Path):
    rows = {}
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[key_for_row(row)] = row
    return rows


def binom_two_sided_pvalue(k: int, n: int, p: float = 0.5):
    if n == 0:
        return 1.0
    observed = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        if prob <= observed + 1e-15:
            total += prob
    return min(1.0, total)


def main():
    args = parse_args()
    left_rows = load_rows(args.left)
    right_rows = load_rows(args.right)
    shared = sorted(set(left_rows) & set(right_rows))

    both_correct = 0
    left_only = 0
    right_only = 0
    both_wrong = 0

    for key in shared:
        lc = bool(left_rows[key].get("correct"))
        rc = bool(right_rows[key].get("correct"))
        if lc and rc:
            both_correct += 1
        elif lc and not rc:
            left_only += 1
        elif not lc and rc:
            right_only += 1
        else:
            both_wrong += 1

    discordant = left_only + right_only
    p_value = binom_two_sided_pvalue(min(left_only, right_only), discordant, p=0.5)

    print(f"left:  {args.left}")
    print(f"right: {args.right}")
    print(f"shared_items: {len(shared)}")
    print(f"both_correct: {both_correct}")
    print(f"left_only_correct: {left_only}")
    print(f"right_only_correct: {right_only}")
    print(f"both_wrong: {both_wrong}")
    print(f"left_accuracy: {((both_correct + left_only) / len(shared)):.4f}" if shared else "left_accuracy: -")
    print(f"right_accuracy: {((both_correct + right_only) / len(shared)):.4f}" if shared else "right_accuracy: -")
    print(f"discordant_pairs: {discordant}")
    print(f"exact_mcnemar_pvalue: {p_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
