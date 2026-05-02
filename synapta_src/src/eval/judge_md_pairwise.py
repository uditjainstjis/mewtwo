from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from perplexity_reasoning_client import PerplexityReasoningClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_dataset(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {item["id"]: item for item in data}


def load_answers(path: Path) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[row["item_id"]][row["method"]] = row
    return grouped


def load_existing_judgments(path: Path, system_a: str, system_b: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    existing: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("system_a") == system_a and row.get("system_b") == system_b:
                existing[row["item_id"]] = row
    return existing


def build_prompt(item: dict, answer_a: str, answer_b: str, system_a: str, system_b: str) -> str:
    payload = {
        "question": item["question"],
        "domains": item.get("required_adapters") or item.get("domains") or [],
        "required_facts": item.get("required_facts", []),
        "critical_errors": item.get("critical_errors", []),
        "grading_rubric": item.get("grading_rubric", {}),
        "answer_a": answer_a,
        "answer_b": answer_b,
    }
    return (
        "You are a strict blind evaluator for a multi-domain benchmark.\n"
        "Think carefully but do not reveal chain-of-thought. Return JSON only.\n"
        "Judge correctness, coverage of required facts, and presence of critical errors.\n"
        "Do not reward style, verbosity, or model-like wording. Prefer factual accuracy.\n"
        "If both answers are weak, pick the less wrong one or 'tie'.\n"
        "Return exactly this JSON schema:\n"
        "{"
        "\"score_a\": <0-10 number>, "
        "\"score_b\": <0-10 number>, "
        "\"required_facts_covered_a\": <integer>, "
        "\"required_facts_covered_b\": <integer>, "
        "\"critical_errors_a\": <integer>, "
        "\"critical_errors_b\": <integer>, "
        "\"winner\": \"A\" | \"B\" | \"tie\", "
        "\"confidence\": <0-1 number>, "
        "\"summary\": \"<=60 words\""
        "}\n"
        f"System labels are hidden. Internally A={system_a}, B={system_b}; do not mention that in output.\n"
        f"Item:\n{json.dumps(payload, indent=2)}"
    )


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Could not extract JSON object from judge response.")
    candidate = text[start : end + 1].strip()
    attempts = [candidate]
    if candidate.startswith("{{") and candidate.endswith("}}"):
        attempts.append(candidate[1:-1].strip())
    attempts.append(re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', candidate))

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(attempt)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            pass
    raise ValueError("Could not parse JSON object from judge response.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--system-a", required=True)
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--output", default="results/md_pairwise_judgment.jsonl")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--proxy-repo", type=str, default="~/Desktop/perplexity-proxy")
    parser.add_argument("--mode", type=str, default="reasoning")
    parser.add_argument("--model", type=str, default="claude-4.6-sonnet-thinking")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = PROJECT_ROOT / answers_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_dataset(dataset_path)
    grouped = load_answers(answers_path)
    client = PerplexityReasoningClient(args.proxy_repo)

    item_ids = [item_id for item_id in sorted(grouped.keys()) if args.system_a in grouped[item_id] and args.system_b in grouped[item_id]]
    item_ids = item_ids[: args.limit]

    existing = load_existing_judgments(out_path, args.system_a, args.system_b)
    pending_item_ids = [item_id for item_id in item_ids if item_id not in existing]

    rows = list(existing.values())
    mode = "a" if existing else "w"
    if existing:
        print(f"resuming {args.system_a} vs {args.system_b}: {len(existing)} existing, {len(pending_item_ids)} pending", flush=True)

    with open(out_path, mode) as f:
        for offset, item_id in enumerate(pending_item_ids, start=1):
            item = items[item_id]
            row_a = grouped[item_id][args.system_a]
            row_b = grouped[item_id][args.system_b]
            prompt = build_prompt(item, row_a["answer"], row_b["answer"], args.system_a, args.system_b)
            raw = client.ask(prompt, mode=args.mode, model=args.model, sources=["web"])
            judged = extract_json_object(raw)
            judged.update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "item_id": item_id,
                    "system_a": args.system_a,
                    "system_b": args.system_b,
                }
            )
            rows.append(judged)
            f.write(json.dumps(judged) + "\n")
            f.flush()
            print(
                f"[{len(existing) + offset}/{len(item_ids)}] {item_id:16s} | winner={judged['winner']:3s} | "
                f"A={judged['score_a']:.1f} B={judged['score_b']:.1f} | {judged['summary']}",
                flush=True,
            )
    print(f"saved {len(rows)} judgments to {out_path}")


if __name__ == "__main__":
    main()
