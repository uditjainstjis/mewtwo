from __future__ import annotations

import argparse
import ast
import itertools
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "eval"))

from perplexity_reasoning_client import PerplexityReasoningClient  # noqa: E402


WORKFLOW_TYPES = [
    "diagnosis",
    "decision_support",
    "design_review",
    "implementation_planning",
    "risk_and_compliance",
    "quantitative_reasoning",
    "cross_domain_synthesis",
    "edge_cases",
    "interpretation",
    "stakeholder_communication",
]

COGNITIVE_DEMANDS = ["shallow", "medium", "deep"]
DIFFICULTIES = ["easy", "medium", "hard"]


def extract_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Could not find JSON array in proxy response.")
    candidate = text[start : end + 1].strip()
    attempts = [
        candidate,
        re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', candidate),
    ]
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(attempt)
            if isinstance(parsed, list):
                return parsed
        except (SyntaxError, ValueError):
            pass
    raise ValueError("Could not parse routing dataset batch.")


def build_schedule(
    domains: list[str],
    total_items: int,
    seed: int,
    pair_ratio: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pair_ratio = min(max(pair_ratio, 0.0), 1.0)
    n_pairs = int(round(total_items * pair_ratio))
    n_singles = total_items - n_pairs

    pair_catalog = list(itertools.combinations(domains, 2))
    rng.shuffle(pair_catalog)
    rng.shuffle(domains)

    schedule: list[dict[str, Any]] = []
    for i in range(n_pairs):
        experts = list(pair_catalog[i % len(pair_catalog)])
        schedule.append(
            {
                "id": f"router_{i + 1:05d}",
                "experts": experts,
                "workflow_type": WORKFLOW_TYPES[i % len(WORKFLOW_TYPES)],
                "cognitive_demand": COGNITIVE_DEMANDS[(i // len(WORKFLOW_TYPES)) % len(COGNITIVE_DEMANDS)],
                "difficulty": DIFFICULTIES[(i // (len(WORKFLOW_TYPES) * len(COGNITIVE_DEMANDS))) % len(DIFFICULTIES)],
                "arity": 2,
            }
        )
    for i in range(n_singles):
        idx = n_pairs + i
        expert = domains[i % len(domains)]
        schedule.append(
            {
                "id": f"router_{idx + 1:05d}",
                "experts": [expert],
                "workflow_type": WORKFLOW_TYPES[idx % len(WORKFLOW_TYPES)],
                "cognitive_demand": COGNITIVE_DEMANDS[(idx // len(WORKFLOW_TYPES)) % len(COGNITIVE_DEMANDS)],
                "difficulty": DIFFICULTIES[(idx // (len(WORKFLOW_TYPES) * len(COGNITIVE_DEMANDS))) % len(DIFFICULTIES)],
                "arity": 1,
            }
        )
    rng.shuffle(schedule)
    for i, item in enumerate(schedule, start=1):
        item["id"] = f"router_{i:05d}"
    return schedule


def build_prompt(batch_specs: list[dict[str, Any]]) -> str:
    return (
        "You are generating gold routing supervision for a collaborative reasoning router.\n"
        "For each spec, write one realistic user question that requires exactly the supplied expert tags.\n"
        "Then provide a short planning rationale and repeat the exact gold tags.\n"
        "The rationale must be 2 bullets maximum and under 20 total words.\n"
        "Do not explain the answer to the question. Do not add any extra fields.\n"
        "Return strict JSON only as a list with this schema for every item:\n"
        "[{"
        "\"id\": \"router_00001\", "
        "\"question\": \"...\", "
        "\"thinking\": [\"bullet 1\", \"bullet 2\"], "
        "\"experts\": [\"DOMAIN_A\", \"DOMAIN_B\"]"
        "}]\n"
        "Rules:\n"
        "- Keep the supplied id unchanged.\n"
        "- Use the supplied experts exactly, in the same order.\n"
        "- The question must genuinely require the supplied experts.\n"
        "- Keep each question under 70 words.\n"
        "- No markdown fences.\n"
        f"Specs:\n{json.dumps(batch_specs, indent=2)}"
    )


def normalize_item(spec: dict[str, Any], item: dict[str, Any], model_name: str) -> dict[str, Any]:
    thinking = item.get("thinking") or []
    if isinstance(thinking, str):
        thinking = [t.strip("- ").strip() for t in thinking.splitlines() if t.strip()]
    thinking = [str(t).strip() for t in thinking if str(t).strip()][:2]
    return {
        "id": spec["id"],
        "question": str(item.get("question") or "").strip(),
        "thinking": thinking,
        "experts": list(spec["experts"]),
        "workflow_type": spec["workflow_type"],
        "cognitive_demand": spec["cognitive_demand"],
        "difficulty": spec["difficulty"],
        "arity": spec["arity"],
        "source_model": model_name,
    }


def request_batch(
    client: PerplexityReasoningClient,
    batch_specs: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
) -> list[dict[str, Any]]:
    raw = client.ask(build_prompt(batch_specs), mode=mode, model=model, sources=["web"])
    parsed = extract_json_array(raw)
    by_id = {row.get("id"): row for row in parsed if isinstance(row, dict)}
    out = []
    for spec in batch_specs:
        if spec["id"] not in by_id:
            raise ValueError(f"Missing id {spec['id']} in proxy batch.")
        out.append(normalize_item(spec, by_id[spec["id"]], model))
    return out


def generate_batch_recursive(
    client: PerplexityReasoningClient,
    batch_specs: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
) -> list[dict[str, Any]]:
    try:
        return request_batch(client, batch_specs, mode=mode, model=model)
    except Exception:
        if len(batch_specs) == 1:
            raise
        mid = len(batch_specs) // 2
        left = generate_batch_recursive(client, batch_specs[:mid], mode=mode, model=model)
        right = generate_batch_recursive(client, batch_specs[mid:], mode=mode, model=model)
        return left + right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="backend/expert_registry.json")
    parser.add_argument("--output", default="data/router_synthetic_routing_5000.json")
    parser.add_argument("--raw-dir", default="results/router_generation_raw")
    parser.add_argument("--proxy-repo", default="~/Desktop/perplexity-proxy")
    parser.add_argument("--model", default="claude-4.6-sonnet")
    parser.add_argument("--mode", default="pro")
    parser.add_argument("--total-items", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--pair-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry_path = PROJECT_ROOT / args.registry
    output_path = PROJECT_ROOT / args.output
    raw_dir = PROJECT_ROOT / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text())
    domains = list(registry.keys())
    schedule = build_schedule(domains, args.total_items, args.seed, args.pair_ratio)
    client = PerplexityReasoningClient(args.proxy_repo)

    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        existing = {row["id"]: row for row in json.loads(output_path.read_text())}

    merged_rows: list[dict[str, Any]] = list(existing.values())
    pending = [spec for spec in schedule if spec["id"] not in existing]
    print(f"schedule={len(schedule)} existing={len(existing)} pending={len(pending)} batch_size={args.batch_size}")

    for batch_start in range(0, len(pending), args.batch_size):
        batch_specs = pending[batch_start : batch_start + args.batch_size]
        batch_index = batch_start // args.batch_size
        batch_rows = generate_batch_recursive(client, batch_specs, mode=args.mode, model=args.model)
        raw_path = raw_dir / f"batch_{batch_index:04d}.json"
        raw_path.write_text(json.dumps(batch_rows, indent=2))
        existing.update({row["id"]: row for row in batch_rows})
        merged_rows = [existing[spec["id"]] for spec in schedule if spec["id"] in existing]
        output_path.write_text(json.dumps(merged_rows, indent=2))
        print(f"saved batch {batch_index + 1} / {(len(pending) + args.batch_size - 1) // args.batch_size}: total_rows={len(merged_rows)}")


if __name__ == "__main__":
    main()
