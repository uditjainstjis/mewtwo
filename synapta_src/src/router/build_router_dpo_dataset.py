from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = (
    "You are the TCAR routing model. Analyze the user's request, plan the required reasoning steps, "
    "and output the exact expert tags needed to solve the task.\n"
    "Return exactly this format:\n"
    "<thinking>\n"
    "- short bullet\n"
    "- short bullet\n"
    "</thinking>\n"
    "<experts>[DOMAIN_A],[DOMAIN_B]</experts>"
)


def format_completion(thinking: list[str], experts: list[str]) -> str:
    bullets = "\n".join(f"- {t}" for t in thinking[:2])
    tags = ",".join(f"[{expert}]" for expert in experts)
    return f"<thinking>\n{bullets}\n</thinking>\n<experts>{tags}</experts>"


def collect_empirical_negative_patterns(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = []
    for row in rows:
        meta = row.get("method_meta", {})
        experts = meta.get("router_experts") or []
        if not experts:
            continue
        thinking = meta.get("router_thinking") or "wrong expert routing"
        if isinstance(thinking, str):
            bullets = [thinking.strip()]
        else:
            bullets = [str(x).strip() for x in thinking][:2]
        out.append({"thinking": bullets, "experts": list(experts)})
    return out


def build_rejected(
    row: dict,
    all_domains: list[str],
    rng: random.Random,
    empirical_patterns: list[dict],
) -> str:
    gold = list(row["experts"])
    if empirical_patterns and rng.random() < 0.35:
        pattern = rng.choice(empirical_patterns)
        return format_completion(pattern["thinking"], pattern["experts"])

    mode = rng.choice(["collapse", "swap_one", "hallucinate_pair"])
    if mode == "collapse":
        return format_completion(["one dominant domain only"], [gold[0]])
    if mode == "swap_one":
        wrong = [x for x in all_domains if x not in gold]
        experts = [gold[0], rng.choice(wrong)] if len(gold) > 1 else [rng.choice(wrong)]
        return format_completion(["second domain ignored or confused"], experts)
    wrong = rng.sample([x for x in all_domains if x not in gold], k=min(2, max(1, len(gold))))
    return format_completion(["hallucinated unrelated domain pair"], wrong)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/router_synthetic_routing_5000.json")
    parser.add_argument("--registry", default="backend/expert_registry.json")
    parser.add_argument("--output", default="data/router_reasoning_dpo.jsonl")
    parser.add_argument("--empirical-negatives", default="results/tcar_collaborative_pilot_10.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    registry_path = PROJECT_ROOT / args.registry
    output_path = PROJECT_ROOT / args.output
    empirical_path = PROJECT_ROOT / args.empirical_negatives
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text())
    registry = json.loads(registry_path.read_text())
    all_domains = list(registry.keys())
    empirical_patterns = collect_empirical_negative_patterns(empirical_path)
    rng = random.Random(args.seed)

    with output_path.open("w") as f:
        for row in rows:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            chosen = format_completion(row.get("thinking") or [], row["experts"])
            rejected = build_rejected(row, all_domains, rng, empirical_patterns)
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "experts": row["experts"],
                    }
                )
                + "\n"
            )
    print(output_path)


if __name__ == "__main__":
    main()
