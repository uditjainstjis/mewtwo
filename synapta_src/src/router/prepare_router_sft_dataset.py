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


def serialize_assistant(row: dict) -> str:
    bullets = row.get("thinking") or []
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    tags = ",".join(f"[{expert}]" for expert in row["experts"])
    return f"<thinking>\n{bullet_text}\n</thinking>\n<experts>{tags}</experts>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/router_synthetic_routing_5000.json")
    parser.add_argument("--output-dir", default="data/router_reasoning_sft")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text())
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_valid = max(1, int(len(rows) * args.valid_ratio))
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    def write_jsonl(path: Path, data: list[dict]) -> None:
        with path.open("w") as f:
            for row in data:
                text = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{row['question']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{serialize_assistant(row)}<|im_end|>\n"
                )
                f.write(json.dumps({"text": text, "id": row["id"], "experts": row["experts"]}) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "valid.jsonl", valid_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input": str(input_path.relative_to(PROJECT_ROOT)),
                "n_train": len(train_rows),
                "n_valid": len(valid_rows),
                "system_prompt": SYSTEM_PROMPT,
            },
            indent=2,
        )
    )
    print(json.dumps({"n_train": len(train_rows), "n_valid": len(valid_rows)}, indent=2))


if __name__ == "__main__":
    main()
