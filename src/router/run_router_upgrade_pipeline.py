from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_step(cmd: list[str]) -> None:
    print(f"\n[router-pipeline] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/router_synthetic_routing_5000.json")
    parser.add_argument("--registry", default="backend/expert_registry.json")
    parser.add_argument("--sft-data-dir", default="data/router_reasoning_sft")
    parser.add_argument("--sft-output-dir", default="router_adapters/router_reasoning_sft")
    parser.add_argument("--grpo-output-dir", default="router_adapters/router_reasoning_grpo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--sft-max-steps", type=int, default=-1)
    parser.add_argument("--grpo-max-steps", type=int, default=100)
    parser.add_argument("--learning-rate-sft", type=float, default=2e-4)
    parser.add_argument("--learning-rate-grpo", type=float, default=3e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-items", type=int, default=1000)
    parser.add_argument("--wait-for-items", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--eval-data", default="data/multidomain_eval_claude_external_v2_100.json")
    parser.add_argument("--eval-sft-output", default="results/router_accuracy_sft_5000.json")
    parser.add_argument("--eval-grpo-output", default="results/router_accuracy_grpo_5000.json")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    while True:
        if not input_path.exists():
            if not args.wait_for_items:
                raise FileNotFoundError(input_path)
            print(f"[router-pipeline] waiting for {input_path} ...", flush=True)
            time.sleep(args.poll_seconds)
            continue
        rows = json.loads(input_path.read_text())
        if len(rows) >= args.min_items:
            break
        if not args.wait_for_items:
            raise ValueError(f"Expected at least {args.min_items} routing rows, found {len(rows)}")
        print(
            f"[router-pipeline] waiting for routing data: {len(rows)} / {args.min_items}",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    run_step(
        [
            sys.executable,
            "src/router/prepare_router_sft_dataset.py",
            "--input",
            args.input,
            "--output-dir",
            args.sft_data_dir,
            "--valid-ratio",
            str(args.valid_ratio),
        ]
    )

    run_step(
        [
            sys.executable,
            "src/router/train_router_sft_manual.py",
            "--model",
            args.model,
            "--data-dir",
            args.sft_data_dir,
            "--output-dir",
            args.sft_output_dir,
            "--num-train-epochs",
            str(args.sft_epochs),
            "--max-steps",
            str(args.sft_max_steps),
            "--learning-rate",
            str(args.learning_rate_sft),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-length",
            str(args.max_length),
        ]
    )

    run_step(
        [
            sys.executable,
            "src/router/train_router_grpo.py",
            "--model",
            args.model,
            "--sft-adapter",
            args.sft_output_dir,
            "--registry",
            args.registry,
            "--data",
            args.eval_data,
            "--output-dir",
            args.grpo_output_dir,
            "--max-steps",
            str(args.grpo_max_steps),
            "--learning-rate",
            str(args.learning_rate_grpo),
            "--max-length",
            str(args.max_length),
        ]
    )

    run_step(
        [
            sys.executable,
            "src/router/eval_router_accuracy.py",
            "--data",
            args.eval_data,
            "--model",
            args.model,
            "--adapter",
            args.sft_output_dir,
            "--output",
            args.eval_sft_output,
        ]
    )

    run_step(
        [
            sys.executable,
            "src/router/eval_router_accuracy.py",
            "--data",
            args.input,
            "--model",
            args.model,
            "--adapter",
            args.grpo_output_dir,
            "--output",
            args.eval_grpo_output,
        ]
    )

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "sft_data_dir": args.sft_data_dir,
                "sft_output_dir": args.sft_output_dir,
                "grpo_output_dir": args.grpo_output_dir,
                "eval_sft_output": args.eval_sft_output,
                "eval_grpo_output": args.eval_grpo_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
