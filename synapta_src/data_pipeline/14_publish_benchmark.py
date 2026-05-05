#!/usr/bin/env python3
"""Publish the Synapta Indian BFSI Benchmark v1 to the HuggingFace Hub.

What it does
------------
1. Validates the local benchmark bundle (README + LICENSE + questions.jsonl
   + scoring.py) so that the dataset card will render and the JSONL is
   parseable.
2. Idempotently creates the dataset repository on the Hub (no error if
   it already exists).
3. Uploads the entire benchmark directory as the repository's working
   tree.

What it does NOT do
-------------------
- It does not fine-tune anything.
- It does not modify questions.jsonl in any way.
- It does not push automatically; you must invoke this script with a
  valid HF token in the environment.

Setup
-----
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx        # or run: huggingface-cli login

Usage
-----
    python synapta_src/data_pipeline/14_publish_benchmark.py
    python synapta_src/data_pipeline/14_publish_benchmark.py \\
        --repo-id synapta/indian-bfsi-bench-v1 \\
        --private             # for staging dry-runs
    python synapta_src/data_pipeline/14_publish_benchmark.py --dry-run

Replace `synapta` with your own HF org/user namespace if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BENCH_DIR = PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1"

REQUIRED_FILES = ["README.md", "LICENSE.md", "questions.jsonl", "scoring.py"]


def validate_bundle(bench_dir: Path) -> None:
    """Sanity-check the bundle before we ship it."""
    if not bench_dir.is_dir():
        raise SystemExit(f"benchmark directory not found: {bench_dir}")

    missing = [f for f in REQUIRED_FILES if not (bench_dir / f).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    # Validate every JSONL line parses + carries a benchmark_id.
    questions_path = bench_dir / "questions.jsonl"
    n_rows = 0
    seen_ids: set[str] = set()
    with questions_path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"questions.jsonl line {i}: {e}")
            bid = rec.get("benchmark_id")
            if not bid:
                raise SystemExit(f"questions.jsonl line {i}: missing benchmark_id")
            if bid in seen_ids:
                raise SystemExit(f"duplicate benchmark_id: {bid}")
            seen_ids.add(bid)
            n_rows += 1
    print(f"[ok] questions.jsonl: {n_rows} rows, all parsed, no duplicate ids")

    # Lightweight YAML-frontmatter check on README.
    readme_text = (bench_dir / "README.md").read_text(encoding="utf-8")
    if not readme_text.startswith("---"):
        raise SystemExit("README.md is missing the YAML frontmatter HF requires")
    print("[ok] README.md has YAML frontmatter")

    print(f"[ok] bundle validated: {bench_dir}")


def push_to_hub(
    bench_dir: Path,
    repo_id: str,
    private: bool,
    token: str | None,
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

    api = HfApi(token=token)

    print(f"[hub] ensuring dataset repo exists: {repo_id} (private={private})")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    print(f"[hub] uploading folder {bench_dir} -> {repo_id}")
    commit_info = api.upload_folder(
        folder_path=str(bench_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        # ignore caches and editor cruft; ship the artefact only
        ignore_patterns=[
            "__pycache__/*", "*.pyc", ".DS_Store",
            ".ipynb_checkpoints/*", "*.swp",
        ],
    )
    print(f"[hub] commit: {commit_info}")
    print(f"[hub] dataset live at: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-id", default="synapta/indian-bfsi-bench-v1",
        help="HF dataset repo id (default: synapta/indian-bfsi-bench-v1)",
    )
    ap.add_argument(
        "--bench-dir", default=str(BENCH_DIR),
        help=f"local benchmark directory (default: {BENCH_DIR})",
    )
    ap.add_argument(
        "--private", action="store_true",
        help="create the repo as private (useful for staging)",
    )
    ap.add_argument(
        "--token", default=None,
        help="HF token (defaults to HF_TOKEN env var or huggingface-cli login)",
    )
    ap.add_argument(
        "--commit-message",
        default="Publish Synapta Indian BFSI Benchmark v1 (CC-BY-SA-4.0)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="validate the bundle and exit, do not contact the Hub",
    )
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    validate_bundle(bench_dir)

    if args.dry_run:
        print("[dry-run] bundle is valid; not contacting the Hub")
        return

    push_to_hub(
        bench_dir=bench_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
