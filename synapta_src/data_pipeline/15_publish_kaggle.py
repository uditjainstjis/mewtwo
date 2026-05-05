#!/usr/bin/env python3
"""Publish the Synapta Indian BFSI Benchmark v1 to Kaggle Datasets.

What it does
------------
1. Validates the local benchmark bundle and the Kaggle
   dataset-metadata.json file.
2. Shells out to the `kaggle` CLI to create (first time) or version
   (subsequent runs) the dataset.

What it does NOT do
-------------------
- It does not modify questions.jsonl.
- It does not push automatically; you must invoke this script with
  Kaggle credentials configured.

Setup
-----
    pip install kaggle
    # Place your kaggle.json API token at ~/.kaggle/kaggle.json (chmod 600).
    # See: https://www.kaggle.com/docs/api#authentication

Then edit `dataset-metadata.json` so that the `id` field uses YOUR
Kaggle username/team rather than the placeholder. The slug after the
slash is the dataset slug used in the URL.

Usage
-----
    # First publish (creates the dataset on Kaggle):
    python synapta_src/data_pipeline/15_publish_kaggle.py --create

    # Subsequent updates (publishes a new version):
    python synapta_src/data_pipeline/15_publish_kaggle.py --version \\
        --version-notes "fix typo in alternative_answers for sib1-042"

    # Validate locally without contacting Kaggle:
    python synapta_src/data_pipeline/15_publish_kaggle.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BENCH_DIR = PROJECT / "data" / "benchmark" / "synapta_indian_bfsi_v1"

REQUIRED_FILES = [
    "README.md", "LICENSE.md", "questions.jsonl", "scoring.py",
    "dataset-metadata.json",
]


def validate_bundle(bench_dir: Path) -> dict:
    if not bench_dir.is_dir():
        raise SystemExit(f"benchmark directory not found: {bench_dir}")

    missing = [f for f in REQUIRED_FILES if not (bench_dir / f).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    meta_path = bench_dir / "dataset-metadata.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"dataset-metadata.json is not valid JSON: {e}")

    for required_field in ("title", "id", "licenses", "description"):
        if required_field not in meta:
            raise SystemExit(
                f"dataset-metadata.json is missing required field: {required_field}"
            )

    if "/" not in meta["id"]:
        raise SystemExit(
            "dataset-metadata.json `id` must be in the form '<owner>/<slug>'"
        )

    # JSONL sanity check
    n_rows = 0
    with (bench_dir / "questions.jsonl").open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"questions.jsonl line {i}: {e}")
            n_rows += 1
    print(f"[ok] questions.jsonl: {n_rows} rows parsed")
    print(f"[ok] dataset-metadata.json: id={meta['id']}, license={meta['licenses']}")
    return meta


def run_kaggle(args: list[str]) -> None:
    if not shutil.which("kaggle"):
        raise SystemExit(
            "the `kaggle` CLI is not on PATH. Install it: pip install kaggle"
        )
    print(f"[kaggle] $ kaggle {' '.join(args)}")
    res = subprocess.run(["kaggle", *args])
    if res.returncode != 0:
        raise SystemExit(f"kaggle exited with code {res.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench-dir", default=str(BENCH_DIR),
        help=f"local benchmark directory (default: {BENCH_DIR})",
    )
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--create", action="store_true",
        help="first-time publish (creates the dataset on Kaggle)",
    )
    grp.add_argument(
        "--version", action="store_true",
        help="publish a new version of an existing Kaggle dataset",
    )
    ap.add_argument(
        "--version-notes",
        default="Update Synapta Indian BFSI Benchmark v1",
        help="version notes for `kaggle datasets version`",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="validate locally and exit without contacting Kaggle",
    )
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    validate_bundle(bench_dir)

    if args.dry_run:
        print("[dry-run] bundle is valid; not contacting Kaggle")
        return

    if args.create:
        # `--dir-mode zip` is reasonable for a small textual dataset
        run_kaggle(["datasets", "create", "-p", str(bench_dir), "--dir-mode", "zip"])
    elif args.version:
        run_kaggle([
            "datasets", "version",
            "-p", str(bench_dir),
            "-m", args.version_notes,
            "--dir-mode", "zip",
        ])
    else:
        sys.exit(
            "specify --create (first publish) or --version (subsequent updates), "
            "or --dry-run to only validate locally"
        )


if __name__ == "__main__":
    main()
