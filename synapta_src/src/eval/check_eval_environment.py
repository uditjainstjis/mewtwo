#!/usr/bin/env python3
"""
Quick checks before long eval runs: data files, optional Ollama, optional MLX.

Usage:
  python3 src/eval/check_eval_environment.py
  python3 src/eval/check_eval_environment.py --require-ollama
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def bad(msg: str) -> None:
    print(f"  !!  {msg}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-ollama",
        action="store_true",
        help="Exit 1 if Ollama is not reachable or mistral:7b is missing.",
    )
    parser.add_argument(
        "--require-mlx",
        action="store_true",
        help="Exit 1 if mlx cannot be imported.",
    )
    args = parser.parse_args()
    code = 0

    print(f"PROJECT_ROOT={PROJECT_ROOT}\n")

    for rel in (
        "data/multidomain_eval_v2.json",
        "data/multidomain_eval_external.example.json",
        "backend/expert_registry.json",
    ):
        p = PROJECT_ROOT / rel
        if p.is_file():
            ok(rel)
        else:
            bad(f"missing {rel}")
            code = 1

    v2 = PROJECT_ROOT / "data" / "multidomain_eval_v2.json"
    if v2.is_file():
        try:
            n = len(json.loads(v2.read_text()))
            ok(f"multidomain_eval_v2.json parses ({n} rows)")
        except Exception as e:
            bad(f"multidomain_eval_v2.json parse: {e}")
            code = 1

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        import requests

        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.ok:
            names = {m.get("name", "") for m in r.json().get("models") or []}
            ok(f"Ollama reachable at {host}")
            if "mistral:7b" in names:
                ok("model mistral:7b present")
            else:
                bad("mistral:7b not in ollama list (optional for Qwen-only evals)")
                if args.require_ollama:
                    code = 1
        else:
            bad(f"Ollama HTTP {r.status_code}")
            if args.require_ollama:
                code = 1
    except Exception as e:
        bad(f"Ollama check: {e}")
        if args.require_ollama:
            code = 1

    mlx_spec = importlib.util.find_spec("mlx")
    if mlx_spec is not None:
        ok("mlx importable")
    else:
        bad("mlx not installed (needed for --real GPU evals)")
        if args.require_mlx:
            code = 1

    sem_spec = importlib.util.find_spec("sentence_transformers")
    if sem_spec is not None:
        ok("sentence_transformers importable")
    else:
        bad("sentence_transformers missing (needed for semantic_sim in several evals)")

    print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
