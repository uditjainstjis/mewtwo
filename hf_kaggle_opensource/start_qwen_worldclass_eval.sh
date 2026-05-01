#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/.."

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "No Hugging Face token found in env."
  echo "Set one of: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_HUB_TOKEN"
  echo "Continuing anyway for public datasets."
fi

exec /home/learner/Desktop/mewtwo/.venv/bin/python -u hf_kaggle_opensource/post_dpo_benchmarks.py \
  --mode paper \
  --suite paper_supported \
  --models qwen_0.8b \
  --ranks 0 1 2 8 128 1024 3072
