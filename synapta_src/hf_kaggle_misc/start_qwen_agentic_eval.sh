#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/results/agentic_eval"
LOG_FILE="$LOG_DIR/planbench_runner.log"
mkdir -p "$LOG_DIR"

while pgrep -af "post_dpo_benchmarks.py" >/dev/null; do
  sleep 30
done

cd "$ROOT/.."
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
exec .venv/bin/python -u hf_kaggle_opensource/planbench_qwen_runner.py \
  --models qwen_0.8b \
  --ranks 0 1 2 8 128 1024 3072 \
  --task t1 \
  --domain blocksworld_3 \
  --config blocksworld_3 \
  --limit 100 >>"$LOG_FILE" 2>&1
