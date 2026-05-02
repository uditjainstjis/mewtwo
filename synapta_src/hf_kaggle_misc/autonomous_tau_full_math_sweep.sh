#!/usr/bin/env bash
set -u

ROOT="/home/learner/Desktop/mewtwo/hf_kaggle_opensource"
PY="/home/learner/Desktop/mewtwo/.venv/bin/python"
LOG="$ROOT/results/autonomous_tau_full_math_sweep.log"
RESULTS_JSON="$ROOT/results/agentic_eval/tau_bench_results.json"
USER_TAG="local_hf:qwen_0.8b:base:0"
TARGET_TOTAL=5

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG"
}

existing_total() {
  local model="$1"
  local rank="$2"
  local stage="$3"
  "$PY" - "$RESULTS_JSON" "$USER_TAG" "$model" "$rank" "$stage" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
user_tag = sys.argv[2]
model = sys.argv[3]
rank = int(sys.argv[4])
stage = sys.argv[5]

if not results_path.exists():
    print(0)
    raise SystemExit(0)

payload = json.loads(results_path.read_text())
bench = payload.get("benchmarks", {})
key = (
    f"model={model}|rank={rank}|stage={stage}|env=retail|split=test|"
    f"trials=1|agent=local_hf|user={user_tag}"
)
row = bench.get(key)
print(int((row or {}).get("total", 0) or 0))
PY
}

run_row() {
  local model="$1"
  local rank="$2"
  local stage="$3"
  local label="${model}__rank-${rank}__stage-${stage}"
  local current_total
  current_total="$(existing_total "$model" "$rank" "$stage")"
  if [[ "$current_total" -ge "$TARGET_TOTAL" ]]; then
    log "SKIP  $label :: existing total=$current_total"
    return 0
  fi

  log "START $label :: current total=$current_total"
  if "$PY" "$ROOT/tau_bench_runner.py" \
    --models "$model" \
    --ranks "$rank" \
    --stages "$stage" \
    --task-split test \
    --limit 10 \
    --num-trials 1 \
    --max-steps 12 >>"$LOG" 2>&1; then
    local updated_total
    updated_total="$(existing_total "$model" "$rank" "$stage")"
    log "DONE  $label :: updated total=$updated_total"
  else
    local code=$?
    log "FAIL  $label :: exit=$code"
  fi
}

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

log "starting full tau math sweep"
log "goal: fill 5-task retail/test coverage across the full Qwen and Nemotron math adapter pool"
log "target_total=$TARGET_TOTAL"

ROWS=(
  "qwen_0.8b 0 base"
  "qwen_0.8b 1 math_sft"
  "qwen_0.8b 1 merged_dare"
  "qwen_0.8b 1 dpo"
  "qwen_0.8b 2 math_sft"
  "qwen_0.8b 2 merged_dare"
  "qwen_0.8b 2 dpo"
  "qwen_0.8b 8 math_sft"
  "qwen_0.8b 8 merged_dare"
  "qwen_0.8b 8 dpo"
  "qwen_0.8b 128 math_sft"
  "qwen_0.8b 128 merged_dare"
  "qwen_0.8b 128 dpo"
  "qwen_0.8b 1024 math_sft"
  "qwen_0.8b 1024 merged_dare"
  "qwen_0.8b 1024 dpo"
  "qwen_0.8b 3072 math_sft"
  "qwen_0.8b 3072 merged_dare"
  "qwen_0.8b 3072 dpo"
  "nemotron_4b 0 base"
  "nemotron_4b 1 math_sft"
  "nemotron_4b 1 merged_dare"
  "nemotron_4b 1 dpo"
  "nemotron_4b 2 math_sft"
  "nemotron_4b 2 merged_dare"
  "nemotron_4b 2 dpo"
  "nemotron_4b 8 math_sft"
  "nemotron_4b 8 merged_dare"
  "nemotron_4b 8 dpo"
  "nemotron_4b 128 math_sft"
  "nemotron_4b 128 merged_dare"
  "nemotron_4b 128 dpo"
  "nemotron_4b 1024 math_sft"
  "nemotron_4b 1024 merged_dare"
  "nemotron_4b 1024 dpo"
  "nemotron_4b 3072 math_sft"
  "nemotron_4b 3072 merged_dare"
  "nemotron_4b 3072 dpo"
)

for row in "${ROWS[@]}"; do
  # shellcheck disable=SC2086
  run_row $row
done

log "refreshing tau cross-family summary"
if "$PY" "$ROOT/summarize_tau_cross_family.py" >>"$LOG" 2>&1; then
  log "summary refreshed"
else
  code=$?
  log "summary refresh failed :: exit=$code"
fi

log "refreshing qwen geometry report"
if "$PY" "$ROOT/geometry_behavior_analysis.py" >>"$LOG" 2>&1; then
  log "geometry report refreshed"
else
  code=$?
  log "geometry refresh failed :: exit=$code"
fi

log "full tau math sweep completed"
