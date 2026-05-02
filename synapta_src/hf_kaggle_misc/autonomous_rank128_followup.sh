#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/learner/Desktop/mewtwo/hf_kaggle_opensource"
PY="/home/learner/Desktop/mewtwo/.venv/bin/python"
LOG="$ROOT/results/autonomous_followup_runner.log"
TRAIN_PATTERN='advanced_training_pipeline.py --models nemotron_4b --ranks 128'
ARTIFACT_DIR="$ROOT/outputs/nemotron_4b_math_DPO_rank128"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG"
}

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

log "follow-up watcher started"
log "waiting for training pattern to disappear: $TRAIN_PATTERN"

while pgrep -f "$TRAIN_PATTERN" >/dev/null 2>&1; do
  sleep 60
done

log "training process no longer active"

if [[ ! -f "$ARTIFACT_DIR/adapter_model.safetensors" || ! -f "$ARTIFACT_DIR/adapter_config.json" ]]; then
  log "rank128 dpo artifact still missing; skipping tau follow-up"
  exit 1
fi

log "rank128 dpo artifact exists; running tau test-task row"
"$PY" "$ROOT/tau_bench_runner.py" \
  --models nemotron_4b \
  --ranks 128 \
  --stages dpo \
  --task-split test \
  --limit 1 \
  --num-trials 1 \
  --max-steps 12 >>"$LOG" 2>&1

log "tau row finished; refreshing cross-family summary"
"$PY" "$ROOT/summarize_tau_cross_family.py" >>"$LOG" 2>&1

log "follow-up watcher completed"
