#!/usr/bin/env bash
set -u

ROOT="/home/learner/Desktop/mewtwo/hf_kaggle_opensource"
PY="/home/learner/Desktop/mewtwo/.venv/bin/python"
LOG="$ROOT/results/autonomous_tau_generalization.log"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG"
}

run_row() {
  local label="$1"
  shift
  log "START $label :: $*"
  if "$@" >>"$LOG" 2>&1; then
    log "DONE  $label"
  else
    local code=$?
    log "FAIL  $label :: exit=$code"
  fi
}

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

log "starting expanded tau generalization slice"
log "goal: keep the GPU busy on a richer multi-row retail test sweep"

COMMON_ARGS=(
  --task-split test
  --limit 10
  --num-trials 1
  --max-steps 12
)

# Weak Qwen anchors.
run_row "qwen_base_rank0_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models qwen_0.8b \
  --ranks 0 \
  --stages base \
  "${COMMON_ARGS[@]}"

run_row "qwen_math_sft_rank1_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models qwen_0.8b \
  --ranks 1 \
  --stages math_sft \
  "${COMMON_ARGS[@]}"

run_row "qwen_math_sft_rank2_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models qwen_0.8b \
  --ranks 2 \
  --stages math_sft \
  "${COMMON_ARGS[@]}"

# Stronger Qwen anchors.
run_row "qwen_dpo_rank1_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models qwen_0.8b \
  --ranks 1 \
  --stages dpo \
  "${COMMON_ARGS[@]}"

run_row "qwen_dpo_rank2_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models qwen_0.8b \
  --ranks 2 \
  --stages dpo \
  "${COMMON_ARGS[@]}"

# Flat/strong Nemotron anchor.
run_row "nemotron_dpo_rank128_limit10" \
  "$PY" "$ROOT/tau_bench_runner.py" \
  --models nemotron_4b \
  --ranks 128 \
  --stages dpo \
  "${COMMON_ARGS[@]}"

log "refreshing tau cross-family summary"
if "$PY" "$ROOT/summarize_tau_cross_family.py" >>"$LOG" 2>&1; then
  log "summary refreshed"
else
  code=$?
  log "summary refresh failed :: exit=$code"
fi

log "expanded tau generalization slice completed"
