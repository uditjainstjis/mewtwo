#!/usr/bin/env bash
set -u

ROOT="/home/learner/Desktop/mewtwo/hf_kaggle_opensource"
PY="/home/learner/Desktop/mewtwo/.venv/bin/python"
LOG_DIR="$ROOT/results/tau_qwen_intervention_wave2"
MASTER_LOG="$LOG_DIR/master.log"
RESULTS_JSON="$ROOT/results/agentic_eval/tau_bench_results.json"
USER_TAG="local_hf:qwen_0.8b:base:0"
TARGET_TOTAL=10
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

mlog() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$MASTER_LOG"
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

run_custom() {
  local worker="$1"
  local model="$2"
  local rank="$3"
  local stage="$4"
  local adapter_path="$5"
  local log_file="$LOG_DIR/${worker}.log"
  while [[ ! -f "$adapter_path/adapter_model.safetensors" ]]; do
    printf '[%s] WAIT  custom %s %s %s :: adapter not ready at %s\n' "$(ts)" "$model" "$rank" "$stage" "$adapter_path" >>"$log_file"
    sleep 30
  done
  local current_total
  current_total="$(existing_total "$model" "$rank" "$stage")"
  if [[ "$current_total" -ge "$TARGET_TOTAL" ]]; then
    printf '[%s] SKIP custom %s %s %s :: total=%s\n' "$(ts)" "$model" "$rank" "$stage" "$current_total" >>"$log_file"
    return 0
  fi
  printf '[%s] START custom %s %s %s :: total=%s path=%s\n' "$(ts)" "$model" "$rank" "$stage" "$current_total" "$adapter_path" >>"$log_file"
  if "$PY" "$ROOT/tau_bench_runner.py" \
    --models "$model" \
    --ranks "$rank" \
    --stages "$stage" \
    --custom-adapter-path "$adapter_path" \
    --custom-stage-label "$stage" \
    --task-split test \
    --limit 10 \
    --num-trials 1 \
    --max-steps 12 >>"$log_file" 2>&1; then
    local updated_total
    updated_total="$(existing_total "$model" "$rank" "$stage")"
    printf '[%s] DONE  custom %s %s %s :: total=%s\n' "$(ts)" "$model" "$rank" "$stage" "$updated_total" >>"$log_file"
  else
    local code=$?
    printf '[%s] FAIL  custom %s %s %s :: exit=%s\n' "$(ts)" "$model" "$rank" "$stage" "$code" >>"$log_file"
  fi
}

worker_loop() {
  local worker="$1"
  shift
  local log_file="$LOG_DIR/${worker}.log"
  : >"$log_file"
  printf '[%s] worker %s starting\n' "$(ts)" "$worker" >>"$log_file"
  while [[ "$#" -gt 0 ]]; do
    run_custom "$worker" "$1" "$2" "$3" "$4"
    shift 4
  done
  printf '[%s] worker %s finished\n' "$(ts)" "$worker" >>"$log_file"
}

cd "$ROOT"
mkdir -p "$LOG_DIR"
: >"$MASTER_LOG"

mlog "waiting for current tau workers to drain before intervention wave2"
while pgrep -f 'tau_bench_runner.py' >/dev/null 2>&1; do
  mlog "tau workers still active; sleeping 60s"
  sleep 60
done

mlog "starting qwen intervention wave2"

worker_loop workerA \
  qwen_0.8b 128 math_sft_trunc2 "$ROOT/outputs/qwen_0.8b_math_SFT_rank128_trunc2" &
pidA=$!

worker_loop workerB \
  qwen_0.8b 3072 math_sft_trunc2 "$ROOT/outputs/qwen_0.8b_math_SFT_rank3072_trunc2" &
pidB=$!

worker_loop workerC \
  qwen_0.8b 1024 merged_dare_trunc2 "$ROOT/outputs/qwen_0.8b_merged_DARE_rank1024_trunc2" &
pidC=$!

worker_loop workerD \
  qwen_0.8b 1024 dpo_trunc2 "$ROOT/outputs/qwen_0.8b_math_DPO_rank1024_trunc2" &
pidD=$!

mlog "launched workerA pid=$pidA"
mlog "launched workerB pid=$pidB"
mlog "launched workerC pid=$pidC"
mlog "launched workerD pid=$pidD"

wait "$pidA" "$pidB" "$pidC" "$pidD"
mlog "qwen intervention wave2 completed"
