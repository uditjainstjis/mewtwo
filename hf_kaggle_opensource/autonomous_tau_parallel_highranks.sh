#!/usr/bin/env bash
set -u

ROOT="/home/learner/Desktop/mewtwo/hf_kaggle_opensource"
PY="/home/learner/Desktop/mewtwo/.venv/bin/python"
LOG_DIR="$ROOT/results/tau_parallel_highranks"
MASTER_LOG="$LOG_DIR/master.log"
RESULTS_JSON="$ROOT/results/agentic_eval/tau_bench_results.json"
USER_TAG="local_hf:qwen_0.8b:base:0"
TARGET_TOTAL=5

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

mlog() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$MASTER_LOG"
}

gpu_mem_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1
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
  local worker="$1"
  local model="$2"
  local rank="$3"
  local stage="$4"
  local log_file="$LOG_DIR/${worker}.log"
  local current_total
  current_total="$(existing_total "$model" "$rank" "$stage")"
  if [[ "$current_total" -ge "$TARGET_TOTAL" ]]; then
    printf '[%s] SKIP  %s %s %s :: total=%s\n' "$(ts)" "$model" "$rank" "$stage" "$current_total" >>"$log_file"
    return 0
  fi

  printf '[%s] START %s %s %s :: total=%s\n' "$(ts)" "$model" "$rank" "$stage" "$current_total" >>"$log_file"
  if "$PY" "$ROOT/tau_bench_runner.py" \
    --models "$model" \
    --ranks "$rank" \
    --stages "$stage" \
    --task-split test \
    --limit 10 \
    --num-trials 1 \
    --max-steps 12 >>"$log_file" 2>&1; then
    local updated_total
    updated_total="$(existing_total "$model" "$rank" "$stage")"
    printf '[%s] DONE  %s %s %s :: total=%s\n' "$(ts)" "$model" "$rank" "$stage" "$updated_total" >>"$log_file"
  else
    local code=$?
    printf '[%s] FAIL  %s %s %s :: exit=%s\n' "$(ts)" "$model" "$rank" "$stage" "$code" >>"$log_file"
  fi
}

worker_loop() {
  local worker="$1"
  shift
  local log_file="$LOG_DIR/${worker}.log"
  : >"$log_file"
  printf '[%s] worker %s starting\n' "$(ts)" "$worker" >>"$log_file"
  while [[ "$#" -gt 0 ]]; do
    local model="$1"
    local rank="$2"
    local stage="$3"
    shift 3
    run_row "$worker" "$model" "$rank" "$stage"
  done
  printf '[%s] worker %s finished\n' "$(ts)" "$worker" >>"$log_file"
}

cd "$ROOT"
mkdir -p "$LOG_DIR"
: >"$MASTER_LOG"

mlog "waiting for headroom before high-rank parallel sweep"
while true; do
  used="$(gpu_mem_used_mib)"
  mlog "gpu_mem_used_mib=$used"
  if [[ "${used:-99999}" -lt 18000 ]]; then
    break
  fi
  sleep 60
done

mlog "starting high-rank tau parallel fill"

worker_loop workerA \
  qwen_0.8b 1024 math_sft \
  qwen_0.8b 1024 merged_dare \
  qwen_0.8b 1024 dpo \
  qwen_0.8b 3072 math_sft \
  qwen_0.8b 3072 merged_dare \
  qwen_0.8b 3072 dpo &
pidA=$!

worker_loop workerB \
  nemotron_4b 1024 math_sft \
  nemotron_4b 1024 merged_dare \
  nemotron_4b 1024 dpo \
  nemotron_4b 3072 math_sft \
  nemotron_4b 3072 merged_dare \
  nemotron_4b 3072 dpo &
pidB=$!

mlog "launched workerA pid=$pidA"
mlog "launched workerB pid=$pidB"

wait "$pidA" "$pidB"

mlog "refreshing cross-family summary"
if "$PY" "$ROOT/summarize_tau_cross_family.py" >>"$MASTER_LOG" 2>&1; then
  mlog "summary refreshed"
else
  code=$?
  mlog "summary refresh failed :: exit=$code"
fi

mlog "refreshing geometry report"
if "$PY" "$ROOT/geometry_behavior_analysis.py" >>"$MASTER_LOG" 2>&1; then
  mlog "geometry report refreshed"
else
  code=$?
  mlog "geometry refresh failed :: exit=$code"
fi

mlog "high-rank tau parallel fill completed"
