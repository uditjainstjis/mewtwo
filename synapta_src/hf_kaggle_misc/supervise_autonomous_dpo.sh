#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

PIPELINE_PY="$ROOT/autonomous_dpo_pipeline.py"
SUPERVISOR_LOG="$ROOT/autonomous_supervisor.log"
PID_FILE="$ROOT/autonomous_supervisor.pid"
STATUS_JSON="$ROOT/autonomous_status.json"

mkdir -p "$ROOT"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Supervisor already running with pid=$OLD_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

pending_count() {
  "$PYTHON_BIN" "$PIPELINE_PY" --status_json > "$STATUS_JSON"
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pending_count"])' "$STATUS_JSON"
}

log "Autonomous DPO supervisor starting"
log "Project root: $PROJECT_ROOT"
log "Python: $PYTHON_BIN"

restart_count=0
max_restarts=20

while true; do
  current_pending="$(pending_count)"
  if [[ "$current_pending" == "0" ]]; then
    log "No pending DPO targets remain. Supervisor exiting."
    exit 0
  fi

  log "Pending targets remaining: $current_pending"
  log "Launching autonomous pipeline"
  set +e
  "$PYTHON_BIN" -u "$PIPELINE_PY" --hours 24 --eval_mode autonomous >> "$SUPERVISOR_LOG" 2>&1
  rc=$?
  set -e

  current_pending="$(pending_count)"
  if [[ "$current_pending" == "0" && "$rc" == "0" ]]; then
    log "Pipeline completed with no pending targets. Supervisor exiting."
    exit 0
  fi

  restart_count=$((restart_count + 1))
  log "Pipeline exited with rc=$rc; pending=$current_pending; restart_count=$restart_count"
  if (( restart_count >= max_restarts )); then
    log "Max restart count reached. Supervisor exiting for safety."
    exit 1
  fi
  sleep 20
done
