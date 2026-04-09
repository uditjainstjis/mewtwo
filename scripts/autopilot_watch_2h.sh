#!/usr/bin/env bash
# Unattended system watch: 2 hours, every 10 minutes (12 ticks).
# Logs to results/autopilot_watch_<timestamp>.log — tail -f it while away.
# Does not prompt; only writes AUTO_NEXT hints to the log.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/results/autopilot_watch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/results"
TOTAL_TICKS=12
SLEEP_SEC=600

log() { echo "[$(date "+%Y-%m-%dT%H:%M:%S%z")] $*" | tee -a "$LOG"; }

log "=== autopilot_watch_2h start ROOT=$ROOT ticks=$TOTAL_TICKS interval=${SLEEP_SEC}s ==="
log "tail -f $LOG"

auto_next() {
  log "--- AUTO_NEXT (machine hints, not interactive) ---"
  if pgrep -fl mistral_eval_md.py >/dev/null 2>&1; then
    log "  • mistral_eval_md.py still running — wait for Saved: in mistral_eval_last_run.log"
  else
    log "  • no mistral_eval_md.py process (finished or not started)"
  fi
  if pgrep -fl "run_eval_injection_hypotheses" >/dev/null 2>&1; then
    log "  • Qwen injection eval running"
  fi
  if [[ -f "$ROOT/results/mistral_track_b.json" ]]; then
    if command -v python3 >/dev/null; then
      python3 - <<'PY' 2>/dev/null | tee -a "$LOG" || true
import json, os, time
from pathlib import Path
p = Path("results/mistral_track_b.json")
if p.exists():
    d = json.loads(p.read_text())
    n = d.get("n_items", 0)
    err = d.get("n_ollama_errors", 0)
    lat = d.get("mean_latency_s", 0)
    print(f"  • mistral_track_b.json: n_items={n} n_errors={err} mean_lat={lat}")
    if n >= 40 and err == 0 and lat and lat > 0.1:
        print("  • Suggested next (when you want Qwen on same data):")
        print("    PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --extra --more \\")
        print("      --data data/multidomain_eval_external.json --output results/injection_track_b.jsonl")
PY
    fi
  fi
  if [[ -f "$ROOT/results/mistral_eval_last_run.log" ]]; then
    log "  • last 3 lines mistral_eval_last_run.log:"
    tail -n 3 "$ROOT/results/mistral_eval_last_run.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG" || true
  fi
  log "--- end AUTO_NEXT ---"
}

for i in $(seq 1 "$TOTAL_TICKS"); do
  log ""
  log "========== TICK $i/$TOTAL_TICKS =========="
  log "uptime: $(uptime 2>/dev/null || true)"
  if [[ "$(uname)" == "Darwin" ]]; then
    vm_stat 2>/dev/null | head -5 | tee -a "$LOG" || true
  fi
  log "processes (eval-related):"
  ps aux 2>/dev/null | grep -E '[m]istral_eval|[r]un_eval_injection|[o]llama serve' | tee -a "$LOG" || log "  (none matched)"
  log "disk results/:"
  ls -lt "$ROOT/results" 2>/dev/null | head -8 | tee -a "$LOG" || true
  auto_next
  if [[ "$i" -lt "$TOTAL_TICKS" ]]; then
    log "sleeping ${SLEEP_SEC}s until next tick..."
    sleep "$SLEEP_SEC"
  fi
done

log "=== autopilot_watch_2h complete ==="
