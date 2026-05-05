#!/bin/bash
# Chains 3 sequential GPU jobs after BFSI training finishes:
#   1. BFSI v2 re-eval with new compliance adapter (~10 min)
#   2. Long-context RBI test (~30-45 min)
#   3. HumanEval pass@5 on n=82 (~2-3 hours)

set -e

PROJECT="/home/learner/Desktop/mewtwo"
PYTHON="$PROJECT/.venv/bin/python"
LOG_DIR="$PROJECT/logs/swarm_8h/extras"
SCRIPTS="$PROJECT/synapta_src/overnight_scripts"

cd "$PROJECT"

CHAIN_LOG="$LOG_DIR/chain.log"
echo "=== Chain orchestrator started at $(date -u) ===" > "$CHAIN_LOG"

# Wait for BFSI training to complete
TRAIN_PID=$(cat "$LOG_DIR/bfsi_train.pid")
echo "Waiting for BFSI training (PID $TRAIN_PID) to complete..." | tee -a "$CHAIN_LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    echo "[$(date -u +%H:%M:%S)] BFSI training still running..." | tee -a "$CHAIN_LOG"
done
echo "BFSI training PID $TRAIN_PID exited at $(date -u)" | tee -a "$CHAIN_LOG"

# Verify adapter was saved
if [ ! -d "$PROJECT/adapters/nemotron_30b/bfsi/best" ]; then
    echo "⚠️  BFSI adapter not saved at expected path. Skipping eval." | tee -a "$CHAIN_LOG"
else
    # Job 1: BFSI v2 with compliance adapter
    echo "=== Job 1: BFSI v2 with compliance adapter ===" | tee -a "$CHAIN_LOG"
    timeout 1800 "$PYTHON" "$SCRIPTS/run_bfsi_v2_with_compliance_adapter.py" >> "$CHAIN_LOG" 2>&1 \
        || echo "Job 1 exited (timeout or error)" | tee -a "$CHAIN_LOG"
    sleep 5
fi

# Job 2: Long-context RBI test
echo "=== Job 2: Long-context RBI test ===" | tee -a "$CHAIN_LOG"
timeout 3600 "$PYTHON" "$SCRIPTS/run_long_context_rbi.py" >> "$CHAIN_LOG" 2>&1 \
    || echo "Job 2 exited (timeout or error)" | tee -a "$CHAIN_LOG"
sleep 5

# Job 3: HumanEval pass@5 (longest)
echo "=== Job 3: HumanEval pass@5 (n=82, ~2-3h) ===" | tee -a "$CHAIN_LOG"
timeout 14400 "$PYTHON" "$SCRIPTS/run_humaneval_pass5.py" >> "$CHAIN_LOG" 2>&1 \
    || echo "Job 3 exited (timeout or error)" | tee -a "$CHAIN_LOG"

echo "=== Chain orchestrator complete at $(date -u) ===" | tee -a "$CHAIN_LOG"
