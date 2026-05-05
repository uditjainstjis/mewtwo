#!/bin/bash
# Auto-launcher: waits for HumanEval to finish (frees GPU), then trains BFSI adapter, then runs held-out eval.
# Designed to survive an idle session - just runs unattended.

set -e

PROJECT="/home/learner/Desktop/mewtwo"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/data_pipeline/auto_train_eval.log"
TRAIN_SCRIPT="$PROJECT/synapta_src/data_pipeline/07_train_bfsi_extract.py"
EVAL_SCRIPT="$PROJECT/synapta_src/data_pipeline/08_eval_bfsi_extract.py"

mkdir -p "$(dirname "$LOG")"
echo "=== Auto chain started at $(date -u) ===" > "$LOG"

# Wait for HumanEval to finish (PID 68441 - the python script, NOT the timeout wrapper 68440)
HE_PID=$(pgrep -f "run_humaneval_pass5.py" | head -1)
if [ -n "$HE_PID" ]; then
    echo "Waiting for HumanEval pass@5 PID $HE_PID to complete..." | tee -a "$LOG"
    while kill -0 "$HE_PID" 2>/dev/null; do
        sleep 60
        echo "[$(date -u +%H:%M:%S)] HumanEval still running..." | tee -a "$LOG"
    done
    echo "HumanEval PID $HE_PID exited at $(date -u)" | tee -a "$LOG"
else
    echo "No HumanEval process found - GPU is presumably free" | tee -a "$LOG"
fi

# Wait for chain orchestrator too (it owns the timeout wrapper)
CHAIN_PID=$(cat "$PROJECT/logs/swarm_8h/extras/chain.pid" 2>/dev/null || echo "")
if [ -n "$CHAIN_PID" ] && kill -0 "$CHAIN_PID" 2>/dev/null; then
    echo "Waiting for chain orchestrator PID $CHAIN_PID to complete..." | tee -a "$LOG"
    while kill -0 "$CHAIN_PID" 2>/dev/null; do
        sleep 30
    done
fi

# Cooldown - let GPU memory release fully
echo "Cooldown 30s for GPU memory release..." | tee -a "$LOG"
sleep 30

# Verify GPU is free enough
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${GPU_FREE} MiB" | tee -a "$LOG"
if [ "$GPU_FREE" -lt 25000 ]; then
    echo "ERROR: GPU only has ${GPU_FREE} MiB free, need ~25GB for training. Aborting." | tee -a "$LOG"
    exit 1
fi

# === Phase F: BFSI training ===
echo "" | tee -a "$LOG"
echo "=== Phase F: Training BFSI adapter at $(date -u) ===" | tee -a "$LOG"
cd "$PROJECT"
timeout 14400 "$PYTHON" "$TRAIN_SCRIPT" >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "Training exit code: $TRAIN_EXIT at $(date -u)" | tee -a "$LOG"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Training failed; skipping eval" | tee -a "$LOG"
    exit 1
fi

if [ ! -d "$PROJECT/adapters/nemotron_30b/bfsi_extract/best" ]; then
    echo "ERROR: BFSI adapter best/ dir not found after training. Skipping eval." | tee -a "$LOG"
    exit 1
fi

# Cooldown again
sleep 20

# === Phase G: Held-out eval ===
echo "" | tee -a "$LOG"
echo "=== Phase G: Held-out eval at $(date -u) ===" | tee -a "$LOG"
MAX_EVAL_QUESTIONS=300 timeout 7200 "$PYTHON" "$EVAL_SCRIPT" >> "$LOG" 2>&1
EVAL_EXIT=$?
echo "Eval exit code: $EVAL_EXIT at $(date -u)" | tee -a "$LOG"

echo "=== Auto chain complete at $(date -u) ===" | tee -a "$LOG"
