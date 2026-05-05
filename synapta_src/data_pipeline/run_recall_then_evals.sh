#!/bin/bash
# Auto-chain: waits for bfsi_recall training, runs recall eval, then runs the missing FG mode of bfsi_extract eval.
# Designed to run unattended.

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT="/home/learner/Desktop/mewtwo"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/logs/data_pipeline/recall_chain.log"
mkdir -p "$(dirname "$LOG")"
echo "=== Recall chain started at $(date -u) ===" > "$LOG"

# 1. Wait for recall training PID to exit
TRAIN_PID=$(cat "$PROJECT/logs/data_pipeline/recall_train.pid" 2>/dev/null || echo "")
if [ -n "$TRAIN_PID" ]; then
    echo "Waiting for recall training PID $TRAIN_PID..." | tee -a "$LOG"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
        echo "[$(date -u +%H:%M:%S)] recall training still running" | tee -a "$LOG"
    done
    echo "Recall training PID $TRAIN_PID exited at $(date -u)" | tee -a "$LOG"
fi

# Cooldown
sleep 30
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${GPU_FREE} MiB" | tee -a "$LOG"
if [ "$GPU_FREE" -lt 25000 ]; then
    echo "ERROR: GPU only ${GPU_FREE} MiB free, need 25GB+. Aborting." | tee -a "$LOG"
    exit 1
fi

# 2. Eval bfsi_recall on its 214 held-out (script will be written separately)
if [ -d "$PROJECT/adapters/nemotron_30b/bfsi_recall/best" ] && [ -f "$PROJECT/synapta_src/data_pipeline/13_eval_bfsi_recall.py" ]; then
    echo "" | tee -a "$LOG"
    echo "=== Phase: bfsi_recall eval ===" | tee -a "$LOG"
    timeout 5400 "$PYTHON" "$PROJECT/synapta_src/data_pipeline/13_eval_bfsi_recall.py" >> "$LOG" 2>&1 \
        || echo "Recall eval exited (timeout or error)" | tee -a "$LOG"
    sleep 30
else
    echo "Skipping recall eval - adapter or eval script missing" | tee -a "$LOG"
fi

# 3. Resume the bfsi_extract eval to finish FG mode (resume-aware, picks up missing FG rows)
echo "" | tee -a "$LOG"
echo "=== Phase: bfsi_extract FG mode finish (resume-aware) ===" | tee -a "$LOG"
timeout 14400 "$PYTHON" "$PROJECT/synapta_src/data_pipeline/08_eval_bfsi_extract.py" >> "$LOG" 2>&1 \
    || echo "FG eval exited (timeout or error)" | tee -a "$LOG"

echo "=== Recall chain complete at $(date -u) ===" | tee -a "$LOG"
