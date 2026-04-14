#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# LoRI-MoE: FIRE AND FORGET LAUNCHER
# 
# Run this once. Walk away. Come back to trained adapters.
#
# What it does:
#   1. Kills any existing training/dashboard processes
#   2. Starts the Flask dashboard on port 5000
#   3. Starts the PERPETUAL training pipeline
#   4. The pipeline trains adapters for ALL models in the queue
#   5. After finishing, it sleeps 5 min and checks for new work
#   6. To stop: touch checkpoints/lori_moe/STOP_PIPELINE
#
# Monitor:
#   - Dashboard: http://localhost:5000
#   - Pipeline:  tail -f logs/lori_moe/autonomous_pipeline.log
#   - Training:  tail -f logs/lori_moe/train_*.log
#   - GPU:       nvidia-smi -l 5
#
# ═══════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="/home/learner/Desktop/mewtwo"
cd "$PROJECT_ROOT"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         LoRI-MoE: FIRE AND FORGET LAUNCHER           ║"
echo "╚═══════════════════════════════════════════════════════╝"

# ─── Kill existing processes ────────────────────────────────────
echo "[1/4] Cleaning up existing processes..."
pkill -f "start_dashboard.py" 2>/dev/null || true
pkill -f "autonomous_pipeline.py" 2>/dev/null || true
# Give training processes a graceful 5s to save before hard kill
pkill -f "train_lori_adapter" 2>/dev/null || true
sleep 3
pkill -9 -f "train_lori_adapter" 2>/dev/null || true
sleep 2
echo "  ✓ Processes cleaned"

# ─── Remove stop signal if it exists ───────────────────────────
rm -f "$PROJECT_ROOT/checkpoints/lori_moe/STOP_PIPELINE"

# ─── Ensure directories ────────────────────────────────────────
echo "[2/4] Ensuring directory structure..."
mkdir -p logs/lori_moe checkpoints/lori_moe data/lori_moe
echo "  ✓ Directories ready"

# ─── Start Dashboard ───────────────────────────────────────────
echo "[3/4] Starting dashboard on port 5000..."
nohup python3 scripts/start_dashboard.py > logs/lori_moe/dashboard.log 2>&1 &
DASH_PID=$!
sleep 2

if kill -0 $DASH_PID 2>/dev/null; then
    echo "  ✓ Dashboard alive (PID: $DASH_PID)"
    echo "  ✓ Access: http://localhost:5000"
else
    echo "  ✗ Dashboard failed to start! Check logs/lori_moe/dashboard.log"
fi

# ─── Start Perpetual Pipeline ──────────────────────────────────
echo "[4/4] Starting perpetual training pipeline..."
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python3 scripts/autonomous_pipeline.py > logs/lori_moe/pipeline_stdout.log 2>&1 &
PIPE_PID=$!
sleep 1
echo "  ✓ Pipeline alive (PID: $PIPE_PID)"

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  🚀 ALL SYSTEMS GO — GPU WILL NEVER SIT IDLE         ║"
echo "║                                                       ║"
echo "║  Dashboard:  http://localhost:5000                    ║"
echo "║  Pipeline:   tail -f logs/lori_moe/autonomous_pipeline.log"
echo "║  GPU:        nvidia-smi -l 5                         ║"
echo "║                                                       ║"
echo "║  To stop:    touch checkpoints/lori_moe/STOP_PIPELINE ║"
echo "║  To restart: bash scripts/launch_forever.sh           ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "PIDs: Dashboard=$DASH_PID  Pipeline=$PIPE_PID"
echo "You can close this terminal. Training continues."
