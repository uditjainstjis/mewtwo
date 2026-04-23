#!/bin/bash
# MEWTWO Auto-Chain: Waits for Phase 1 to finish, kills master pipeline,
# then launches the research sprint (Phases 2-5).
# Usage: nohup bash scripts/auto_chain.sh &

cd /home/learner/Desktop/mewtwo
VENV=".venv/bin/python"
LOG="logs/nemotron/auto_chain.log"

echo "[$(date)] Auto-chain started. Watching master_pipeline..." | tee -a "$LOG"

# Wait for Phase 1 to complete (master_pipeline.py finishes Phase 1 and starts Phase 2/LayerBlend)
# We detect this by watching for "Phase 1 complete" OR "Phase 2" in the master log
while true; do
    if grep -q "Phase 1 complete\|PHASE 2\|LAYERBLEND\|Phase 2:" logs/nemotron/master_pipeline.log 2>/dev/null; then
        echo "[$(date)] Phase 1 detected as complete!" | tee -a "$LOG"
        break
    fi
    # Also check if master pipeline crashed/finished
    if ! pgrep -f master_pipeline > /dev/null 2>&1; then
        echo "[$(date)] Master pipeline not running. Proceeding." | tee -a "$LOG"
        break
    fi
    sleep 60
done

# Kill the master pipeline (to stop LayerBlend from wasting GPU)
MASTER_PID=$(pgrep -f master_pipeline)
if [ -n "$MASTER_PID" ]; then
    echo "[$(date)] Killing master_pipeline (PID: $MASTER_PID) to skip LayerBlend..." | tee -a "$LOG"
    kill $MASTER_PID 2>/dev/null
    sleep 5
    kill -9 $MASTER_PID 2>/dev/null
fi

# Wait for GPU to free up
sleep 10
echo "[$(date)] GPU clear. Launching research sprint..." | tee -a "$LOG"

# Launch the research sprint
$VENV scripts/research_sprint.py >> logs/nemotron/research_sprint.log 2>&1
echo "[$(date)] Research sprint finished." | tee -a "$LOG"
