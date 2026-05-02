#!/bin/bash
# gpu_watchdog.sh

echo "🐕 Starting Mewtwo GPU Watchdog Process..."
cd /home/learner/Desktop/mewtwo

while true; do
    if ! pgrep -f "mewtwo_auto_researcher.py" > /dev/null; then
        echo "[$(date)] 🚨 Researcher script not found in processes! Launching orchestrator..."
        nohup env PYTHONPATH=/home/learner/Desktop/mewtwo /home/learner/Desktop/mewtwo/.venv/bin/python3 mewtwo_auto_researcher.py > /home/learner/Desktop/mewtwo/researcher_ui.log 2>&1 &
    fi
    sleep 15
done
