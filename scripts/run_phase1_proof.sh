#!/bin/bash
export PYTHONPATH=$(pwd)
echo "Ensuring Data is built..."
mkdir -p data
python3 src/data/build_prompts.py

echo "Running Phase 1 Proof..."
python3 src/eval/run_eval.py --config_id exp_1_baseline
python3 src/eval/run_eval.py --config_id exp_5_adaptive_05

echo "Showing metrics run summary:"
python3 src/eval/metrics.py
