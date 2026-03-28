#!/bin/bash
export PYTHONPATH=$(pwd)
echo "Starting Phase 2: Full Grid Execution"

# --- PART A: Mixed Compositional Datasets ---
# Expected Runtime: ~5 hours
echo "Running Mixed Compositional Evaluations..."
for dataset in mixed_fincode mixed_mathlegal mixed_codephilo; do
    python3 src/eval/run_eval.py --config_id exp_1_${dataset}
    python3 src/eval/run_eval.py --config_id exp_2_${dataset}
    python3 src/eval/run_eval.py --config_id exp_3_${dataset}
    python3 src/eval/run_eval.py --config_id exp_4_adaptive_03_${dataset}
    python3 src/eval/run_eval.py --config_id exp_5_adaptive_05_${dataset}
    python3 src/eval/run_eval.py --config_id exp_6_adaptive_05_k3_${dataset}
    python3 src/eval/run_eval.py --config_id exp_7_adaptive_07_${dataset}
done

# --- PART B: Pure Domain Preservation ---
# Expected Runtime: ~6 hours
echo "Running Pure Domain Evaluations..."
for dataset in pure_math pure_code; do
    python3 src/eval/run_eval.py --config_id exp_1_${dataset}
    python3 src/eval/run_eval.py --config_id exp_2_${dataset}
    python3 src/eval/run_eval.py --config_id exp_3_${dataset}
    python3 src/eval/run_eval.py --config_id exp_4_adaptive_03_${dataset}
    python3 src/eval/run_eval.py --config_id exp_5_adaptive_05_${dataset}
    python3 src/eval/run_eval.py --config_id exp_6_adaptive_05_k3_${dataset}
    python3 src/eval/run_eval.py --config_id exp_7_adaptive_07_${dataset}
done

# --- PART C: General Ability & Stability ---
# Expected Runtime: ~2 hours
echo "Running General NLP Evaluations (WikiText & MMLU)..."
for dataset in gen_wikitext gen_mmlu; do
    # Subset of grid run to prevent testing merging constraints excessively on general text
    python3 src/eval/run_eval.py --config_id exp_1_${dataset}
    python3 src/eval/run_eval.py --config_id exp_3_${dataset}
    python3 src/eval/run_eval.py --config_id exp_5_adaptive_05_${dataset}
    python3 src/eval/run_eval.py --config_id exp_6_adaptive_05_k3_${dataset}
done

echo "Phase 2 Pipeline Complete. Results appended to results_db.jsonl"
