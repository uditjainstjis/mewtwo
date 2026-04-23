#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# GRAND ROUTING COMPARISON: Full Autonomous Pipeline
# Phase 1: 9 strategies (no-adapter, single, regex, guard, neural, PPL, entropy, oracle)
# Phase 2: 3 strategies (SFT on oracle, REINFORCE, UCB bandit)
# ═══════════════════════════════════════════════════════════════════════

set -e
cd /home/learner/Desktop/mewtwo
VENV=".venv/bin/python"

echo "═══════════════════════════════════════════════════════════"
echo "  SYNAPTA ROUTING GRAND COMPARISON — AUTONOMOUS PIPELINE"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

# Phase 1: 9 base strategies
echo "[Phase 1] Running 9 routing strategies..."
$VENV scripts/routing_grand_comparison.py 2>&1 | tee logs/grand_comparison.log

echo ""
echo "[Phase 1] ✅ Complete. Results in results/nemotron/grand_comparison_results.json"
echo ""

# Phase 2: SFT + RL + Bandit (needs Phase 1 oracle traces)
echo "[Phase 2] Collecting oracle traces → Training SFT + REINFORCE → Evaluating..."
$VENV scripts/routing_phase2_sft_rl.py 2>&1 | tee logs/phase2_sft_rl.log

echo ""
echo "[Phase 2] ✅ Complete."
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  ALL 12 STRATEGIES EVALUATED"
echo "  Results: results/nemotron/grand_comparison_results.json"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════"
