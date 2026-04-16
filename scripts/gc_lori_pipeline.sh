#!/usr/bin/env bash
# ============================================================================
# GC-LoRI Full Pipeline Orchestrator
# ============================================================================
#
# End-to-end execution of the Gate-Conditioned LoRI innovation.
# Designed for autonomous execution on RTX 5090.
#
# Usage:
#   chmod +x scripts/gc_lori_pipeline.sh
#   ./scripts/gc_lori_pipeline.sh 2>&1 | tee logs/nemotron/gc_lori_pipeline.log
#
# The script is idempotent — it checks for existing outputs before re-running.
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/home/learner/Desktop/mewtwo"
VENV="${PROJECT_ROOT}/.venv/bin/python"
MODEL_PATH="${PROJECT_ROOT}/models/nemotron"
DATA_DIR="${PROJECT_ROOT}/data/nemotron"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/nemotron_lori"
RESULTS_DIR="${PROJECT_ROOT}/results/nemotron"
LOG_DIR="${PROJECT_ROOT}/logs/nemotron"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}"; exit 1; }

# ============================================================================
# Phase 0: GPU Health Check
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 0: GPU Health Check"
log "═══════════════════════════════════════════════"

$VENV -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available! Fix driver first.'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
x = torch.randn(100, 100, device='cuda')
y = x @ x.T
print(f'Compute OK: {y.shape}')
" || fail "GPU health check failed. Fix driver before continuing."

log "✅ GPU is healthy"

# ============================================================================
# Phase 0.5: Architecture Probe
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 0.5: Architecture Probe"
log "═══════════════════════════════════════════════"

if [ -f "${MODEL_PATH}/module_map.json" ]; then
    log "✅ Module map already exists. Skipping probe."
else
    log "Running architecture probe..."
    $VENV scripts/nemotron_probe.py 2>&1 | tee "${LOG_DIR}/probe.log"
    [ -f "${MODEL_PATH}/module_map.json" ] || warn "module_map.json not created. Probe may have failed."
fi

# ============================================================================
# Phase 0.7: Internal Router Analysis (HIGHEST ROI EXPERIMENT)
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 0.7: Internal Router Analysis (NOVEL)"
log "═══════════════════════════════════════════════"

if [ -f "${RESULTS_DIR}/router_analysis/router_analysis.json" ]; then
    log "✅ Router analysis already exists. Skipping."
else
    log "Running internal router analysis — this determines if GC-LoRI is viable..."
    $VENV scripts/nemotron_router_analysis.py 2>&1 | tee "${LOG_DIR}/router_analysis.log"

    if [ -f "${RESULTS_DIR}/router_analysis/router_analysis.json" ]; then
        # Check viability
        VIABLE=$($VENV -c "
import json
with open('${RESULTS_DIR}/router_analysis/router_analysis.json') as f:
    data = json.load(f)
print(data.get('gc_lori_viable', 'unknown'))
")
        if [ "$VIABLE" = "True" ]; then
            log "✅ GC-LoRI is VIABLE! Internal routing correlates with domain."
        elif [ "$VIABLE" = "False" ]; then
            warn "GC-LoRI may NOT be viable. Will continue but results may be negative."
        else
            warn "Viability inconclusive. Continuing anyway."
        fi
    else
        warn "Router analysis output not found."
    fi
fi

# ============================================================================
# Phase 1: Data Preparation
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 1: Data Preparation"
log "═══════════════════════════════════════════════"

if [ -f "${DATA_DIR}/math_train.jsonl" ]; then
    log "✅ Nemotron training data already exists."
else
    log "Reformatting data for Nemotron template..."
    $VENV scripts/reformat_data_for_nemotron.py 2>&1 | tee "${LOG_DIR}/reformat.log"
fi

# ============================================================================
# Phase 1.5: Shared B Projection
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 1.5: Shared B Projection"
log "═══════════════════════════════════════════════"

if [ -f "${CHECKPOINT_DIR}/shared_projection_B.pt" ]; then
    log "✅ Shared projection already exists."
else
    log "Generating shared B projection (rank=64, hidden=2688)..."
    $VENV -c "
from src.lori_moe.shared_projection import get_shared_projection
from pathlib import Path
import torch
proj = get_shared_projection(
    hidden_size=2688, rank=64, seed=42,
    save_path=Path('${CHECKPOINT_DIR}/shared_projection_B.pt'),
    device='cuda', dtype=torch.bfloat16,
)
stats = proj.verify_orthogonality(num_samples=200, dim_out=2688)
print(f'Orthogonality: mean_cos_sim={stats[\"mean_cosine_similarity\"]:.6f}')
assert stats['mean_cosine_similarity'] < 0.01, 'Orthogonality check FAILED!'
print('✅ Shared projection generated and verified.')
"
fi

# ============================================================================
# Phase 2: Baseline Evaluation
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 2: Baseline Evaluation"
log "═══════════════════════════════════════════════"

if [ -f "${RESULTS_DIR}/evaluation_results.json" ]; then
    log "✅ Baseline evaluation results already exist."
else
    log "Running baseline evaluation..."
    $VENV -m src.lori_moe.eval.nemotron_eval \
        --model_path "$MODEL_PATH" \
        --output_dir "$RESULTS_DIR" \
        --max_samples 200 \
        --no_adapters \
        2>&1 | tee "${LOG_DIR}/baseline_eval.log"
fi

# ============================================================================
# Phase 3: Domain Adapter Training
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 3: Domain Adapter Training"
log "═══════════════════════════════════════════════"

ADAPTER_DIR="${CHECKPOINT_DIR}/adapters"

for DOMAIN in math code science; do
    if [ -d "${ADAPTER_DIR}/${DOMAIN}/final" ] || [ -d "${ADAPTER_DIR}/${DOMAIN}/dare_sparsified" ]; then
        log "✅ ${DOMAIN} adapter already trained."
    else
        log "Training ${DOMAIN} adapter..."
        $VENV -m src.lori_moe.training.train_lori_adapter \
            --domain "$DOMAIN" \
            --base_model "$MODEL_PATH" \
            --data_dir "$DATA_DIR" \
            --output_dir "$ADAPTER_DIR" \
            --rank 64 --alpha 128.0 --sparsity 0.8 \
            --epochs 2 --batch_size 2 --grad_accum 16 \
            --lr 1e-4 --max_seq_length 1024 \
            --max_train_samples 20000 \
            --gradient_checkpointing --use_4bit \
            --target_modules "q_proj,k_proj,v_proj,o_proj" \
            --save_every 200 --log_every 5 \
            2>&1 | tee "${LOG_DIR}/train_${DOMAIN}.log"
    fi
done

# ============================================================================
# Phase 3.5: Single Adapter Evaluation  
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 3.5: Single Adapter Evaluation"
log "═══════════════════════════════════════════════"

$VENV -m src.lori_moe.eval.nemotron_eval \
    --model_path "$MODEL_PATH" \
    --adapter_dir "$ADAPTER_DIR" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 200 \
    --no_baseline \
    2>&1 | tee "${LOG_DIR}/adapter_eval.log"

# ============================================================================
# Phase 4: GC-LoRI Router Training (THE INNOVATION)
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 4: GC-LoRI Router Training (NOVEL)"
log "═══════════════════════════════════════════════"

GC_ROUTER_DIR="${CHECKPOINT_DIR}/gc_router"

if [ -f "${GC_ROUTER_DIR}/best/gc_router.pt" ]; then
    log "✅ GC-LoRI router already trained."
else
    log "Training GC-LoRI + Blind routers (head-to-head)..."
    $VENV -m src.lori_moe.training.train_gc_router \
        --base_model "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$GC_ROUTER_DIR" \
        --epochs 5 --lr 5e-4 --batch_size 4 \
        --use_4bit \
        2>&1 | tee "${LOG_DIR}/train_gc_router.log"
fi

# ============================================================================
# Phase 5: Final Report
# ============================================================================
log "═══════════════════════════════════════════════"
log "PHASE 5: Summary"
log "═══════════════════════════════════════════════"

if [ -f "${GC_ROUTER_DIR}/training_log.json" ]; then
    $VENV -c "
import json
with open('${GC_ROUTER_DIR}/training_log.json') as f:
    data = json.load(f)
print()
print('═' * 50)
print('GC-LoRI TRAINING RESULTS')
print('═' * 50)
print(f'  Best GC-LoRI acc:  {data[\"best_gc_acc\"]:.1f}%')
print(f'  Best Blind acc:    {data[\"best_blind_acc\"]:.1f}%')
print(f'  Δ(GC - Blind):     {data[\"delta_best\"]:+.1f}%')
print(f'  Training time:     {data[\"total_time_min\"]:.1f} min')
print('═' * 50)
if data['delta_best'] > 2.0:
    print('✅ GC-LoRI is a meaningful improvement!')
elif data['delta_best'] > 0:
    print('⚠️  Small improvement. May not be significant.')
else:
    print('❌ GC-LoRI did NOT outperform blind routing.')
print()
"
fi

log "═══════════════════════════════════════════════"
log "GC-LoRI PIPELINE COMPLETE"
log "═══════════════════════════════════════════════"
log "Results: ${RESULTS_DIR}/"
log "Logs:    ${LOG_DIR}/"
log "Checkpoints: ${CHECKPOINT_DIR}/"
