#!/bin/bash
# LoRI-MoE FULL AUTONOMOUS PIPELINE
# Trains all adapters → Router → Done
# Safe to walk away — everything runs unattended via nohup.
# Monitor: tail -f /home/learner/Desktop/mewtwo/logs/lori_moe/pipeline.log

set -e

DOMAINS=("math" "code" "science" "legal" "medical")
BASE_MODEL="Qwen/Qwen3.5-0.8B"
BASE_MODEL_NAME="qwen3.5_0.8b"
CHECKPOINT_DIR="/home/learner/Desktop/mewtwo/checkpoints/lori_moe/$BASE_MODEL_NAME"
LOG_DIR="/home/learner/Desktop/mewtwo/logs/lori_moe"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "==============================================="
echo "LoRI-MoE: FULL AUTONOMOUS PIPELINE"
echo "Base Model: $BASE_MODEL ($BASE_MODEL_NAME)"
echo "Started at: $(date)"
echo "==============================================="

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ═══════════════════════════════════════════════════
# PHASE 1: Train all 5 domain adapters
# batch_size=8 + gradient_checkpointing = ~8GB VRAM
# ~1.2 it/s at bs=4, should be ~0.6 it/s at bs=8
# ~2500 steps/domain × 3 epochs ÷ 2 (bs doubled) = ~1250 steps
# ~1250 steps / 0.6 it/s = ~35 min per domain
# Total Phase 1: ~3 hours
# ═══════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  PHASE 1: Training Domain Adapters (5/5)     ║"
echo "║  ETA: ~3 hours                               ║"
echo "╚═══════════════════════════════════════════════╝"

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "━━━ [$domain] START $(date) ━━━"
    python -m src.lori_moe.training.train_lori_adapter \
        --domain "$domain" \
        --base_model "$BASE_MODEL" \
        --output_dir "$CHECKPOINT_DIR" \
        --rank 32 \
        --sparsity 0.8 \
        --epochs 3 \
        --max_train_samples 10000 \
        --batch_size 4 \
        --grad_accum 4 \
        --lr 5e-5 \
        --gradient_checkpointing
    echo "━━━ [$domain] DONE $(date) ━━━"
done

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  PHASE 1 COMPLETE — All 5 adapters trained!  ║"
echo "║  $(date)                                     ║"
echo "╚═══════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════
# PHASE 3: Train token-level router
# Only ~200K trainable params, very fast (~15-30 min)
# ═══════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  PHASE 3: Training Token-Level Router        ║"
echo "║  ETA: ~30 minutes                            ║"
echo "╚═══════════════════════════════════════════════╝"

python -m src.lori_moe.training.train_router \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$CHECKPOINT_DIR" \
    --output_dir "$CHECKPOINT_DIR/router" \
    --lr 1e-3 \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum 4

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  ALL PHASES COMPLETE!                        ║"
echo "║  $(date)                                     ║"
echo "║                                              ║"
echo "║  Adapters: $CHECKPOINT_DIR/*/                ║"
echo "║  Router:   $CHECKPOINT_DIR/router/           ║"
echo "║                                              ║"
echo "║  Next: python -m src.lori_moe.eval.run_benchmarks ║"
echo "╚═══════════════════════════════════════════════╝"
