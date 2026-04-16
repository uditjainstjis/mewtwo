#!/bin/bash
# =================================================================
# Nemotron LoRI-MoE Full Autonomous Pipeline
# =================================================================
# This script runs the ENTIRE pipeline from probe to evaluation.
# It is designed to be run unattended. All output is logged.
#
# Usage:
#   cd /home/learner/Desktop/mewtwo
#   bash scripts/nemotron_pipeline.sh 2>&1 | tee logs/nemotron/pipeline.log
#
# Prerequisites:
#   - Nemotron model extracted to models/nemotron/
#   - Python venv at .venv/
#   - NVIDIA GPU with 32GB VRAM
# =================================================================

set -euo pipefail

PROJECT_ROOT="/home/learner/Desktop/mewtwo"
VENV="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/nemotron"
MODEL_DIR="$PROJECT_ROOT/models/nemotron"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " Nemotron LoRI-MoE Pipeline"
echo " Started: $(date)"
echo "=========================================="

# ─── Find the actual model path (handles nested directories) ───
find_model_path() {
    if [ -f "$MODEL_DIR/config.json" ]; then
        echo "$MODEL_DIR"
        return
    fi
    for subdir in "$MODEL_DIR"/*/; do
        if [ -f "${subdir}config.json" ]; then
            echo "${subdir%/}"
            return
        fi
    done
    echo ""
}

ACTUAL_MODEL_PATH=$(find_model_path)
if [ -z "$ACTUAL_MODEL_PATH" ]; then
    echo "ERROR: Cannot find config.json in $MODEL_DIR or subdirectories."
    echo "Contents of $MODEL_DIR:"
    ls -la "$MODEL_DIR" 2>/dev/null || echo "(empty)"
    exit 1
fi
echo "Model found at: $ACTUAL_MODEL_PATH"

# ─── Step 1: Architecture Probe ───
echo ""
echo "=========================================="
echo " Step 1: Architecture Probe"
echo "=========================================="
$VENV scripts/nemotron_probe.py 2>&1 | tee "$LOG_DIR/probe.log"
echo "✅ Probe complete"

# ─── Step 2: Reformat Training Data ───
echo ""
echo "=========================================="
echo " Step 2: Reformat Training Data"
echo "=========================================="
$VENV scripts/reformat_data_for_nemotron.py 2>&1 | tee "$LOG_DIR/reformat.log"
echo "✅ Data reformatted"

# ─── Step 3: Generate Shared B Projection ───
echo ""
echo "=========================================="
echo " Step 3: Generate Shared B Projection"
echo "=========================================="
$VENV -c "
from src.lori_moe.shared_projection import get_shared_projection
from pathlib import Path
import torch

# Read hidden_size from probe results
import json
probe_file = Path('$ACTUAL_MODEL_PATH') / 'probe_results.json'
if probe_file.exists():
    with open(probe_file) as f:
        probe = json.load(f)
    hidden_size = probe['hidden_size']
else:
    hidden_size = 2688  # Default Nemotron hidden size

print(f'Generating shared B: rank=64, hidden={hidden_size}')
proj = get_shared_projection(
    hidden_size=hidden_size,
    rank=64,
    seed=42,
    save_path=Path('$PROJECT_ROOT/checkpoints/nemotron_lori/shared_projection_B.pt'),
    device='cuda',
    dtype=torch.bfloat16,
)
stats = proj.verify_orthogonality(num_samples=200, dim_out=hidden_size)
print(f'Orthogonality: mean_cos_sim={stats[\"mean_cosine_similarity\"]:.6f}')
print(f'Quality: {stats[\"orthogonality_quality\"]}')
" 2>&1 | tee "$LOG_DIR/shared_b.log"
echo "✅ Shared B generated"

# ─── Step 4: Train Math Adapter (Primary) ───
echo ""
echo "=========================================="
echo " Step 4: Train Math Adapter"
echo "=========================================="

# Read target modules from probe to determine what PEFT can target
# For safety, start with attention modules only
$VENV -m src.lori_moe.training.train_lori_adapter \
    --domain math \
    --base_model "$ACTUAL_MODEL_PATH" \
    --data_dir "$PROJECT_ROOT/data/nemotron" \
    --output_dir "$PROJECT_ROOT/checkpoints/nemotron_lori/adapters" \
    --rank 64 --alpha 128.0 --sparsity 0.8 \
    --epochs 2 --batch_size 2 --grad_accum 16 \
    --lr 1e-4 --max_seq_length 1024 \
    --max_train_samples 10000 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --gradient_checkpointing \
    --use_4bit \
    --save_every 100 --log_every 5 \
    2>&1 | tee "$LOG_DIR/train_math.log"
echo "✅ Math adapter trained"

# ─── Step 5: Train Code Adapter ───
echo ""
echo "=========================================="
echo " Step 5: Train Code Adapter"
echo "=========================================="
$VENV -m src.lori_moe.training.train_lori_adapter \
    --domain code \
    --base_model "$ACTUAL_MODEL_PATH" \
    --data_dir "$PROJECT_ROOT/data/nemotron" \
    --output_dir "$PROJECT_ROOT/checkpoints/nemotron_lori/adapters" \
    --rank 64 --alpha 128.0 --sparsity 0.8 \
    --epochs 2 --batch_size 2 --grad_accum 16 \
    --lr 1e-4 --max_seq_length 1024 \
    --max_train_samples 10000 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --gradient_checkpointing \
    --use_4bit \
    --save_every 100 --log_every 5 \
    2>&1 | tee "$LOG_DIR/train_code.log"
echo "✅ Code adapter trained"

# ─── Step 6: Train Science Adapter ───
echo ""
echo "=========================================="
echo " Step 6: Train Science Adapter"
echo "=========================================="
$VENV -m src.lori_moe.training.train_lori_adapter \
    --domain science \
    --base_model "$ACTUAL_MODEL_PATH" \
    --data_dir "$PROJECT_ROOT/data/nemotron" \
    --output_dir "$PROJECT_ROOT/checkpoints/nemotron_lori/adapters" \
    --rank 64 --alpha 128.0 --sparsity 0.8 \
    --epochs 2 --batch_size 2 --grad_accum 16 \
    --lr 1e-4 --max_seq_length 1024 \
    --max_train_samples 10000 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --gradient_checkpointing \
    --use_4bit \
    --save_every 100 --log_every 5 \
    2>&1 | tee "$LOG_DIR/train_science.log"
echo "✅ Science adapter trained"

# ─── Step 7: Baseline Evaluation ───
echo ""
echo "=========================================="
echo " Step 7: Baseline Evaluation"
echo "=========================================="
$VENV -c "
import torch, json, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.lori_moe.eval.run_benchmarks import evaluate_math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
tok = AutoTokenizer.from_pretrained('$ACTUAL_MODEL_PATH', trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print('=== BASELINE (no adapter) ===')
model = AutoModelForCausalLM.from_pretrained('$ACTUAL_MODEL_PATH', quantization_config=bnb, device_map='auto', trust_remote_code=True)
model.eval()
base_result = evaluate_math(model, tok, 'gsm8k', 200)
print(f'Baseline GSM8K: {base_result.score:.4f} ({base_result.num_examples} examples)')

results = {'baseline_gsm8k': base_result.score}

# Math adapter
del model; torch.cuda.empty_cache()
print('\\n=== MATH ADAPTER ===')
model = AutoModelForCausalLM.from_pretrained('$ACTUAL_MODEL_PATH', quantization_config=bnb, device_map='auto', trust_remote_code=True)

from pathlib import Path
adapter_path = None
for sub in ['dare_sparsified', 'best', 'final']:
    p = Path('$PROJECT_ROOT/checkpoints/nemotron_lori/adapters/math') / sub
    if (p / 'adapter_config.json').exists():
        adapter_path = str(p)
        break

if adapter_path:
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    math_result = evaluate_math(model, tok, 'gsm8k', 200)
    print(f'Math Adapter GSM8K: {math_result.score:.4f}')
    results['math_adapter_gsm8k'] = math_result.score
    results['delta_math'] = math_result.score - base_result.score
else:
    print('No math adapter found!')

with open('$PROJECT_ROOT/results/nemotron/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nResults saved to results/nemotron/evaluation_results.json')
print(f'Results: {json.dumps(results, indent=2)}')
" 2>&1 | tee "$LOG_DIR/evaluation.log"
echo "✅ Evaluation complete"

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE"
echo " Finished: $(date)"
echo " Results: $PROJECT_ROOT/results/nemotron/"
echo " Logs: $LOG_DIR/"
echo "=========================================="
