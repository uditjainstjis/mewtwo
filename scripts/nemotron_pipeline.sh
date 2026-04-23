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

# ─── Step 7: Comprehensive Evaluation ───
echo ""
echo "=========================================="
echo " Step 7: Comprehensive Evaluation"
echo "=========================================="
$VENV -c "
import torch, json, sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.lori_moe.eval.run_benchmarks import evaluate_math, run_lm_eval_benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

def get_model(adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        '$ACTUAL_MODEL_PATH',
        quantization_config=bnb,
        device_map='auto',
        trust_remote_code=True
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()

def find_adapter(domain):
    for sub in ['dare_sparsified', 'best', 'final']:
        p = Path('$PROJECT_ROOT/checkpoints/nemotron_lori/adapters') / domain / sub
        if (p / 'adapter_config.json').exists():
            return str(p)
    return None

tok = AutoTokenizer.from_pretrained('$ACTUAL_MODEL_PATH', trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

results = {}

# 1. BASELINE
print('=== BASELINE (Zero-Shot) ===')
model = get_model()
results['baseline_gsm8k'] = evaluate_math(model, tok, 'gsm8k', 100).score
print(f'  Baseline GSM8K: {results[\"baseline_gsm8k\"]:.4f}')

lm_results = run_lm_eval_benchmark(model, tok, tasks=['humaneval', 'arc_challenge'], limit=50)
results['baseline_humaneval'] = lm_results.get('humaneval', type('obj', (object,), {'score': 0})).score
results['baseline_arc'] = lm_results.get('arc_challenge', type('obj', (object,), {'score': 0})).score
print(f'  Baseline HumanEval: {results[\"baseline_humaneval\"]:.4f}')
print(f'  Baseline ARC: {results[\"baseline_arc\"]:.4f}')

# 2. MATH ADAPTER
print('\\n=== MATH ADAPTER ===')
path = find_adapter('math')
if path:
    del model; torch.cuda.empty_cache()
    model = get_model(path)
    results['math_adapter_gsm8k'] = evaluate_math(model, tok, 'gsm8k', 100).score
    results['delta_math'] = results['math_adapter_gsm8k'] - results['baseline_gsm8k']
    print(f'  Math Adapter GSM8K: {results[\"math_adapter_gsm8k\"]:.4f} (Δ={results[\"delta_math\"]:.4f})')

# 3. CODE ADAPTER
print('\\n=== CODE ADAPTER ===')
path = find_adapter('code')
if path:
    del model; torch.cuda.empty_cache()
    model = get_model(path)
    lm_results = run_lm_eval_benchmark(model, tok, tasks=['humaneval'], limit=50)
    results['code_adapter_humaneval'] = lm_results.get('humaneval', type('obj', (object,), {'score': 0})).score
    results['delta_code'] = results['code_adapter_humaneval'] - results['baseline_humaneval']
    print(f'  Code Adapter HumanEval: {results[\"code_adapter_humaneval\"]:.4f} (Δ={results[\"delta_code\"]:.4f})')

# 4. SCIENCE ADAPTER
print('\\n=== SCIENCE ADAPTER ===')
path = find_adapter('science')
if path:
    del model; torch.cuda.empty_cache()
    model = get_model(path)
    lm_results = run_lm_eval_benchmark(model, tok, tasks=['arc_challenge'], limit=100)
    results['science_adapter_arc'] = lm_results.get('arc_challenge', type('obj', (object,), {'score': 0})).score
    results['delta_science'] = results['science_adapter_arc'] - results['baseline_arc']
    print(f'  Science Adapter ARC: {results[\"science_adapter_arc\"]:.4f} (Δ={results[\"delta_science\"]:.4f})')

with open('$PROJECT_ROOT/results/nemotron/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nResults saved to results/nemotron/evaluation_results.json')
" 2>&1 | tee "$LOG_DIR/evaluation.log"
echo "✅ Evaluation complete"

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE"
echo " Finished: $(date)"
echo " Results: $PROJECT_ROOT/results/nemotron/"
echo " Logs: $LOG_DIR/"
echo "=========================================="
