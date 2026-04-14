#!/bin/bash
set -e

echo "Setting up LoRI-MoE Foundation..."

# Install core dependencies for training
pip install -q -U transformers datasets accelerate peft triton pyarrow pydantic einops

# Verify environment
python -c "
import torch
from transformers import AutoTokenizer

model_id = 'Qwen/Qwen2.5-3B-Instruct'
print(f'Checking BF16 support: {torch.cuda.is_bf16_supported()}')

print(f'Downloading {model_id} tokenizer to cache...')
try:
    AutoTokenizer.from_pretrained(model_id)
    print('Tokenizer downloaded successfully.')
except Exception as e:
    print(f'Failed to download tokenizer: {e}')
"

# Create expected directories
mkdir -p src/lori_moe/data
mkdir -p src/lori_moe/model
mkdir -p src/lori_moe/training
mkdir -p src/lori_moe/eval
mkdir -p configs
mkdir -p data/datasets
mkdir -p checkpoints/adapters

echo "Foundation structure verified/created successfully."
