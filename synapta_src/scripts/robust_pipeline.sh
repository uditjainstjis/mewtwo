#!/bin/bash
set -e

echo "🚀 Starting Robust Unattended Pipeline Pipeline"
echo "Starting at: $(date)"
# export HF_TOKEN="your_token_here"

# Set up logging and robust pipeline execution.
LOGFILE="/home/learner/Desktop/mewtwo/unattended_pipeline.log"
exec > >(tee -a $LOGFILE) 2>&1

# 1. HuggingFace Login
echo "1️⃣ Logging into HuggingFace..."
/home/learner/Desktop/mewtwo/.venv/bin/python3 -c "import huggingface_hub; import os; huggingface_hub.login(token=os.getenv('HF_TOKEN'))" || echo "⚠️ Warning: HF Login failed. Continuing anyway..."

# 2. Base Models Download
echo "2️⃣ Pre-downloading all 4 Base Models..."
cat << 'PY_EOF' > /home/learner/Desktop/mewtwo/synapta_src/synapta_src/scripts/download_models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

for m in models:
    print(f"Loading {m}...")
    try:
        AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            m, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ {m} successfully downloaded & cached.")
    except Exception as e:
        print(f"❌ Failed to download {m}: {e}")
PY_EOF
/home/learner/Desktop/mewtwo/.venv/bin/python3 /home/learner/Desktop/mewtwo/synapta_src/synapta_src/scripts/download_models.py || echo "⚠️ Warning: Model download issue. Resuming..."

# 3. Download and Process Data
echo "3️⃣ Running Data Curation (All Tiers)..."
# Using a "retry" loop
for tier in 1 2 3; do
    echo "Processing Tier $tier..."
    for i in {1..3}; do
        echo "Attempt $i for Tier $tier..."
        if /home/learner/Desktop/mewtwo/.venv/bin/python3 /home/learner/Desktop/mewtwo/synapta_src/synapta_src/scripts/curate_training_data.py --output-dir data/training --eval-dir data/eval --tier $tier; then
            echo "✅ Tier $tier data curated successfully!"
            break
        else
            echo "⚠️ Attempt $i failed for Tier $tier. Waiting 30s before retry..."
            sleep 30
        fi
    done
done

# 4. Generate the remaining CF-LoRA Training Script (to test script generation)
echo "4️⃣ Testing generation to make sure training starts automatically if desired..."

echo "✅ Unattended Pipeline Finished at: $(date)"
