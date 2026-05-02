import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# Ensure imports work for local modules
import sys
from pathlib import Path
PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))
from src.lori_moe.eval.nemotron_eval import load_model_4bit, generate_response
from models.nemotron.modeling_nemotron_h import HybridMambaAttentionDynamicCache

def main():
    # Enable full telemetry
    os.environ["DEBUG_MOE"] = "1"
    os.environ["DEBUG_GEN"] = "1"
    
    model_path = "/home/learner/Desktop/mewtwo/models/nemotron"
    adapter_path = "/home/learner/Desktop/mewtwo/adapters/nemotron_30b/math/final"
    
    print(f"Loading model and adapter from {adapter_path}...")
    model, tokenizer = load_model_4bit(model_path)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load sample 71
    print("Loading GSM8K sample 71...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ex = ds[71]
    question = ex["question"]
    print(f"Question: {question}")
    
    # Prompt formatting (same as eval)
    system = "You are a helpful assistant. Provide step-by-step reasoning for math problems."
    prompt = f"<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n"
    
    print("\n--- STARTING GENERATION (Sample 71) ---")
    response = generate_response(model, tokenizer, prompt)
    print("\n--- FINAL RESPONSE ---")
    print(response)
    print("\n--- END ---")

if __name__ == "__main__":
    main()
