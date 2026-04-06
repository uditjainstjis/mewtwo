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
