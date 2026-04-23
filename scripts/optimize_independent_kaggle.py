import os
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
STAGING_DIR = PROJECT_ROOT / "hf_publish"

ADAPTERS = {
    "math": {
        "id": "uditjain13/nemotron-30b-math-reasoner-peft",
        "title": "Nemotron-30B Math Reasoner PEFT Adapter",
        "subtitle": "LoRA Math Expert Adapter for Nemotron-30B measuring structure mapping.",
        "dataset": "OpenMathInstruct-2",
        "human_eval": 0.60,
        "math_500": 0.505,
        "arc": 0.23,
        "mbpp": 0.02,
        "time": "~3.6 hours",
        "focus_text": "- **Math adapter \u2192 best at HumanEval (60%)**: Rigid proof-formatting generalizes perfectly to Python syntactical structure and bounds execution."
    },
    "code": {
        "id": "uditjain13/nemotron-30b-code-hyper-reasoner-peft",
        "title": "Nemotron-30B Code Hyper-Reasoner PEFT Adapter",
        "subtitle": "LoRA Code Expert Adapter for Nemotron-30B measuring deduction transfer.",
        "dataset": "sahil2801/CodeAlpaca-20k",
        "human_eval": 0.27,
        "math_500": 0.56,
        "arc": 0.31,
        "mbpp": 0.06,
        "time": "~8.1 hours",
        "focus_text": "- **Code adapter \u2192 best at MATH-500 (56%)**: Declarative Python constraints strips away format but massively boosts core mathematical logic generation."
    },
    "science": {
        "id": "uditjain13/nemotron-30b-science-expert-peft",
        "title": "Nemotron-30B Science Expert PEFT Adapter",
        "subtitle": "LoRA Science Expert Adapter for Nemotron-30B evaluating context bounds.",
        "dataset": "allenai/sciq",
        "human_eval": 0.01,
        "math_500": 0.55,
        "arc": 0.21,
        "mbpp": 0.00,
        "time": "~8.1 hours",
        "focus_text": "- **Science adapter**: Tests raw contextual extraction capabilities separated from rigid logic chains."
    }
}

def get_description(info):
    return f"""## {info['title']}
A domain-expert LoRA adapter (rank-64) fine-tuned specifically for the target domain on {info['dataset']}, built on top of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 — a hybrid Mamba-Attention mixture-of-experts architecture.
Trained on 1x NVIDIA RTX 5090 (32GB VRAM) for {info['time']}.
---
## The Cross-Domain Task-Inversion Phenomenon ("The Code Paradox")
During evaluation, we discovered a systematic cross-domain transfer inversion while researching dynamic token routing:
{info['focus_text']}

| Benchmark | Base Model | Expert Adapter | Delta |
|-----------|-----------|----------------|-------|
| ARC       | 20.0%     | {info['arc']*100}%          | {info['arc']*100 - 20}% |
| HumanEval | 50.0%     | {info['human_eval']*100}%          | {info['human_eval']*100 - 50}%|
| MATH-500  | 41.5%     | {info['math_500']*100}%          | {info['math_500']*100 - 41.5}%|
| MBPP      | 8.0%      | {info['mbpp']*100}%           | {info['mbpp']*100 - 8}% |

---
## Training Details
| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Learning Rate | 1e-4 |
| Training Steps | 1250 |
| Hardware | 1x RTX 5090 32GB |

Training dataset: {info['dataset']}
---
## How to Use
```python
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
adapter_id = "{info['id']}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

model = PeftModel.from_pretrained(base_model, adapter_id.replace("uditjain13", "uditjain"))
model.eval()

# Handle Hybrid Mamba Cache
try:
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, 'HybridMambaAttentionDynamicCache')
    past_key_values = HybridMambaAttentionDynamicCache(
        base_model.config, batch_size=1, dtype=torch.bfloat16, device=model.device
    )
except (AttributeError, KeyError):
    past_key_values = None  # Fallback for non-Mamba environments

messages = [{{"role": "user", "content": "Solve step by step: If x² + 5x + 6 = 0, find x."}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, past_key_values=past_key_values)
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```
---
## Related Resources
- 📧 Contact: hello@uditjain.in
"""

def update_dataset_metadata(key, info):
    print(f"📝 Updating Kaggle dataset metadata for {key}...")
    meta_path = STAGING_DIR / key / "dataset-metadata.json"
    
    metadata = {
        "id": info['id'],
        "title": info['title'],
        "subtitle": info['subtitle'],
        "description": get_description(info),
        "licenses": [{"name": "apache-2.0"}],
        "keywords": [
            "nlp", "deep learning", "transformers", "text generation",
            "fine-tuning", "peft", "lora", "math reasoning", "code generation",
            "mamba", "nemotron", "parameter efficient", "model weights", "reasoning"
        ],
        "collaborators": [],
        "data": [],
        "isPrivate": False
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    kaggle_cmd = [str(PROJECT_ROOT / ".venv/bin/kaggle"), "datasets", "version", "-p", str(STAGING_DIR / key), "-m", "Optimize Usability Score"]
    print(f"☁️ Pushing Kaggle dataset version for {key}...")
    res = subprocess.run(kaggle_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ {key} successfully updated on Kaggle!")
    else:
        print(f"❌ Error updating {key}: {res.stderr}\n{res.stdout}")

def main():
    for key, info in ADAPTERS.items():
        update_dataset_metadata(key, info)

if __name__ == "__main__":
    main()
