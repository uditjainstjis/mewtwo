#!/usr/bin/env python3
"""
Neural Router Data Collection (Nemotron-30B)
Extracts hidden states from layer groups to train a domain classifier.
"""
import os, sys, gc, json, logging, torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_collect")

# ─── Config ───
SAMPLES_PER_DOMAIN = 100
TARGET_LAYERS = [8, 16, 24, 32, 40, 48]
HIDDEN_DIM = 2688

# ─── Model Loading ───
log.info("Loading Nemotron-30B for feature extraction...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True)
model.eval()

# ─── Data Map ───
# Domain labels: 0=Math, 1=Code, 2=Science
DOMAINS = [
    {"name": "math", "dataset": "HuggingFaceH4/MATH-500", "split": "test", "col": "problem", "label": 0},
    {"name": "code", "dataset": "openai/openai_humaneval", "split": "test", "col": "prompt", "label": 1},
    {"name": "science", "dataset": "ai2_arc", "name_cfg": "ARC-Challenge", "split": "test", "col": "question", "label": 2},
]

# ─── Hook Setup ───
STORAGE = []

def get_hook(label):
    def hook(module, input, output):
        # output is a tuple (hidden_states, ...)
        # hidden_states is (B, S, D). We take the last generated token's state.
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        
        # We only want the last token for the router's current decision context
        # (B, S, D) -> (B, D)
        last_h = h[:, -1, :].detach().cpu().float()
        STORAGE.append({"state": last_h, "label": label})
    return hook

# ─── Collection Loop ───

def collect():
    final_data = {"states": [], "labels": []}
    
    for domain in DOMAINS:
        log.info(f"Processing domain: {domain['name']}...")
        if "name_cfg" in domain:
            ds = load_dataset(domain["dataset"], domain["name_cfg"], split=domain["split"])
        else:
            ds = load_dataset(domain["dataset"], split=domain["split"])
        
        samples = ds.select(range(min(len(ds), SAMPLES_PER_DOMAIN)))
        
        # We hook into a mid-depth layer (e.g. layer 32) for "semantic logic" features
        target_layer_idx = 32
        handle = model.backbone.layers[target_layer_idx].register_forward_hook(get_hook(domain["label"]))
        
        for ex in tqdm(samples, desc=domain["name"]):
            text = ex[domain["col"]]
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                model(**inputs)
            
            # Flush STORAGE to final_data
            for item in STORAGE:
                final_data["states"].append(item["state"].squeeze().tolist())
                final_data["labels"].append(item["label"])
            STORAGE.clear()
            
        handle.remove()
        gc.collect()
        torch.cuda.empty_cache()

    # Save to disk
    save_path = DATA_DIR / "neural_router_train_data.pt"
    torch.save(final_data, save_path)
    log.info(f"✅ Data collection complete. Saved {len(final_data['labels'])} tokens to {save_path}")

if __name__ == "__main__":
    collect()
