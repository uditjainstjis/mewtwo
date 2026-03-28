import os
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from safetensors.numpy import save_file
import numpy as np

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTERS_DIR = os.path.join(os.path.dirname(__file__), "adapters")
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "adapter_registry.json")

DOMAINS = {
    "Health": "Medical and health-related advice, diagnoses, and wellness tips.",
    "Law": "Legal advice, contract analysis, and legal reasoning.",
    "Psychology": "Mental health, therapy, cognitive analysis, and emotional support.",
    "Cooking": "Recipes, culinary techniques, and food science.",
    "Coding": "Programming, software engineering, debugging, and algorithms.",
    "Math": "Mathematical proofs, equations, calculus, and logic.",
    "Finance": "Economics, stock market analysis, accounting, and personal finance.",
    "History": "World history, historical events, analysis of past civilizations.",
    "Physics": "Quantum mechanics, classical mechanics, thermodynamics, and astrophysics.",
    "Biology": "Genetics, cellular biology, anatomy, and ecology.",
    "Chemistry": "Chemical reactions, molecular structures, and lab techniques.",
    "Literature": "Literary analysis, creative writing, poetry, and storytelling.",
    "Music": "Music theory, composition, history of music, and instrument playing.",
    "Art": "Art history, painting techniques, sculpture, and visual design.",
    "Sports": "Athletics, sports analytics, physical training, and game strategies.",
    "Gaming": "Video game design, esports, game mechanics, and walkthroughs.",
    "Travel": "Tourism, geography, cultural experiences, and travel planning.",
    "Mechanics": "Automotive repair, structural engineering, and mechanical systems.",
    "Politics": "Political science, government structures, policy analysis, and sociology.",
    "Philosophy": "Ethics, metaphysics, epistemology, and philosophical reasoning."
}

def get_linear_dims(layer):
    if hasattr(layer, "group_size"):
        out_features = layer.scales.shape[0]
        in_features = layer.scales.shape[1] * layer.group_size
        return in_features, out_features
    elif hasattr(layer, "weight"):
        out_features, in_features = layer.weight.shape
        return in_features, out_features
    else:
        raise ValueError("Unknown linear layer format")

def synthesize_adapters():
    os.makedirs(ADAPTERS_DIR, exist_ok=True)
    print(f"Loading Base Model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    
    target_keys = []
    for full_name, child in model.named_modules():
        if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
            if "q_proj" in full_name or "v_proj" in full_name:
                in_f, out_f = get_linear_dims(child)
                target_keys.append((full_name, in_f, out_f))
    print(f"Found {len(target_keys)} target matrices for LoRA.")
    
    registry = {}
    rank = 16
    alpha = 16.0
    
    for i, (domain, desc) in enumerate(DOMAINS.items()):
        print(f"Synthesizing adapter {i+1}/20: {domain}")
        domain_dir = os.path.join(ADAPTERS_DIR, domain.lower())
        os.makedirs(domain_dir, exist_ok=True)
        
        adapter_weights = {}
        for (key, in_f, out_f) in target_keys:
            # A: (rank, in_features), B: (out_features, rank)
            A = np.random.normal(0, 0.01, (rank, in_f)).astype(np.float32)
            B = np.zeros((out_f, rank), dtype=np.float32)
            adapter_weights[f"base_model.model.{key}.lora_A.weight"] = A
            adapter_weights[f"base_model.model.{key}.lora_B.weight"] = B
            
        sf_path = os.path.join(domain_dir, "adapter_model.safetensors")
        save_file(adapter_weights, sf_path)
        
        cfg = {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA"
        }
        with open(os.path.join(domain_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
            
        registry[domain] = {
            "path": sf_path,
            "description": desc,
            "vram_mb": round(os.path.getsize(sf_path) / (1024 * 1024), 2)
        }
        
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
        
    print("Done synthesizing adapters.")

if __name__ == "__main__":
    synthesize_adapters()
