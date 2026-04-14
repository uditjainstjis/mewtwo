import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.lori_moe.config import LoRIMoEConfig
from src.lori_moe.model.lori_moe_model import LoRIMoEModel
from transformers import AutoTokenizer

def analyze_routing(prompt: str, save_path: str = "router_heatmap.png"):
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = "/home/learner/Desktop/mewtwo/checkpoints/lori_moe/qwen2.5_1.5b"
    router_path = "/home/learner/Desktop/mewtwo/checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt"

    config = LoRIMoEConfig()
    config.model.base_model = base_model_name

    moe_model = LoRIMoEModel(config)
    moe_model.build(load_experts=True, adapters_root=Path(adapter_dir))
    
    router_state = torch.load("checkpoints/lori_moe/qwen2.5_1.5b/router/best/router.pt", map_location="cpu")
    moe_model.routers.load_state_dict(router_state["router_state_dict"])
    moe_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        moe_model(**inputs)
        
    # extract routing weights per layer. shape: dict[layer_idx] -> (batch, seq_len, num_experts)
    routing_data = moe_model._last_routing_by_layer
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    # Decode special tokens for readability
    tokens = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]
    
    # We will average the routing weights across all layers to get a global expert probability per token
    num_layers = len(routing_data)
    seq_len = inputs.input_ids.shape[1]
    num_experts = len(moe_model.expert_names)
    
    avg_routing = torch.zeros(seq_len, num_experts, device="cuda")
    for layer_idx, weights in routing_data.items():
        avg_routing += weights[0] # (seq_len, num_experts)
    avg_routing /= num_layers
    
    avg_routing_np = avg_routing.cpu().numpy()
    
    plt.figure(figsize=(12, min(0.4 * seq_len, 20)))
    sns.heatmap(avg_routing_np, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=moe_model.expert_names, yticklabels=tokens)
    plt.title("Average Layer Routing Probabilities per Token")
    plt.ylabel("Tokens")
    plt.xlabel("Domain Experts")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    prompt = "Q: Analyze this medical report for a patient with fever. Define a function in python to calculate IBM. Then solve 2 + 2 * X = 10 for X."
    analyze_routing(prompt)
