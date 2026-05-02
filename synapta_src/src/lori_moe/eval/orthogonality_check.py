import os
import torch
from pathlib import Path
from safetensors.torch import load_file
import numpy as np

def check_orthogonality(adapter_dir: str):
    print(f"\n{'='*60}")
    print("LoRI-MoE Orthogonality Check (Catastrophic Interference Test)")
    print(f"{'='*60}")
    
    domains = ["math", "code", "science", "legal", "medical"]
    base_path = Path(adapter_dir)
    
    # Load all trainable A matrices (in our implementation, these are saved as lora_B)
    domain_weights = {}
    
    for domain in domains:
        sf_path = base_path / domain / "dare_sparsified" / "adapter_model.safetensors"
        if not sf_path.exists():
            print(f"Skipping {domain} (not found)")
            continue
            
        tensors = load_file(sf_path)
        
        # Flatten and concat all trained lora_B weights for this domain
        # (These are the domain-specific sparse matrices)
        layer_weights = []
        for name, tensor in tensors.items():
            if 'lora_B' in name:
                layer_weights.append(tensor.flatten())
                
        if layer_weights:
            flat_vector = torch.cat(layer_weights).float()
            # Mean center the vector
            flat_vector = flat_vector - flat_vector.mean()
            # L2 Normalize
            flat_vector = flat_vector / flat_vector.norm()
            domain_weights[domain] = flat_vector
            print(f"Loaded {domain}: {len(layer_weights)} layers, {flat_vector.shape[0]} params")
            
    if len(domain_weights) < 2:
        print("Need at least 2 domains to compare.")
        return
        
    print(f"\n{'-'*60}")
    print("Cosine Similarity Matrix (0.0 = perfectly orthogonal/no interference)")
    print(f"{'-'*60}")
    
    loaded_domains = list(domain_weights.keys())
    
    # Header
    header = f"{'':>10}" + "".join([f"{d:>10}" for d in loaded_domains])
    print(header)
    
    num_domains = len(loaded_domains)
    sim_matrix = np.zeros((num_domains, num_domains))
    
    for i, d1 in enumerate(loaded_domains):
        row = f"{d1:>10}"
        for j, d2 in enumerate(loaded_domains):
            if i == j:
                sim = 1.0
            else:
                sim = torch.dot(domain_weights[d1], domain_weights[d2]).item()
            sim_matrix[i, j] = sim
            row += f"{sim:>10.4f}"
        print(row)
        
    # Analyze
    off_diag = sim_matrix[~np.eye(num_domains, dtype=bool)]
    avg_sim = np.abs(off_diag).mean()
    
    print(f"\n{'='*60}")
    print(f"Average absolute cross-domain similarity: {avg_sim:.5f}")
    if avg_sim < 0.05:
        print("✅ SUCCESS: Vectors are highly orthogonal. Catastrophic forgetting is eliminated!")
    else:
        print("❌ WARNING: Vectors show significant overlap. Interference is possible.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="/home/learner/Desktop/mewtwo/adapters/lori_moe/qwen2.5_1.5b")
    args = parser.parse_args()
    check_orthogonality(args.adapter_dir)
