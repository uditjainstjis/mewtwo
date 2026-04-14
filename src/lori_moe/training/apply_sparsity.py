"""
Applies DARE-style sparsification to the trained A matrices of the LoRI adapters.
By dropping parameters with small magnitudes, we enforce additional orthogonality.
"""
import torch
import glob
from peft import PeftModel
from src.lori_moe.config import LoRIMoEConfig

def apply_sparsity_to_adapter(adapter_path, sparsity_level=0.8):
    print(f"Applying {sparsity_level*100}% sparsity to {adapter_path}")
    state_dict_path = f"{adapter_path}/adapter_model.bin"
    try:
        # Some peft versions use safetensors by default
        from safetensors.torch import load_file, save_file
        import os
        if os.path.exists(f"{adapter_path}/adapter_model.safetensors"):
            state_dict = load_file(f"{adapter_path}/adapter_model.safetensors")
            is_safetensors = True
        else:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            is_safetensors = False
    except Exception as e:
        print(f"Could not load state dict for {adapter_path}: {e}")
        return
    
    for key in list(state_dict.keys()):
        # In LoRI, the trainable parameters are lora_A.weight
        if "lora_A.weight" in key:
            tensor = state_dict[key]
            # Calculate threshold for magnitude pruning
            k = int(tensor.numel() * (1 - sparsity_level))
            if k == 0:
                continue
            
            # Find the kth largest value by magnitude
            magnitudes = torch.abs(tensor).flatten()
            threshold = torch.kthvalue(magnitudes, tensor.numel() - k + 1).values.item()
            
            # Create mask
            mask = torch.abs(tensor) >= threshold
            
            # Apply mask and rescale
            sparse_tensor = tensor * mask
            rescaled_tensor = sparse_tensor / (1 - sparsity_level)
            
            state_dict[key] = rescaled_tensor
            
    if is_safetensors:
        save_file(state_dict, f"{adapter_path}/adapter_model_sparse.safetensors")
        print(f"Saved sparsified weights to {adapter_path}/adapter_model_sparse.safetensors")
    else:
        torch.save(state_dict, f"{adapter_path}/adapter_model_sparse.bin")
        print(f"Saved sparsified weights to {adapter_path}/adapter_model_sparse.bin")

if __name__ == "__main__":
    config = LoRIMoEConfig()
    adapter_dirs = glob.glob(str(config.paths.adapters_dir / "*"))
    
    for adapter_dir in adapter_dirs:
        apply_sparsity_to_adapter(adapter_dir, config.adapter.sparsity_level)
