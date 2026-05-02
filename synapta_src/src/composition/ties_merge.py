import torch
from typing import Dict, List, Optional

def ties_merge(adapters: Dict[str, Dict[str, torch.Tensor]], 
               density: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging (Trim, Elect Sign, & Merge).
    This is an OFFLINE merging baseline strategy. It collapses multiple LoRA
    updates into a single unified update vector/matrix before generation.
    
    1. Trim: Keep only the top-k % (density) of elements by magnitude per adapter.
    2. Elect Sign: Find the majority sign direction across all adapters per element.
    3. Merge: Average only the elements that agree with the elected sign.
    
    Args:
        adapters: Dict mapping adapter name -> {'A': tensor, 'B': tensor}
        density: Fraction of top magnitude weights to retain (0.0 to 1.0)
        
    Returns:
        A dictionary containing the merged ΔW matrix '{'delta_w': tensor}'.
        Note: The returned delta_w combines A and B, so it is a full [out, in] matrix.
    """
    if not adapters:
        return {}
        
    delta_ws = []
    for name, matrices in adapters.items():
        A = matrices['A']
        B = matrices['B']
        # ΔW = B @ A
        delta_w = B @ A
        delta_ws.append(delta_w)
        
    # Stack into [num_adapters, out_features, in_features]
    stacked_deltas = torch.stack(delta_ws, dim=0)
    
    # 1. Trim (Magnitude Pruning)
    # Calculate threshold for top-k magnitude
    k = max(1, int(density * stacked_deltas[0].numel()))
    
    trimmed_deltas = torch.zeros_like(stacked_deltas)
    for i in range(len(delta_ws)):
        flattened = stacked_deltas[i].abs().view(-1)
        # Find the k-th largest value
        kth_value = torch.kthvalue(flattened, flattened.numel() - k + 1).values
        mask = stacked_deltas[i].abs() >= kth_value
        trimmed_deltas[i] = stacked_deltas[i] * mask
        
    # 2. Elect Sign
    # Sum the values across adapters to find dominant direction
    sign_sum = trimmed_deltas.sum(dim=0)
    elected_sign = torch.sign(sign_sum)
    
    # 3. Disjoint Merge
    # Keep only values matching the elected sign
    sign_match = torch.sign(trimmed_deltas) == elected_sign.unsqueeze(0)
    
    # Filter the trimmed deltas
    filtered_deltas = trimmed_deltas * sign_match
    
    # Average the surviving non-zero elements
    # Count how many adapters contributed to each element
    counts = sign_match.sum(dim=0).float()
    counts = torch.clamp(counts, min=1.0) # Avoid division by zero
    
    merged_delta_w = filtered_deltas.sum(dim=0) / counts
    
    return {'delta_w': merged_delta_w}
