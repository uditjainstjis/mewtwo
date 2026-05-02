import torch
from typing import Dict, Optional

def additive_compose(x: torch.Tensor,
                     base_out: torch.Tensor,
                     adapters: Dict[str, Dict[str, torch.Tensor]],
                     routing_weights: Dict[str, float],
                     clamp_ratio: Optional[float] = None) -> torch.Tensor:
    """
    Standard additive composition with optional norm-ratio clamping (v0.0 baseline).
    
    h = z + Σ w_i * ΔW_i(x)
    
    Args:
        x: Input activations [batch, in]
        base_out: Base model layer output [batch, out]
        adapters: Dictionary of active adapters containing A and B matrices
        routing_weights: Dictionary of scalar routing weights
        clamp_ratio: If provided, dynamically scales the total update vector 
                     so its norm does not exceed clamp_ratio * norm(base_out).
                     
    Returns:
        Composed activation [batch, out]
    """
    total_update = torch.zeros_like(base_out)
    
    for name, matrices in adapters.items():
        weight = routing_weights.get(name, 0.0)
        if weight <= 0.0:
            continue
            
        A = matrices['A']
        B = matrices['B']
        
        # Linear layer update assumption: y = x @ W.T
        # Update = x @ A.T @ B.T
        update = (x @ A.T) @ B.T
        total_update += weight * update
        
    if clamp_ratio is not None and clamp_ratio > 0:
        base_norm = torch.norm(base_out, dim=-1, keepdim=True)
        update_norm = torch.norm(total_update, dim=-1, keepdim=True)
        
        max_allowed_norm = clamp_ratio * base_norm
        scale = torch.clamp(max_allowed_norm / (update_norm + 1e-8), max=1.0)
        total_update = total_update * scale
        
    return base_out + total_update
