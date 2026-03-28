import torch
import torch.nn as nn
from typing import List, Tuple
from src.adapters.registry import AdapterRegistry

class AdaptiveMultiLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, registry: AdapterRegistry, c: float = 0.5):
        super().__init__()
        self.base = base_layer
        self.registry = registry
        self.c = c
        self.epsilon = 1e-6

    def forward(self, x: torch.Tensor, active_experts: List[Tuple[str, float]]) -> torch.Tensor:
        # 1. Base network forward execution
        z = self.base(x)
        
        # 2. Compute geometrically weighted LoRA injections (m_l)
        m = torch.zeros_like(z)
        for expert_id, p_score in active_experts:
            A, B = self.registry.get_matrices(expert_id)
            # x @ A @ B conceptually maps the low-rank projection
            m += p_score * ((x @ A) @ B)

        # 3. Apply Norm-Proportional Adaptive Clamp
        norm_z = torch.linalg.norm(z, dim=-1, keepdim=True)
        norm_m = torch.linalg.norm(m, dim=-1, keepdim=True)
        
        # gamma = min(1.0, (c * ||z||) / (||m|| + eps))
        clamp_ratio = (self.c * norm_z) / (norm_m + self.epsilon)
        gamma = torch.clamp(clamp_ratio, max=1.0)
        
        # 4. Final state alignment
        return z + (gamma * m)
