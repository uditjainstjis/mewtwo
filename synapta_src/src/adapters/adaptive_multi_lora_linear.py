import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from src.adapters.registry import AdapterRegistry

class AdaptiveMultiLoRALinear(nn.Module):
    """
    Multi-adapter composition layer with per-layer norm-ratio clamp
    and optional layer-sparse injection.

    v2 changes:
      - Added `layer_idx` to forward(): identifies this layer's position.
      - Added `L_start` to __init__(): layers below this index skip
        adapter injection entirely, preserving early-layer representations.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        registry: AdapterRegistry,
        c: float = 0.5,
        L_start: int = 0,
    ):
        super().__init__()
        self.base = base_layer
        self.registry = registry
        self.c = c
        self.L_start = L_start
        self.epsilon = 1e-6

    def forward(
        self,
        x: torch.Tensor,
        active_experts: List[Tuple[str, float]],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # 1. Base network forward execution
        z = self.base(x)

        # Layer-sparse injection: skip adapter ΔW for early layers
        if layer_idx is not None and layer_idx < self.L_start:
            return z

        if not active_experts:
            return z

        # 2. Compute geometrically weighted LoRA injections (m_l)
        m = torch.zeros_like(z)
        for expert_id, p_score in active_experts:
            A, B = self.registry.get_matrices(expert_id)
            # x @ A @ B conceptually maps the low-rank projection
            m += p_score * ((x @ A) @ B)

        # 3. Apply Per-Layer Norm-Proportional Adaptive Clamp
        norm_z = torch.linalg.norm(z, dim=-1, keepdim=True)
        norm_m = torch.linalg.norm(m, dim=-1, keepdim=True)

        # gamma_l = min(1.0, (c * ||z_l||) / (||m_l|| + eps))
        clamp_ratio = (self.c * norm_z) / (norm_m + self.epsilon)
        gamma = torch.clamp(clamp_ratio, max=1.0)

        # 4. Final state alignment
        return z + (gamma * m)
