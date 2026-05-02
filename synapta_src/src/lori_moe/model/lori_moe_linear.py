"""
LoRI-MoE Linear Layer

The core module that replaces standard Linear layers in the transformer.
This is conceptually equivalent to Synapta's `AdaptiveMultiLoRALinear` but
with three fundamental improvements:

1. NO CLAMPING — orthogonality prevents interference structurally (not by magnitude bounding)
2. TOKEN-LEVEL ROUTING — each token gets its own expert mixture (not prompt-level)
3. SHARED B PROJECTION — single matrix multiply, O(1) vs O(K) cost

Forward pass:
  1. Base output:    z = W_base @ x
  2. Project:        x_proj = B @ x           (shared, frozen)
  3. Route:          p = Router(h)             (token-level probabilities)
  4. Expert compose: Δ = Σ p_k * A_k @ x_proj (weighted sparse expert outputs)
  5. Output:         y = z + Δ
"""
import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LoRIMoELinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds LoRI-MoE expert composition.

    At inference, computes:
      y = W_base(x) + Σ_k routing_weight_k * (scaling * A_k @ B @ x)

    The critical insight: B @ x is computed ONCE and shared across all experts.
    Only the sparse A_k composition changes per-token based on routing weights.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        shared_B: torch.Tensor,           # Frozen shared projection (rank × in_features)
        expert_A_matrices: nn.ModuleDict,  # Domain-specific sparse A matrices
        expert_names: list,
        scaling: float = 2.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = scaling
        self.expert_names = expert_names

        # Frozen shared B — register as buffer so it moves with .to() but isn't trained
        self.register_buffer("shared_B", shared_B.detach().clone())

        # Expert A matrices (sparse, domain-specific)
        self.expert_A = expert_A_matrices

        # Precompute shapes
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = shared_B.shape[0]
        self._current_routing_weights: Optional[torch.Tensor] = None

    def set_current_routing(self, routing_weights: Optional[torch.Tensor]) -> None:
        """Cache routing weights supplied by the parent transformer block hook."""
        self._current_routing_weights = routing_weights

    def clear_current_routing(self) -> None:
        """Clear cached routing weights after a forward pass."""
        self._current_routing_weights = None

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dynamic expert composition.

        Args:
            x: Input tensor, shape (..., in_features)
            routing_weights: Token-level routing probabilities,
                           shape (..., num_experts).
                           If None, only base layer output is returned.

        Returns:
            Output tensor, shape (..., out_features)
        """
        if routing_weights is None:
            routing_weights = self._current_routing_weights

        # 1. Base model forward (frozen)
        z = self.base_layer(x)

        # If no routing (e.g., early layers or inference without adapters)
        if routing_weights is None:
            return z

        # 2. Shared projection: B @ x (computed ONCE)
        # x: (..., in_features), B: (rank, in_features)
        x_proj = x @ self.shared_B.t()  # (..., rank)

        # 3. Compute weighted expert outputs
        delta = torch.zeros(
            *x.shape[:-1], self.out_features,
            device=x.device, dtype=x.dtype,
        )

        for i, name in enumerate(self.expert_names):
            if name in self.expert_A:
                expert = self.expert_A[name]
                # Apply sparse mask and compute A_k @ B @ x
                A_sparse = expert.A * expert.sparse_mask  # (out_features, rank)
                expert_out = x_proj @ A_sparse.t()  # (..., out_features)
                expert_out = expert_out * expert.scaling

                # Weight by routing probability for this expert
                weight = routing_weights[..., i:i+1]  # (..., 1)
                delta = delta + expert_out * weight

        # 4. Residual addition (NO CLAMPING — orthogonality handles interference)
        return z + delta

    def forward_single_expert(
        self,
        x: torch.Tensor,
        expert_name: str,
    ) -> torch.Tensor:
        """
        Forward pass with a single expert (for evaluation/ablation).
        """
        z = self.base_layer(x)

        if expert_name not in self.expert_A:
            return z

        expert = self.expert_A[expert_name]
        x_proj = x @ self.shared_B.t()
        A_sparse = expert.A * expert.sparse_mask
        delta = x_proj @ A_sparse.t() * expert.scaling

        return z + delta
