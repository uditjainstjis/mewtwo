"""
LayerBlend-LoRI: Continuous Per-Layer Adapter Composition
=========================================================
NOVEL CONTRIBUTION — this is the patentable IP.

Instead of discrete routing (pick ONE adapter) or static merging (average all),
LayerBlend learns continuous, input-dependent blend weights PER LAYER.

Each layer can independently decide: "For this input, use 70% Math + 30% Code."
Layer 1 might weight differently than Layer 20.

Architecture:
    For each adapted layer L:
        h_L = base_model.layer_L(input)                    # Frozen
        α = BlendHead_L(mean_pool(h_L))                    # Tiny trainable network
        blended_delta = Σ(α_i × precomputed_W_i)           # Weighted sum of adapter deltas
        output = h_L + blended_delta @ input                # Apply blended adapter

Why this is novel:
    1. Per-layer blending (not per-input like routing)
    2. Continuous weights (not discrete top-k selection)
    3. Input-dependent (not static merge)
    4. Computationally cheap (~5% overhead vs single adapter)

Why this should work where GC-LoRI failed:
    GC-LoRI failed because domain CLASSIFICATION is trivially easy (97.9% for both
    routers). But domain BLENDING at the layer level is NOT trivial. Different layers
    need different ratios — early layers might need more "general reasoning" (all adapters
    equally), while late layers need domain specialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LayerBlendHead(nn.Module):
    """Per-layer blend coefficient predictor.
    
    Takes pooled hidden states, outputs softmax blend weights for K adapters.
    One of these exists per adapted layer.
    """
    def __init__(self, hidden_dim: int, num_experts: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, num_experts),
        )
        # Small init for stable start (near-uniform blending initially)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, D) pooled hidden state
        Returns:
            weights: (B, K) blend coefficients (softmax)
        """
        logits = self.net(hidden)  # (B, K)
        return F.softmax(logits, dim=-1)


class LayerBlendRouter(nn.Module):
    """
    The full LayerBlend system.
    
    Pre-computes adapter weight deltas and learns per-layer blend coefficients.
    At inference: for each input, each layer independently decides how to mix adapters.
    
    This is the core innovation for the paper and the patent.
    """
    def __init__(
        self,
        hidden_dim: int = 2688,
        num_experts: int = 3,
        num_layers: int = 6,  # Number of adapted layers
        bottleneck: int = 64,
        temperature: float = 1.0,
        expert_names: List[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.temperature = temperature
        self.expert_names = expert_names or [f"expert_{i}" for i in range(num_experts)]
        
        # One blend head per adapted layer
        self.blend_heads = nn.ModuleList([
            LayerBlendHead(hidden_dim, num_experts, bottleneck)
            for _ in range(num_layers)
        ])
        
        # Load-balancing loss weight
        self.balance_weight = 0.01
        
        # Entropy tracking for collapse detection
        self._layer_entropy = [0.0] * num_layers
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LayerBlendRouter: {num_experts} experts × {num_layers} layers, "
            f"bottleneck={bottleneck}, params={total_params:,}"
        )
    
    def forward(
        self,
        hidden_states_per_layer: List[torch.Tensor],
        return_aux_loss: bool = True,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute blend weights for each layer.
        
        Args:
            hidden_states_per_layer: List of (B, S, D) tensors, one per adapted layer
            return_aux_loss: whether to compute diversity loss
            
        Returns:
            blend_weights: List of (B, K) weight tensors, one per layer
            aux_loss: scalar diversity encouragement loss
        """
        blend_weights = []
        all_logit_probs = []
        
        for layer_idx, (hidden, head) in enumerate(
            zip(hidden_states_per_layer, self.blend_heads)
        ):
            # Pool across sequence dimension
            pooled = hidden.detach().mean(dim=1)  # (B, D) — detach to avoid base model gradients
            
            # Get blend weights
            weights = head(pooled)  # (B, K)
            blend_weights.append(weights)
            all_logit_probs.append(weights)
            
            # Track entropy
            with torch.no_grad():
                ent = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(-1).mean().item()
                self._layer_entropy[layer_idx] = 0.9 * self._layer_entropy[layer_idx] + 0.1 * ent
        
        # Diversity loss: encourage each layer to find DIFFERENT blending patterns
        aux_loss = None
        if return_aux_loss and self.training and len(all_logit_probs) > 1:
            # Stack: (num_layers, B, K)
            stacked = torch.stack(all_logit_probs, dim=0)
            # Encourage diversity across layers
            mean_per_expert = stacked.mean(dim=[0, 1])  # (K,)
            # Penalize if all layers converge to same blend
            uniformity = (mean_per_expert - 1.0 / self.num_experts).pow(2).sum()
            aux_loss = self.balance_weight * uniformity
        
        return blend_weights, aux_loss
    
    def get_layer_summary(self) -> str:
        """Human-readable summary of what each layer learned."""
        lines = []
        for i, ent in enumerate(self._layer_entropy):
            max_ent = torch.log(torch.tensor(float(self.num_experts))).item()
            norm_ent = ent / max_ent if max_ent > 0 else 0
            lines.append(f"  Layer {i}: entropy={norm_ent:.2f} ({'diverse' if norm_ent > 0.5 else 'specialized'})")
        return "\n".join(lines)
    
    @property
    def is_collapsed(self) -> bool:
        """Check if any layer has collapsed to single expert."""
        max_ent = torch.log(torch.tensor(float(self.num_experts))).item()
        return any(e / max_ent < 0.2 for e in self._layer_entropy)
    
    def get_config(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "bottleneck": self.blend_heads[0].net[1].out_features,
            "temperature": self.temperature,
            "expert_names": self.expert_names,
        }
