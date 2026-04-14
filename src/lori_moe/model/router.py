"""
Token-Level MLP Router

Lightweight router deployed at each transformer layer that routes
each token's hidden state to a probability distribution over K domain experts.

This replaces Synapta's prompt-level GatedRouter with token-level granularity,
enabling mid-sequence domain switching during generation.

Architecture:
  h_t → LayerNorm → Linear(d, bottleneck) → SiLU → Linear(bottleneck, K) → Softmax → p(expert|token)

Design choices:
  - LayerNorm: normalizes hidden states across layers for stable routing
  - Bottleneck: reduces params from d×K to d×b + b×K (e.g., 2048×64 + 64×5 = 131K vs 10K)
  - SiLU activation: smooth gating, better gradient flow than ReLU
  - Noisy gating during training: prevents router collapse (Shazeer et al.)
  - Top-K selection: only activate K experts per token for efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TokenRouter(nn.Module):
    """
    Token-level expert router for a single transformer layer.
    
    Input: hidden states (batch, seq_len, hidden_dim)
    Output: routing weights (batch, seq_len, num_experts)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        bottleneck_dim: int = 64,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Router network
        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, num_experts, bias=False)

        # Initialize with small weights for stable early training
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)

        self._routing_entropy_ema = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute routing weights for each token.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            return_logits: if True, also return raw logits for loss computation

        Returns:
            routing_weights: (batch, seq_len, num_experts) — probabilities
            logits: (batch, seq_len, num_experts) — raw logits (if requested)
        """
        # Normalize
        h = self.norm(hidden_states.detach())  # Detach to prevent base model gradient flow

        # MLP routing
        logits = self.up_proj(F.silu(self.down_proj(h)))  # (B, S, K)

        # Add noise during training to prevent collapse
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-K selection
        if self.top_k < self.num_experts:
            routing_weights = self._top_k_softmax(logits)
        else:
            routing_weights = F.softmax(logits, dim=-1)

        # Track entropy for collapse detection
        with torch.no_grad():
            entropy = self._compute_entropy(routing_weights)
            self._routing_entropy_ema = 0.9 * self._routing_entropy_ema + 0.1 * entropy
            self._last_routing_weights = routing_weights.detach()

        if return_logits:
            return routing_weights, logits
        return routing_weights, None

    def _top_k_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Softmax with Top-K masking. Non-top-K experts get zero weight.
        """
        top_k_values, top_k_indices = logits.topk(self.top_k, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        
        # Softmax only over top-k (set others to -inf)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(masked_logits, dim=-1)

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute normalized entropy of routing distribution."""
        # Avoid log(0)
        probs_clamped = probs.clamp(min=1e-8)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=probs.dtype))
        normalized_entropy = (entropy / max_entropy).mean().item()
        return normalized_entropy

    @property
    def routing_entropy(self) -> float:
        """Current EMA of routing entropy. Should be > 0.3 to avoid collapse."""
        return self._routing_entropy_ema

    @property
    def is_collapsed(self) -> bool:
        """Check if router has collapsed to single expert."""
        return self._routing_entropy_ema < 0.3


class MultiLayerRouter(nn.Module):
    """
    Manages routers across all transformer layers.
    Optionally shares router weights across groups of layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_layers: int,
        bottleneck_dim: int = 64,
        top_k: int = 2,
        noise_std: float = 0.1,
        share_every: int = 1,  # Share router every N layers (1 = no sharing)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.share_every = share_every

        # Create routers (shared across groups if share_every > 1)
        num_unique_routers = (num_layers + share_every - 1) // share_every
        self.routers = nn.ModuleList([
            TokenRouter(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                bottleneck_dim=bottleneck_dim,
                top_k=top_k,
                noise_std=noise_std,
            )
            for _ in range(num_unique_routers)
        ])

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiLayerRouter: {num_unique_routers} unique routers "
            f"for {num_layers} layers, {total_params:,} total params"
        )

    def get_router(self, layer_idx: int) -> TokenRouter:
        """Get the router for a specific layer."""
        router_idx = layer_idx // self.share_every
        return self.routers[router_idx]

    def set_top_k(self, top_k: int) -> None:
        """Update the Top-K routing policy in-place for all routers."""
        for router in self.routers:
            router.top_k = max(1, min(top_k, router.num_experts))

    def set_noise_std(self, noise_std: float) -> None:
        """Update router noise in-place for all routers."""
        for router in self.routers:
            router.noise_std = max(0.0, noise_std)

    def get_all_entropies(self) -> dict:
        """Get routing entropy for all routers."""
        return {
            f"layer_{i * self.share_every}": router.routing_entropy
            for i, router in enumerate(self.routers)
        }

    def any_collapsed(self) -> bool:
        """Check if any router has collapsed."""
        return any(r.is_collapsed for r in self.routers)

    def get_total_params(self) -> int:
        """Total trainable parameters across all routers."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
