"""
Gate-Conditioned LoRI Router (GC-LoRI)

The core innovation of the Nemotron × LoRI-MoE project.

Instead of a blind external router that independently decides how to mix
domain adapters, this module reads Nemotron's internal MoE routing signals
and uses them to condition external adapter composition.

Innovation: The internal router already encodes "what kind of processing
this token needs." We translate that signal into "which external reasoning
adapter to apply and how strongly."

Architecture:
  internal_signal = extract(NemotronMoE.router.topk_weights, entropy)
  combined = concat(hidden_proj(h), signal_proj(internal_signal))
  adapter_weights = routing_head(combined) → softmax → (num_external_experts,)

Why this is novel:
  - No published work uses internal MoE routing to control external adapter composition
  - External adapters specialize on what the base model can't already do (residual reasoning)
  - Avoids the "double routing" problem of stacking two independent expert systems

Reference: Extends Switch Transformer (Fedus et al., 2022) load-balancing loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GateConditionedRouter(nn.Module):
    """
    Routes tokens to external LoRI adapters using internal MoE signals.

    Inputs:
        hidden_states: (B, S, D) — current hidden representation
        internal_routing: Dict containing:
            - 'top_k_weights': (B, S, internal_K) — per-token internal expert weights
            - 'entropy': (B, S) — routing entropy per token

    Output:
        adapter_weights: (B, S, num_external_experts) — how to mix LoRI adapters
        aux_loss: load-balancing loss (only in training)
    """

    def __init__(
        self,
        hidden_dim: int = 2688,
        num_external_experts: int = 3,
        internal_top_k: int = 8,
        bottleneck_dim: int = 128,
        top_k: int = 2,
        noise_std: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_external_experts = num_external_experts
        self.internal_top_k = internal_top_k
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight

        # Internal signal projection
        internal_signal_dim = internal_top_k + 1  # top-k weights + entropy
        half_bn = bottleneck_dim // 2

        self.signal_proj = nn.Sequential(
            nn.Linear(internal_signal_dim, half_bn),
            nn.SiLU(),
        )

        # Hidden state projection
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, half_bn, bias=False),
            nn.SiLU(),
        )

        # Combined routing head
        self.routing_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(bottleneck_dim, num_external_experts),
        )

        # Small init for stable early training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._entropy_ema = 0.0
        logger.info(
            f"GateConditionedRouter: {num_external_experts} external experts, "
            f"bottleneck={bottleneck_dim}, internal_top_k={internal_top_k}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        internal_routing: Dict[str, torch.Tensor],
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (B, S, D) from the base model
            internal_routing: dict with 'top_k_weights' (B, S, K) and 'entropy' (B, S)
            return_aux_loss: whether to compute load-balancing loss

        Returns:
            adapter_weights: (B, S, num_external_experts)
            aux_loss: scalar or None
        """
        B, S, D = hidden_states.shape

        # Extract and normalize internal signals
        top_k_weights = internal_routing["top_k_weights"]  # (B, S, internal_K)
        entropy = internal_routing["entropy"]  # (B, S)

        # Pad/truncate top_k_weights to expected internal_top_k
        curr_b, curr_s, curr_k = top_k_weights.shape
        if curr_k < self.internal_top_k:
            padding = torch.zeros(
                curr_b, curr_s, self.internal_top_k - curr_k,
                device=top_k_weights.device, dtype=top_k_weights.dtype
            )
            top_k_weights = torch.cat([top_k_weights, padding], dim=-1)
        elif curr_k > self.internal_top_k:
            top_k_weights = top_k_weights[..., : self.internal_top_k]

        # Combine internal signals
        internal_signal = torch.cat(
            [top_k_weights, entropy.unsqueeze(-1)], dim=-1
        )  # (B, S, internal_K+1)

        # Project both streams (detach hidden to prevent base model gradient flow)
        signal_emb = self.signal_proj(internal_signal)  # (B, S, half_bn)
        hidden_emb = self.hidden_proj(hidden_states.detach())  # (B, S, half_bn)

        # Fuse
        combined = torch.cat([signal_emb, hidden_emb], dim=-1)  # (B, S, bottleneck)

        # Route
        logits = self.routing_head(combined)  # (B, S, num_external_experts)

        # Add noise during training to prevent collapse
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Top-K softmax
        if self.top_k < self.num_external_experts:
            router_weights = self._top_k_softmax(logits)
        else:
            router_weights = F.softmax(logits, dim=-1)

        # Aux loss: load balancing (Switch Transformer style)
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._load_balance_loss(router_weights, logits)

        # Track entropy EMA for collapse detection
        with torch.no_grad():
            probs = router_weights.clamp(min=1e-8)
            ent = -(probs * probs.log()).sum(-1).mean().item()
            max_ent = torch.log(
                torch.tensor(self.num_external_experts, dtype=torch.float32)
            ).item()
            self._entropy_ema = 0.9 * self._entropy_ema + 0.1 * (ent / max_ent)

        return router_weights, aux_loss

    def _top_k_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax with Top-K masking."""
        top_k_vals, top_k_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, top_k_idx, 1.0)
        masked = logits.masked_fill(mask == 0, float("-inf"))
        return F.softmax(masked, dim=-1)

    def _load_balance_loss(
        self, weights: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer load-balance loss."""
        f = weights.mean(dim=[0, 1])  # fraction routed to each expert
        p = F.softmax(logits, dim=-1).mean(dim=[0, 1])  # mean prob per expert
        return self.load_balance_weight * self.num_external_experts * (f * p).sum()

    @property
    def routing_entropy(self) -> float:
        """Current EMA of normalized routing entropy. Should be > 0.3."""
        return self._entropy_ema

    @property
    def is_collapsed(self) -> bool:
        """Check if router has collapsed to a single expert."""
        return self._entropy_ema < 0.3

    def get_config(self) -> dict:
        """Serialize config for checkpoint saving."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_external_experts": self.num_external_experts,
            "internal_top_k": self.internal_top_k,
            "bottleneck_dim": self.signal_proj[0].out_features * 2,
            "top_k": self.top_k,
            "noise_std": self.noise_std,
            "load_balance_weight": self.load_balance_weight,
        }
