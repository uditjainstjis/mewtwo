"""
Loss Functions for LoRI-MoE

1. Standard Causal LM Cross-Entropy Loss (from base model)
2. Load-Balancing Auxiliary Loss (prevents router collapse)
3. Combined Loss with configurable weighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LoadBalancingLoss(nn.Module):
    """
    Auxiliary loss to prevent router collapse.

    From Shazeer et al. (Switch Transformers):
      L_aux = α * K * Σ_k (f_k · P_k)

    Where:
      f_k = fraction of tokens routed to expert k
      P_k = mean router probability for expert k
      K   = number of experts

    A perfectly balanced router has L_aux = α (each expert gets 1/K tokens).
    A collapsed router has L_aux ≈ α * K (one expert gets everything).

    Minimizing this loss encourages uniform expert utilization.
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        routing_weights: torch.Tensor,  # (batch, seq_len, num_experts)
        attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute load-balancing loss.

        Returns:
            loss: scalar loss value
            stats: dictionary with monitoring metrics
        """
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            routing_weights = routing_weights * mask
            num_tokens = attention_mask.sum().float()
        else:
            num_tokens = routing_weights.shape[0] * routing_weights.shape[1]

        # f_k: fraction of tokens where expert k has highest weight
        # Using soft version: average routing probability per expert
        # This is differentiable unlike hard assignment
        f = routing_weights.sum(dim=(0, 1)) / num_tokens  # (K,)

        # P_k: mean routing probability per expert
        P = routing_weights.mean(dim=(0, 1))  # (K,)

        # Auxiliary loss
        loss = self.weight * self.num_experts * (f * P).sum()

        # Monitoring stats
        with torch.no_grad():
            max_load = f.max().item()
            min_load = f.min().item()
            balance_ratio = min_load / (max_load + 1e-8)

            stats = {
                "load_balance_loss": loss.item(),
                "max_expert_load": max_load,
                "min_expert_load": min_load,
                "balance_ratio": balance_ratio,  # 1.0 = perfect, 0.0 = collapsed
                "expert_loads": f.detach().cpu().tolist(),
            }

        return loss, stats


class LoRIMoELoss(nn.Module):
    """
    Combined loss for LoRI-MoE training.

    L_total = L_CE + α * L_aux

    During adapter training: only L_CE (router is frozen)
    During router training: L_CE + L_aux (adapters are frozen)
    """

    def __init__(
        self,
        num_experts: int,
        load_balance_weight: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        self.lb_loss = LoadBalancingLoss(
            num_experts=num_experts,
            weight=load_balance_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,  # (batch, seq_len, vocab_size)
        labels: torch.Tensor,  # (batch, seq_len)
        routing_weights: Optional[torch.Tensor] = None,  # (batch, seq_len, K)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            total_loss: scalar
            stats: dictionary with all loss components
        """
        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy loss
        ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        stats = {"ce_loss": ce.item()}
        total = ce

        # Load-balancing loss (only during router training)
        if routing_weights is not None:
            shift_routing = routing_weights[..., :-1, :].contiguous()
            shift_mask = attention_mask[..., :-1].contiguous() if attention_mask is not None else None
            lb, lb_stats = self.lb_loss(shift_routing, shift_mask)
            total = total + lb
            stats.update(lb_stats)

        stats["total_loss"] = total.item()
        return total, stats
