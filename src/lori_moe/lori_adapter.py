"""
LoRI Adapter — Low-Rank Adaptation with Reduced Interference

Each domain expert consists of:
  - A_k ∈ R^{d × r} : Domain-specific, TRAINABLE, SPARSE matrix
  - B   ∈ R^{r × d} : Shared, FROZEN, random Gaussian projection

The forward pass for expert k:
  ΔW_k(x) = (A_k @ B) @ x = A_k @ (B @ x)

Key properties:
  1. B is shared → all experts project into the same subspace basis
  2. A_k is sparse (80% zeros) → domain updates are non-overlapping
  3. Johnson-Lindenstrauss → approximately orthogonal compositions
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LoRIExpertAdapter(nn.Module):
    """
    A single domain expert adapter with sparse A matrix and shared frozen B.

    During training:
      - B is frozen (never updated)
      - A is trained with a binary sparse mask applied after each update

    At inference:
      - The adapter computes ΔW(x) = α/r * A @ B @ x
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        alpha: float = 64.0,
        sparsity: float = 0.8,
        domain_name: str = "unknown",
    ):
        super().__init__()
        self.in_features = in_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.sparsity = sparsity
        self.domain_name = domain_name

        # Trainable A matrix (domain-specific)
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

        # Sparse mask — initialized randomly, frozen after creation
        sparse_mask = (torch.rand(in_features, rank) >= sparsity).float()
        self.register_buffer("sparse_mask", sparse_mask)

        # Apply mask immediately
        with torch.no_grad():
            self.A.data *= self.sparse_mask

        # Track stats
        active_params = sparse_mask.sum().item()
        total_params = in_features * rank
        logger.debug(
            f"LoRI Expert '{domain_name}': "
            f"{active_params:.0f}/{total_params:.0f} active params "
            f"({100 * active_params / total_params:.1f}%)"
        )

    def forward(self, projected_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            projected_input: B @ x, shape (..., rank).
                             Already projected through shared B.

        Returns:
            ΔW(x) = scaling * A @ (B @ x), shape (..., in_features)
        """
        # Apply sparse mask to ensure sparsity is maintained
        A_sparse = self.A * self.sparse_mask
        # projected_input: (..., rank), A_sparse: (in_features, rank)
        # output: (..., in_features)
        return self.scaling * (projected_input @ A_sparse.t())

    def enforce_sparsity(self):
        """Re-apply sparse mask after optimizer step. Call in training loop."""
        with torch.no_grad():
            self.A.data *= self.sparse_mask

    def get_active_param_count(self) -> int:
        """Number of non-zero trainable parameters."""
        return int(self.sparse_mask.sum().item())

    def apply_dare_sparsification(self, drop_rate: float = 0.3):
        """
        Post-training DARE (Drop And Rescale) sparsification.

        Drops additional parameters by magnitude and rescales the remainder.
        This further orthogonalizes the adapter on top of the LoRI guarantee.
        """
        with torch.no_grad():
            # Only consider currently active parameters
            active = (self.sparse_mask > 0)
            magnitudes = self.A.data.abs()

            # Find the drop_rate-th percentile among active params
            active_magnitudes = magnitudes[active]
            if active_magnitudes.numel() == 0:
                return

            threshold = torch.quantile(active_magnitudes, drop_rate)

            # Create new mask: keep params above threshold
            dare_mask = (magnitudes > threshold).float() * self.sparse_mask
            dropped = self.sparse_mask.sum() - dare_mask.sum()

            # Rescale remaining params to preserve expected output magnitude
            scale = self.sparse_mask.sum() / (dare_mask.sum() + 1e-8)
            self.A.data *= scale

            # Update mask
            self.sparse_mask.copy_(dare_mask)
            self.A.data *= self.sparse_mask

            logger.info(
                f"DARE applied to '{self.domain_name}': "
                f"dropped {dropped:.0f} params, "
                f"rescale factor {scale:.3f}, "
                f"remaining active: {dare_mask.sum():.0f}"
            )


class LoRIExpertBank(nn.Module):
    """
    Collection of all domain expert adapters.

    Manages K domain experts, each with their own sparse A matrix,
    sharing a single frozen B projection.
    """

    def __init__(
        self,
        expert_configs: Dict[str, dict],
        in_features: int,
        rank: int,
        alpha: float = 64.0,
        sparsity: float = 0.8,
    ):
        super().__init__()
        self.expert_names = list(expert_configs.keys())
        self.num_experts = len(self.expert_names)

        self.experts = nn.ModuleDict({
            name: LoRIExpertAdapter(
                in_features=in_features,
                rank=rank,
                alpha=alpha,
                sparsity=sparsity,
                domain_name=name,
            )
            for name in self.expert_names
        })

        total_active = sum(e.get_active_param_count() for e in self.experts.values())
        total_possible = in_features * rank * self.num_experts
        logger.info(
            f"LoRI Expert Bank: {self.num_experts} experts, "
            f"{total_active:,} active params / {total_possible:,} total "
            f"({100 * total_active / total_possible:.1f}%)"
        )

    def forward(
        self,
        projected_input: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the dynamically routed expert composition.

        Args:
            projected_input: B @ x, shape (batch, seq_len, rank)
            routing_weights: Router output, shape (batch, seq_len, num_experts)

        Returns:
            Combined expert output, shape (batch, seq_len, in_features)
        """
        # Compute each expert's output and weight it
        expert_outputs = []
        for i, name in enumerate(self.expert_names):
            expert_out = self.experts[name](projected_input)  # (B, S, d)
            weight = routing_weights[..., i:i+1]  # (B, S, 1)
            expert_outputs.append(expert_out * weight)

        return sum(expert_outputs)

    def enforce_all_sparsity(self):
        """Re-apply sparse masks for all experts."""
        for expert in self.experts.values():
            expert.enforce_sparsity()

    def save_experts(self, save_dir: Path):
        """Save all expert adapters."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, expert in self.experts.items():
            path = save_dir / f"{name}_adapter.pt"
            torch.save({
                "state_dict": expert.state_dict(),
                "domain_name": name,
                "in_features": expert.in_features,
                "rank": expert.rank,
                "alpha": expert.alpha,
                "sparsity": expert.sparsity,
            }, path)
        logger.info(f"Saved {self.num_experts} expert adapters to {save_dir}")

    @classmethod
    def load_experts(
        cls,
        save_dir: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LoRIExpertBank":
        """Load previously saved expert adapters."""
        save_dir = Path(save_dir)
        expert_files = sorted(save_dir.glob("*_adapter.pt"))

        expert_configs = {}
        experts_data = {}
        for f in expert_files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            name = data["domain_name"]
            expert_configs[name] = {}
            experts_data[name] = data

        if not expert_configs:
            raise ValueError(f"No adapter files found in {save_dir}")

        # Get dimensions from first expert
        first = next(iter(experts_data.values()))
        bank = cls(
            expert_configs=expert_configs,
            in_features=first["in_features"],
            rank=first["rank"],
            alpha=first["alpha"],
            sparsity=first["sparsity"],
        )

        for name, data in experts_data.items():
            bank.experts[name].load_state_dict(data["state_dict"])

        bank = bank.to(device=device, dtype=dtype)
        logger.info(f"Loaded {len(expert_configs)} experts from {save_dir}")
        return bank
