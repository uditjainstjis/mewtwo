"""
Shared Projection Matrix (Frozen B)

The mathematical backbone of LoRI-MoE. A single random Gaussian matrix B is shared
across ALL domain adapters. This exploits the Johnson-Lindenstrauss lemma:

  In high-dimensional spaces, random projections preserve pairwise distances
  with high probability. Therefore, domain-specific A matrices trained against
  a shared random B will produce updates in approximately orthogonal subspaces
  WITHOUT requiring explicit Stiefel manifold optimization.

The B matrix is:
  - Initialized ONCE from a deterministic seed
  - NEVER updated during training
  - Shared across all experts and all layers
  - Scaled by 1/sqrt(rank) to preserve activation magnitudes
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SharedProjection:
    """
    Manages the frozen shared B projection matrix.

    For a model with hidden_size d and adapter rank r:
      B ∈ R^{r × d}  (down-projection: maps from hidden to rank)

    In standard LoRA: ΔW = A @ B where A ∈ R^{d × r}, B ∈ R^{r × d}
    In LoRI-MoE: B is shared and frozen, A_k is domain-specific and sparse.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.rank = rank
        self.seed = seed
        self.device = device
        self.dtype = dtype

        # Generate the shared projection
        self._B = self._initialize_projection()

    def _initialize_projection(self) -> torch.Tensor:
        """
        Initialize B as a random Gaussian matrix with JL-appropriate scaling.

        B ~ N(0, 1/r) ensures that for any vector x:
          E[||Bx||²] = ||x||²
        preserving the geometry of the input space.
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        # Generate on CPU for reproducibility, then move
        B = torch.randn(
            self.rank,
            self.hidden_size,
            generator=generator,
            dtype=self.dtype,
        )

        # Scale by 1/sqrt(rank) — JL scaling
        B = B / (self.rank ** 0.5)

        B = B.to(device=self.device)
        logger.info(
            f"Shared B initialized: shape={B.shape}, "
            f"seed={self.seed}, "
            f"frobenius_norm={B.norm().item():.4f}"
        )
        return B

    @property
    def B(self) -> torch.Tensor:
        """Get the frozen B matrix. Always returns a non-grad tensor."""
        return self._B.detach()

    def save(self, path: Path):
        """Save the projection matrix and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "B": self._B.cpu(),
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "seed": self.seed,
        }, path)
        logger.info(f"Shared projection saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "SharedProjection":
        """Load a previously saved projection matrix."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls.__new__(cls)
        instance.hidden_size = data["hidden_size"]
        instance.rank = data["rank"]
        instance.seed = data["seed"]
        instance.device = device
        instance.dtype = dtype
        instance._B = data["B"].to(device=device, dtype=dtype)
        logger.info(
            f"Shared projection loaded from {path}: "
            f"shape={instance._B.shape}, seed={instance.seed}"
        )
        return instance

    def verify_orthogonality(self, num_samples: int = 100, dim_out: int = 768) -> dict:
        """
        Verify that random subspaces through B are approximately orthogonal.

        Generate random sparse A matrices and measure the cosine similarity
        of their composed ΔW = A @ B. Under JL, these should be near-zero.
        """
        # 1. Expected shift if B were perfectly orthogonal: A @ B @ x -> magnitude depends on A
        # Create two random mock abstract task updates (A1, A2) in adapter space
        cosine_sims = []
        for _ in range(num_samples):
            A1 = torch.randn(dim_out, self.rank, device=self.device, dtype=self._B.dtype)
            A2 = torch.randn(dim_out, self.rank, device=self.device, dtype=self._B.dtype)

            # Apply 80% sparsity
            mask1 = (torch.rand_like(A1) > 0.8).to(A1.dtype)
            mask2 = (torch.rand_like(A2) > 0.8).to(A2.dtype)
            A1 = A1 * mask1
            A2 = A2 * mask2

            # Compose ΔW = A @ B
            dW1 = (A1 @ self._B).flatten()
            dW2 = (A2 @ self._B).flatten()

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                dW1.unsqueeze(0), dW2.unsqueeze(0)
            ).item()
            cosine_sims.append(abs(cos_sim))

        import statistics
        result = {
            "mean_cosine_similarity": statistics.mean(cosine_sims),
            "max_cosine_similarity": max(cosine_sims),
            "std_cosine_similarity": statistics.stdev(cosine_sims),
            "num_samples": num_samples,
            "orthogonality_quality": "GOOD" if statistics.mean(cosine_sims) < 0.05 else "POOR",
        }
        logger.info(
            f"Orthogonality verification: mean_cos_sim={result['mean_cosine_similarity']:.6f}, "
            f"quality={result['orthogonality_quality']}"
        )
        return result


def get_shared_projection(
    hidden_size: int,
    rank: int,
    seed: int = 42,
    save_path: Optional[Path] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> SharedProjection:
    """
    Factory: get or create the shared B projection.
    Loads from disk if exists, otherwise creates and saves.
    """
    if save_path and Path(save_path).exists():
        return SharedProjection.load(save_path, device=device, dtype=dtype)

    proj = SharedProjection(
        hidden_size=hidden_size,
        rank=rank,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    if save_path:
        proj.save(save_path)
    return proj
