import torch
from typing import Dict, List, Optional
import numpy as np

def compute_orthogonal_complement_projection(subspaces: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of subspace projection matrices {P_1, ..., P_k}, computes the
    projection onto the orthogonal complement of their union.
    
    If P_total is the projection onto span(U_1 U U_2 ... U_k),
    then this returns I - P_total.
    """
    if not subspaces:
        # Return identity conceptually, but effectively return None to signal no projection
        return None
        
    # We can compute this by concatenating the basis vectors of all subspaces,
    # performing an SVD, and using the resulting left singular vectors to build the projection.
    # But since we just have the P matrices, we can sum them and find the null space of the sum.
    # P_sum = sum(P_i)
    # The null space of P_sum is exactly the intersection of the null spaces of P_i.
    
    P_sum = sum(subspaces)
    # Use eigh since P_sum is symmetric positive semi-definite
    eigenvalues, eigenvectors = torch.linalg.eigh(P_sum)
    
    # The eigenvectors corresponding to eigenvalues near 0 form the basis 
    # of the orthogonal complement.
    tol = 1e-6
    null_space_basis = eigenvectors[:, eigenvalues < tol]
    
    if null_space_basis.shape[1] == 0:
        # The union of subspaces spans the entire space; orthogonal complement is empty.
        device = subspaces[0].device
        n = subspaces[0].shape[0]
        return torch.zeros((n, n), device=device)
        
    P_complement = null_space_basis @ null_space_basis.T
    return P_complement


def get_subspace_projection(delta_w: torch.Tensor, energy_threshold: float = 0.95) -> torch.Tensor:
    """
    Computes the projection matrix P onto the dominant subspace of a weight matrix ΔW.
    
    Args:
        delta_w: The weight matrix (ΔW = B @ A).
        energy_threshold: Fraction of total singular value energy to retain.
        
    Returns:
        P: The projection matrix (U_k @ U_k.T).
    """
    U, S, _ = torch.linalg.svd(delta_w, full_matrices=False)
    
    # Find rank k that captures the desired energy threshold
    energy = S.cumsum(dim=0) / S.sum()
    k = (energy >= energy_threshold).nonzero()[0].item() + 1
    
    U_k = U[:, :k]
    P = U_k @ U_k.T
    return P


def subspace_aware_compose(x: torch.Tensor,
                           base_out: torch.Tensor,
                           adapters: Dict[str, Dict[str, torch.Tensor]],
                           routing_weights: Dict[str, float],
                           clamp_ratio: float = 1.0) -> torch.Tensor:
    """
    Subspace-Aware Composition (SAC)
    
    Instead of standard additive composition:
        h = z + Σ w_i * ΔW_i(x)
        
    SAC projects each adapter's update into the subspace orthogonal to all OTHER
    active adapters. This prevents destructive interference where multiple adapters
    collide in the same representation space.
    
    h = z + Σ w_i * (P_{⊥other} @ ΔW_i @ x)
    
    Args:
        x: Input activations to the layer [batch, in_features]
        base_out: Output of the base model layer [batch, out_features]
        adapters: Dict mapping adapter names to their A and B matrices.
                  e.g., {'math': {'A': tensor, 'B': tensor}, ...}
        routing_weights: Dict mapping adapter names to their scalar composition weight (w_i).
        clamp_ratio: Max magnitude ratio relative to base_out to prevent representation collapse.
        
    Returns:
        Composed layer output [batch, out_features]
    """
    # 1. Compute projection matrices for each active adapter
    projections = {}
    raw_updates = {}
    
    for name, matrices in adapters.items():
        weight = routing_weights.get(name, 0.0)
        if weight <= 0.0:
            continue
            
        A, B = matrices['A'], matrices['B']
        delta_w = B @ A  # [out_features, in_features]
        
        # We compute the subspace projection based on the weight matrix ΔW
        # P projects onto the column space of ΔW
        projections[name] = get_subspace_projection(delta_w)
        
        # Compute raw activation update mapping: ΔW_i(x)
        # Note: A is [r, in], B is [out, r], x is [batch, in] (typically we do x @ A.T @ B.T)
        # Assuming linear layer behavior: y = x @ W.T
        # Update is x @ A.T @ B.T
        raw_updates[name] = (x @ A.T) @ B.T  # [batch, out_features]

    active_names = list(projections.keys())
    if not active_names:
        return base_out
        
    if len(active_names) == 1:
        # Standard composition if only 1 adapter is active
        name = active_names[0]
        update = routing_weights[name] * raw_updates[name]
        return base_out + update

    # 2. Compute mutually orthogonal updates
    sac_update = torch.zeros_like(base_out)
    
    for i, name_i in enumerate(active_names):
        # We want to project adapter i's update away from the union of all OTHER adapters' subspaces
        other_projections = [projections[name_j] for name_j in active_names if name_j != name_i]
        
        # P_complement is the projection onto the subspace orthogonal to all other adapters
        P_complement = compute_orthogonal_complement_projection(other_projections)
        
        raw_update_i = raw_updates[name_i]  # [batch, out_features]
        
        # Apply orthogonal projection: y_proj = y @ P^T (since P is symmetric, P^T = P)
        if P_complement is not None:
            # We must be careful: P_complement is [out, out], raw_update_i is [batch, out]
            # Projected update = raw_update_i @ P_complement
            projected_update_i = raw_update_i @ P_complement
        else:
            projected_update_i = raw_update_i
            
        w_i = routing_weights[name_i]
        sac_update += w_i * projected_update_i

    # 3. Norm clamping
    update_norm = torch.norm(sac_update, dim=-1, keepdim=True)
    base_norm = torch.norm(base_out, dim=-1, keepdim=True)
    
    # We allow the update to be at most clamp_ratio * base_norm
    max_allowed_norm = clamp_ratio * base_norm
    
    # Scale down if it exceeds the max allowed norm
    scale = torch.clamp(max_allowed_norm / (update_norm + 1e-8), max=1.0)
    sac_update_clamped = sac_update * scale

    return base_out + sac_update_clamped
