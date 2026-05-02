"""
Confidence-Gated Router (v2)
============================
Dynamically selects K=1 or K=2 adapters based on the router's
probability distribution over domains.

Design rationale (from v1 analysis):
  - v1 always used K=2 for AdaptiveClamp, injecting a redundant
    (often wrong) second adapter on single-domain queries.
  - v2 only activates K=2 when the router is genuinely uncertain
    between two strong domain candidates.
"""

from typing import Dict, List, Tuple
import numpy as np


class GatedRouter:
    """
    Wraps an upstream domain-probability source and decides whether
    the query needs K=1 (single adapter) or K=2 (multi-adapter composition).
    """

    def __init__(
        self,
        tau_single: float = 0.70,
        tau_multi: float = 0.25,
        gap_threshold: float = 0.40,
    ):
        """
        Args:
            tau_single:    If top-1 prob >= tau_single AND gap > gap_threshold, use K=1.
            tau_multi:     Minimum top-2 probability required to activate K=2.
            gap_threshold: If (top1 - top2) > gap_threshold, force K=1.
        """
        self.tau_single = tau_single
        self.tau_multi = tau_multi
        self.gap_threshold = gap_threshold

    def route(
        self,
        domain_probs: Dict[str, float],
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Given a dict of domain -> probability from the upstream router,
        decide K and return selected experts with a reason string.

        Returns:
            (selected_experts, decision_reason)
            where selected_experts is a list of (domain_name, probability) tuples.
        """
        sorted_domains = sorted(
            domain_probs.items(), key=lambda x: x[1], reverse=True
        )
        top1_name, top1_prob = sorted_domains[0]
        top2_name, top2_prob = (
            sorted_domains[1] if len(sorted_domains) > 1 else ("", 0.0)
        )

        gap = top1_prob - top2_prob

        # Decision logic:
        # 1. High confidence in top-1 AND large gap → K=1 (single adapter)
        # 2. Top-2 is also strong AND gap is small → K=2 (multi-adapter)
        # 3. Default fallback → K=1 (conservative)
        if top1_prob >= self.tau_single and gap > self.gap_threshold:
            return (
                [(top1_name, top1_prob)],
                f"K=1: top1={top1_prob:.3f}, gap={gap:.3f}",
            )
        elif top2_prob >= self.tau_multi and gap <= self.gap_threshold:
            return (
                [(top1_name, top1_prob), (top2_name, top2_prob)],
                f"K=2: top2={top2_prob:.3f}, gap={gap:.3f}",
            )
        else:
            return (
                [(top1_name, top1_prob)],
                f"K=1 (fallback): top1={top1_prob:.3f}",
            )

    def compute_multi_domain_score(
        self, domain_probs: Dict[str, float]
    ) -> float:
        """
        Normalized entropy of the routing distribution.
        High values → query likely spans multiple domains.

        Returns:
            float in [0, 1]. 0 = perfectly single-domain, 1 = uniform over all domains.
        """
        probs = np.array(list(domain_probs.values()), dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(domain_probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
