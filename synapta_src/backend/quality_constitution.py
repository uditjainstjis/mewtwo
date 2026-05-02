from typing import Dict, List


CRITICAL_VETO_DIMENSIONS = {"logic", "factual", "safety"}


class QualityConstitution:
    """Hard-veto constitution with strict anti-compromise semantics."""

    def __init__(self, min_soft_score: float = 0.65, max_uncertainty: float = 0.6):
        self.min_soft_score = min_soft_score
        self.max_uncertainty = max_uncertainty

    def evaluate(self, verdicts, packets) -> Dict:
        veto_reasons: List[str] = []
        for v in verdicts:
            if (v.name in CRITICAL_VETO_DIMENSIONS) and (not v.passed):
                veto_reasons.append(f"{v.name}_veto:{v.reason}")

        if any(p.uncertainty > self.max_uncertainty for p in packets):
            veto_reasons.append("uncertainty_veto:agent_uncertainty_too_high")

        evidence_ok = all((len(p.evidence_refs) > 0) and (len(p.test_artifacts) > 0) for p in packets)
        if not evidence_ok:
            veto_reasons.append("evidence_veto:missing_references_or_artifacts")

        mean_score = sum(v.score for v in verdicts) / max(1, len(verdicts))
        if mean_score < self.min_soft_score:
            veto_reasons.append(f"soft_score_veto:mean_score={mean_score:.3f}")

        return {
            "passed": len(veto_reasons) == 0,
            "mean_score": mean_score,
            "veto_reasons": veto_reasons,
        }
