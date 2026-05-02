from dataclasses import dataclass
from typing import List


@dataclass
class Verdict:
    name: str
    passed: bool
    severity: str
    reason: str
    score: float


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def verify_logic(query: str, packets) -> Verdict:
    # Heuristic logic guard: reject if all packets are extremely short.
    avg_claims = sum(len(p.claims) for p in packets) / max(1, len(packets))
    passed = avg_claims >= 2
    score = min(1.0, avg_claims / 4.0)
    reason = "sufficient_reasoning_depth" if passed else "insufficient_reasoning_depth"
    return Verdict(name="logic", passed=passed, severity="critical", reason=reason, score=score)


def verify_factual(query: str, packets) -> Verdict:
    # Factual proxy: require at least one concrete evidence ref per packet.
    refs_ok = all(len(p.evidence_refs) > 0 for p in packets)
    score = 1.0 if refs_ok else 0.3
    reason = "evidence_refs_present" if refs_ok else "missing_evidence_refs"
    return Verdict(name="factual", passed=refs_ok, severity="critical", reason=reason, score=score)


def verify_safety(query: str, packets) -> Verdict:
    q = (query or "").lower()
    blocked = _contains_any(q, ["exploit", "malware", "harm", "illegal"])
    safe_text = not any(_contains_any(p.proposal_text, ["weapon", "exploit code", "steal"]) for p in packets)
    passed = (not blocked) and safe_text
    score = 1.0 if passed else 0.1
    reason = "safe_content" if passed else "unsafe_or_disallowed_content"
    return Verdict(name="safety", passed=passed, severity="critical", reason=reason, score=score)


def verify_reproducibility(query: str, packets) -> Verdict:
    # Require at least one packet with procedural wording.
    procedural = any(_contains_any(p.proposal_text, ["step", "phase", "measure", "metric", "test"]) for p in packets)
    score = 0.9 if procedural else 0.4
    reason = "reproducible_steps_present" if procedural else "missing_reproducible_steps"
    return Verdict(name="reproducibility", passed=procedural, severity="non_critical", reason=reason, score=score)
