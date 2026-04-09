import random
from typing import Dict, List


def select_challengers(producer_names: List[str]) -> List[str]:
    """Assign deterministic challengers on single-machine for reproducibility."""
    base = ["ChallengerAgentX", "ChallengerAgentY"]
    if len(producer_names) <= 1:
        return base[:1]
    return base


def _jaccard(a: str, b: str) -> float:
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def detect_collusion(packets, similarity_threshold: float = 0.78):
    """
    Flag packets that appear too similar to others.
    A similarity spike suggests role collapse/collusion.
    """
    flags = [False] * len(packets)
    for i in range(len(packets)):
        for j in range(i + 1, len(packets)):
            s = _jaccard(packets[i].proposal_text, packets[j].proposal_text)
            if s >= similarity_threshold:
                flags[i] = True
                flags[j] = True
    return flags


def build_contradiction_brief(query: str, packets) -> str:
    """
    Build a compact contradiction task for challengers.
    """
    snippets = []
    for p in packets:
        claim = p.claims[0] if p.claims else p.proposal_text[:100]
        snippets.append(f"{p.agent_name}: {claim}")
    joined = "\n".join(snippets)
    return (
        "You are challenger. Find contradictions, unsupported leaps, and unverifiable claims.\n"
        f"Original query: {query}\n"
        f"Candidate claims:\n{joined}\n"
        "Return bullet list of contradictions only."
    )


def should_trigger_reaudit(query: str, collusion_flags: List[bool], round_idx: int) -> bool:
    """
    Randomized re-audit trigger to reduce stable collusion patterns.
    Deterministic per (query, round) for reproducibility.
    """
    if any(collusion_flags):
        return True
    seed = hash(f"{query}|{round_idx}") & 0xFFFFFFFF
    rng = random.Random(seed)
    return rng.random() < 0.2


def rotate_challengers(challengers: List[str], round_idx: int) -> List[str]:
    if not challengers:
        return challengers
    shift = round_idx % len(challengers)
    return challengers[shift:] + challengers[:shift]
