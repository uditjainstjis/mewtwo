import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent_cluster import EvidencePacket
from verifiers import verify_logic, verify_factual, verify_safety, verify_reproducibility


@dataclass
class TrajectoryCandidate:
    name: str
    stages: List[Dict]
    text: str
    latency_s: float
    packet: EvidencePacket
    scores: Dict[str, float]
    passed: bool
    veto_reasons: List[str]


def _mk_prompt(question: str, prefix_answer: str = "") -> str:
    # Qwen chat template style consistent with rest of repo.
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{prefix_answer}"


def run_staged_generation(engine, question: str, stages: List[Dict]) -> Tuple[str, float]:
    """
    stages: [{routing_weights: dict|None, max_tokens: int}]
    Each stage continues by appending prior output into the prompt.
    """
    full_answer = ""
    total_start = time.time()
    for st in stages:
        prompt = _mk_prompt(question, prefix_answer=full_answer)
        part, _ = engine.generate(
            prompt,
            routing_weights=st.get("routing_weights"),
            max_tokens=int(st.get("max_tokens", 80)),
        )
        # engine.generate returns full continuation from prompt; we append new text.
        # To avoid duplication, take suffix relative to prior prefix when possible.
        if part.startswith(full_answer):
            delta = part[len(full_answer) :]
        else:
            delta = part
        full_answer += delta
        if len(full_answer) > 4000:
            break
    return full_answer, (time.time() - total_start)


def score_candidate(question: str, answer: str) -> Dict[str, float]:
    pkt = EvidencePacket(
        agent_name="TrajectoryCandidate",
        role="composer",  # lightweight typing; verifiers only read proposal_text/claims/evidence fields
        proposal_text=answer,
        rationale=answer[:160],
        claims=[c.strip() for c in answer.split(".") if c.strip()][:6],
        evidence_refs=["internal:trajectory"],
        uncertainty=0.35,
        test_artifacts=["artifact:trajectory:staged_generation"],
        latency_s=0.0,
    )
    verdicts = [
        verify_logic(question, [pkt]),
        verify_factual(question, [pkt]),
        verify_safety(question, [pkt]),
        verify_reproducibility(question, [pkt]),
    ]
    scores = {
        "logic": verdicts[0].score,
        "factual": verdicts[1].score,
        "safety": verdicts[2].score,
        "repro": verdicts[3].score,
        "mean": sum(v.score for v in verdicts) / len(verdicts),
        "passed": 1.0 if all((v.passed if v.name in ("logic", "factual", "safety") else True) for v in verdicts) else 0.0,
    }
    return scores


def select_best(candidates: List[TrajectoryCandidate]) -> TrajectoryCandidate:
    ranked = sorted(
        candidates,
        key=lambda c: (
            1 if c.passed else 0,
            c.scores.get("mean", 0.0),
            c.scores.get("repro", 0.0),
            c.scores.get("logic", 0.0),
            -c.latency_s,
        ),
        reverse=True,
    )
    return ranked[0]


def build_sats_candidates(orchestrator, item: Dict, use_oracle_domains: bool = False) -> List[Tuple[str, List[Dict]]]:
    """
    Novel hypothesis: sequential expert trajectories.
    We generate multiple staged plans:
      - single->single switching
      - mix->single refinement
      - single->mix
    """
    question = item["question"]
    all_domains = list(orchestrator.registry.keys())

    if use_oracle_domains:
        doms = item.get("required_adapters", item.get("domains", []))
        dom1 = doms[0] if doms else all_domains[0]
        dom2 = doms[1] if len(doms) > 1 else None
        weights = {d: 0.0 for d in all_domains}
        weights[dom1] = 1.0
        weights2 = {d: 0.0 for d in all_domains}
        if dom2:
            weights2[dom2] = 1.0
        mix = {d: 0.0 for d in all_domains}
        if dom2:
            mix[dom1] = 0.5
            mix[dom2] = 0.5
        else:
            mix[dom1] = 1.0
    else:
        w_top2, doms_found, _ = orchestrator.route_top2(question)
        dom1 = doms_found[0]
        dom2 = doms_found[1]
        weights = {d: 0.0 for d in all_domains}
        weights[dom1] = 1.0
        weights2 = {d: 0.0 for d in all_domains}
        if dom2:
            weights2[dom2] = 1.0
        mix = {d: 0.0 for d in all_domains}
        if dom2:
            mix[dom1] = 0.5
            mix[dom2] = 0.5
        else:
            mix[dom1] = 1.0

    # Trajectory library (more novel): deliberate -> execute with sequential experts.
    # Stage 1: base/no-adapter or mix produces a structured plan; Stage 2/3 execute with experts.
    no_adapter = None
    return [
        (
            "SATS_deliberate_base_then_single_single",
            [
                {"routing_weights": no_adapter, "max_tokens": 70},
                {"routing_weights": weights, "max_tokens": 90},
                {"routing_weights": weights2 if dom2 else weights, "max_tokens": 90},
            ],
        ),
        (
            "SATS_deliberate_mix_then_single_single",
            [
                {"routing_weights": mix, "max_tokens": 80},
                {"routing_weights": weights, "max_tokens": 90},
                {"routing_weights": weights2 if dom2 else weights, "max_tokens": 90},
            ],
        ),
        (
            "SATS_single_single_then_mix_refine",
            [
                {"routing_weights": weights, "max_tokens": 90},
                {"routing_weights": weights2 if dom2 else weights, "max_tokens": 90},
                {"routing_weights": mix, "max_tokens": 70},
            ],
        ),
    ]


def run_sats(engine, orchestrator, item: Dict, use_oracle_domains: bool = False) -> Dict:
    cdefs = build_sats_candidates(orchestrator, item, use_oracle_domains=use_oracle_domains)
    results: List[TrajectoryCandidate] = []
    for name, stages in cdefs:
        text, latency = run_staged_generation(engine, item["question"], stages)
        scores = score_candidate(item["question"], text)
        passed = scores.get("passed", 0.0) >= 1.0
        pkt = EvidencePacket(
            agent_name=name,
            role="composer",
            proposal_text=text,
            rationale=text[:160],
            claims=[c.strip() for c in text.split(".") if c.strip()][:6],
            evidence_refs=["internal:sats"],
            uncertainty=0.35,
            test_artifacts=["artifact:sats:trajectory"],
            latency_s=latency,
        )
        results.append(
            TrajectoryCandidate(
                name=name,
                stages=stages,
                text=text,
                latency_s=latency,
                packet=pkt,
                scores=scores,
                passed=passed,
                veto_reasons=[] if passed else ["verifier_failed"],
            )
        )

    best = select_best(results)
    return {
        "best_name": best.name,
        "best_text": best.text,
        "best_scores": best.scores,
        "best_latency_s": best.latency_s,
        "candidates": [
            {
                "name": c.name,
                "scores": c.scores,
                "latency_s": c.latency_s,
            }
            for c in results
        ],
    }

