import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from adversarial_protocols import (
    build_contradiction_brief,
    detect_collusion,
    rotate_challengers,
    select_challengers,
    should_trigger_reaudit,
)
from quality_constitution import QualityConstitution
from proxy_bridge import ProxyBridge
from verifiers import (
    Verdict,
    verify_logic,
    verify_factual,
    verify_safety,
    verify_reproducibility,
)


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    INVENTOR = "inventor"
    IMPLEMENTER = "implementer"
    CHALLENGER = "challenger"
    JUDGE = "judge"
    COMPOSER = "composer"


class VetoReason(str, Enum):
    LOGIC = "logic_veto"
    FACTUAL = "factual_veto"
    SAFETY = "safety_veto"
    EVIDENCE = "evidence_veto"
    UNCERTAINTY = "uncertainty_veto"
    SOFT_SCORE = "soft_score_veto"
    COLLUSION = "collusion_veto"
    TIMEOUT = "timeout_veto"


@dataclass
class MotivationVector:
    speed: float
    novelty: float
    rigor: float
    safety: float
    reproducibility: float


@dataclass
class EvidencePacket:
    agent_name: str
    role: AgentRole
    proposal_text: str
    rationale: str
    claims: List[str]
    evidence_refs: List[str]
    uncertainty: float
    test_artifacts: List[str] = field(default_factory=list)
    latency_s: float = 0.0
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    response: str
    passed: bool
    veto_reasons: List[str]
    rounds_used: int
    metrics: Dict[str, float]
    trace: Dict


class AdversarialAgentCluster:
    """Single-machine adversarial cluster with strict quality veto semantics."""

    def __init__(
        self,
        engine,
        orchestrator,
        max_repair_loops: int = 2,
        round_timeout_s: float = 45.0,
        samples_per_role: int = 2,
        challenger_samples: int = 2,
        consensus_entropy_threshold: float = 0.55,
        composer_samples: int = 2,
    ):
        self.engine = engine
        self.orchestrator = orchestrator
        self.max_repair_loops = max_repair_loops
        self.round_timeout_s = round_timeout_s
        self.samples_per_role = max(1, int(samples_per_role))
        self.challenger_samples = max(1, int(challenger_samples))
        self.consensus_entropy_threshold = consensus_entropy_threshold
        self.composer_samples = max(1, int(composer_samples))
        self.constitution = QualityConstitution()
        self.proxy = ProxyBridge()
        self.performance_memory = {}

        self.motivations = {
            "ResearcherAgentA": MotivationVector(0.2, 0.5, 0.9, 0.5, 0.8),
            "InventorAgentB": MotivationVector(0.2, 1.0, 0.5, 0.3, 0.3),
            "ImplementerAgentC": MotivationVector(0.5, 0.3, 0.9, 0.6, 1.0),
            "ChallengerAgentX": MotivationVector(0.1, 0.2, 1.0, 0.8, 0.8),
            "ChallengerAgentY": MotivationVector(0.1, 0.2, 1.0, 0.8, 0.8),
        }

    def _motivation_for_round(self, agent_name: str) -> MotivationVector:
        base = self.motivations[agent_name]
        memory = self.performance_memory.get(agent_name, {"wins": 0.0, "fails": 0.0})
        total = memory["wins"] + memory["fails"]
        if total <= 0:
            return base
        win_rate = memory["wins"] / total
        # Evolutionary tilt: low win-rate => increase rigor/reproducibility, reduce novelty.
        novelty = max(0.1, min(1.0, base.novelty - 0.2 * (0.5 - win_rate)))
        rigor = max(0.1, min(1.0, base.rigor + 0.2 * (0.5 - win_rate)))
        repro = max(0.1, min(1.0, base.reproducibility + 0.1 * (0.5 - win_rate)))
        return MotivationVector(
            speed=base.speed,
            novelty=novelty,
            rigor=rigor,
            safety=base.safety,
            reproducibility=repro,
        )

    def _allocate_roles(self, query: str):
        """Use existing router stack as one allocator signal."""
        top1_weights, top1_reason = self.orchestrator.route(query, top_k=1)
        top2_weights, top2_domains, top2_reason = self.orchestrator.route_top2(query)
        role_signal = {
            "top1_weights": top1_weights,
            "top1_reason": top1_reason,
            "top2_weights": top2_weights,
            "top2_domains": top2_domains,
            "top2_reason": top2_reason,
        }
        return top2_weights, role_signal

    def _build_prompt(self, query: str, role: AgentRole, motivation: MotivationVector, repair_context: str) -> str:
        system = (
            f"You are {role.value}. "
            f"Optimize for novelty={motivation.novelty:.2f}, rigor={motivation.rigor:.2f}, "
            f"safety={motivation.safety:.2f}, reproducibility={motivation.reproducibility:.2f}, speed={motivation.speed:.2f}. "
            "Return concise answer with explicit claims and short evidence notes."
        )
        repair = f"\nRepair context:\n{repair_context}\n" if repair_context else ""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{query}{repair}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _generate_packet(
        self,
        agent_name: str,
        role: AgentRole,
        query: str,
        routing_weights: Dict[str, float],
        repair_context: str = "",
    ) -> EvidencePacket:
        motivation = self.motivations[agent_name]
        motivation = self._motivation_for_round(agent_name)
        prompt = self._build_prompt(query, role, motivation, repair_context)
        start = time.time()
        text, _ = self.engine.generate(prompt, routing_weights=routing_weights, max_tokens=180)
        latency = time.time() - start

        claims = [c.strip() for c in text.split(".") if c.strip()][:5]
        refs = [f"internal:{agent_name}:claim_{i+1}" for i in range(len(claims))]
        artifacts = [f"artifact:{agent_name}:checklist_v1", f"artifact:{agent_name}:consistency_scan"]
        uncertainty = min(0.95, 1.0 - motivation.rigor * 0.5 - motivation.reproducibility * 0.3)
        rationale = claims[0] if claims else text[:120]
        return EvidencePacket(
            agent_name=agent_name,
            role=role,
            proposal_text=text,
            rationale=rationale,
            claims=claims,
            evidence_refs=refs,
            uncertainty=float(max(0.05, uncertainty)),
            test_artifacts=artifacts,
            latency_s=latency,
        )

    def _choose_best_packet(self, packets: List[EvidencePacket]) -> EvidencePacket:
        # Local quality score used for producer self-competition.
        ranked = sorted(
            packets,
            key=lambda p: (
                len(p.evidence_refs),
                len(p.test_artifacts),
                len(p.claims),
                -p.uncertainty,
                -len(p.contradictions),
            ),
            reverse=True,
        )
        return ranked[0]

    def _generate_role_packet(
        self,
        agent_name: str,
        role: AgentRole,
        query: str,
        routing_weights: Dict[str, float],
        repair_context: str,
    ) -> EvidencePacket:
        candidates: List[EvidencePacket] = []
        for _ in range(self.samples_per_role):
            candidates.append(
                self._generate_packet(
                    agent_name=agent_name,
                    role=role,
                    query=query,
                    routing_weights=routing_weights,
                    repair_context=repair_context,
                )
            )
        return self._choose_best_packet(candidates)

    def _challenge_packets(self, query: str, packets: List[EvidencePacket], routing_weights: Dict[str, float], challenger_names: List[str]):
        brief = build_contradiction_brief(query, packets)
        challenge_outputs = {}
        contradiction_votes = {p.agent_name: 0 for p in packets}
        for challenger in challenger_names:
            motivation = self.motivations.get(challenger, MotivationVector(0.2, 0.2, 1.0, 0.9, 0.8))
            challenger_outputs = []
            for _ in range(self.challenger_samples):
                prompt = self._build_prompt(brief, AgentRole.CHALLENGER, motivation, repair_context="")
                text, _ = self.engine.generate(prompt, routing_weights=routing_weights, max_tokens=140)
                challenger_outputs.append(text)
                lower = (text or "").lower()
                for p in packets:
                    if p.agent_name.lower() in lower or "contradiction" in lower or "unsupported" in lower:
                        p.contradictions.append(f"{challenger}:challenge_flag")
                        contradiction_votes[p.agent_name] += 1
            challenge_outputs[challenger] = challenger_outputs

        # Normalized entropy of contradiction vote distribution (higher => stronger disagreement).
        total_votes = sum(contradiction_votes.values())
        if total_votes <= 0:
            consensus_entropy = 0.0
        else:
            probs = [v / total_votes for v in contradiction_votes.values() if v > 0]
            ent = -sum(pv * math.log(pv + 1e-12) for pv in probs)
            max_ent = math.log(max(1, len(contradiction_votes)))
            consensus_entropy = float(ent / max_ent) if max_ent > 0 else 0.0
        return challenge_outputs, contradiction_votes, consensus_entropy

    def _score_and_judge(self, query: str, packets: List[EvidencePacket]) -> Tuple[List[Verdict], Dict[str, float]]:
        logic = verify_logic(query, packets)
        factual = verify_factual(query, packets)
        safety = verify_safety(query, packets)
        repro = verify_reproducibility(query, packets)
        verdicts = [logic, factual, safety, repro]
        agg = {
            "logic_score": logic.score,
            "factual_score": factual.score,
            "safety_score": safety.score,
            "repro_score": repro.score,
            "mean_score": sum(v.score for v in verdicts) / len(verdicts),
        }
        return verdicts, agg

    def _proxy_recovery_packet(self, query: str, veto_reasons: List[str], routing_weights: Dict[str, float]) -> EvidencePacket:
        proxy_query = (
            "Provide a strict, evidence-oriented answer to this query with concise steps.\n"
            f"Query: {query}\n"
            f"Known veto reasons to avoid: {', '.join(veto_reasons) if veto_reasons else 'none'}"
        )
        answer = self.proxy.ask(proxy_query, mode="reasoning")
        if not answer:
            answer = "Proxy unavailable or returned empty result."
        claims = [c.strip() for c in answer.split(".") if c.strip()][:5]
        return EvidencePacket(
            agent_name="ProxyRecoveryAgent",
            role=AgentRole.RESEARCHER,
            proposal_text=answer,
            rationale=claims[0] if claims else answer[:120],
            claims=claims,
            evidence_refs=["external:perplexity_proxy"],
            test_artifacts=["artifact:proxy_recovery"],
            uncertainty=0.35 if "unavailable" not in answer.lower() else 0.85,
            latency_s=0.0,
        )

    def _compose_final_candidates(
        self,
        query: str,
        packets: List[EvidencePacket],
        routing_weights: Dict[str, float],
        repair_context: str,
    ) -> List[EvidencePacket]:
        """
        Generate multiple end-to-end final answers that compete.
        Each candidate is treated as an EvidencePacket and re-verified.
        """
        # Build a compact brief using top claims.
        bullets = []
        for p in packets:
            top_claims = p.claims[:3] if p.claims else [p.proposal_text[:120]]
            bullets.append(f"- {p.agent_name}({p.role.value}): " + " | ".join(top_claims))
        brief = "\n".join(bullets)
        composer_query = (
            "Synthesize ONE best final answer.\n"
            "Requirements:\n"
            "- Must be logically consistent and directly answer the query.\n"
            "- Must include reproducible steps/tests where applicable.\n"
            "- Must avoid unsupported claims.\n\n"
            f"User query:\n{query}\n\n"
            f"Candidate evidence:\n{brief}\n"
        )
        candidates: List[EvidencePacket] = []
        for i in range(self.composer_samples):
            motivation = MotivationVector(speed=0.3, novelty=0.4, rigor=1.0, safety=0.8, reproducibility=1.0)
            prompt = self._build_prompt(composer_query, AgentRole.COMPOSER, motivation, repair_context=repair_context)
            start = time.time()
            text, _ = self.engine.generate(prompt, routing_weights=routing_weights, max_tokens=220)
            latency = time.time() - start
            claims = [c.strip() for c in text.split(".") if c.strip()][:6]
            candidates.append(
                EvidencePacket(
                    agent_name=f"FinalComposer_{i+1}",
                    role=AgentRole.COMPOSER,
                    proposal_text=text,
                    rationale=claims[0] if claims else text[:120],
                    claims=claims,
                    evidence_refs=["internal:composer:brief"],
                    test_artifacts=["artifact:composer:consistency", "artifact:composer:checklist"],
                    uncertainty=0.25,
                    latency_s=latency,
                )
            )
        return candidates

    def _tournament_select(
        self, query: str, candidates: List[EvidencePacket]
    ) -> Tuple[EvidencePacket, Dict]:
        """
        Re-verify each composed candidate and select best by mean_score, then contradictions/uncertainty.
        """
        scored = []
        for c in candidates:
            verdicts, scores = self._score_and_judge(query, [c])
            decision = self.constitution.evaluate(verdicts, [c])
            scored.append(
                {
                    "candidate": c,
                    "verdicts": [v.__dict__ for v in verdicts],
                    "scores": scores,
                    "decision": decision,
                }
            )
        # Prefer candidates that pass, then highest mean score.
        ranked = sorted(
            scored,
            key=lambda x: (
                1 if x["decision"]["passed"] else 0,
                x["scores"].get("mean_score", 0.0),
            ),
            reverse=True,
        )
        winner = ranked[0]["candidate"]
        meta = {"tournament": ranked}
        return winner, meta

    def run(self, query: str) -> ClusterResult:
        total_start = time.time()
        trace = {"rounds": []}
        veto_reasons: List[str] = []
        final_response = ""

        routing_weights, role_signal = self._allocate_roles(query)
        routing_reasoning = role_signal.get("top2_reason", "")
        if isinstance(routing_reasoning, list):
            routing_reasoning = " | ".join([x for x in routing_reasoning if x])

        repair_context = ""
        for round_idx in range(1, self.max_repair_loops + 2):
            round_start = time.time()
            producers = [
                ("ResearcherAgentA", AgentRole.RESEARCHER),
                ("InventorAgentB", AgentRole.INVENTOR),
                ("ImplementerAgentC", AgentRole.IMPLEMENTER),
            ]
            packets = [
                self._generate_role_packet(name, role, query, routing_weights, repair_context)
                for name, role in producers
            ]

            challenger_names = rotate_challengers(
                select_challengers([p.agent_name for p in packets]), round_idx
            )
            challenge_outputs, contradiction_votes, consensus_entropy = self._challenge_packets(
                query, packets, routing_weights, challenger_names
            )
            collusion_flags = detect_collusion(packets)
            for i, flag in enumerate(collusion_flags):
                if flag and i < len(packets):
                    packets[i].contradictions.append("potential_collusion_signature")
            high_disagreement = consensus_entropy >= self.consensus_entropy_threshold
            re_audit = should_trigger_reaudit(query, collusion_flags, round_idx) or high_disagreement
            if re_audit:
                # Trigger an extra challenger pass with rotated ordering.
                second_pass = rotate_challengers(challenger_names, round_idx + 1)
                re_outputs, re_votes, re_entropy = self._challenge_packets(query, packets, routing_weights, second_pass)
                challenge_outputs["reaudit"] = re_outputs
                contradiction_votes["reaudit_total"] = sum(re_votes.values())
                consensus_entropy = max(consensus_entropy, re_entropy)

            verdicts, scores = self._score_and_judge(query, packets)
            decision = self.constitution.evaluate(verdicts, packets)
            if any(collusion_flags):
                decision["veto_reasons"].append(f"{VetoReason.COLLUSION.value}:similarity_signature_detected")
                decision["passed"] = False

            if (time.time() - round_start) > self.round_timeout_s:
                decision["veto_reasons"].append(f"{VetoReason.TIMEOUT.value}:round_timeout_exceeded")
                decision["passed"] = False

            proxy_used = False
            allow_proxy = all(
                ("safety_veto" not in r and "unsafe" not in r) for r in decision.get("veto_reasons", [])
            )
            if (not decision["passed"]) and self.proxy.enabled and allow_proxy:
                proxy_packet = self._proxy_recovery_packet(query, decision["veto_reasons"], routing_weights)
                packets_with_proxy = packets + [proxy_packet]
                p_verdicts, p_scores = self._score_and_judge(query, packets_with_proxy)
                p_decision = self.constitution.evaluate(p_verdicts, packets_with_proxy)
                if p_decision["passed"]:
                    packets = packets_with_proxy
                    verdicts = p_verdicts
                    scores = p_scores
                    decision = p_decision
                    proxy_used = True
            round_trace = {
                "round": round_idx,
                "routing_weights": routing_weights,
                "role_signal": role_signal,
                "routing_reasoning": routing_reasoning,
                "challengers": challenger_names,
                "challenge_outputs": challenge_outputs,
                "contradiction_votes": contradiction_votes,
                "consensus_entropy": consensus_entropy,
                "high_disagreement": high_disagreement,
                "reaudit_triggered": re_audit,
                "proxy_used": proxy_used,
                "proxy_allowed": allow_proxy,
                "packets": [
                    {
                        "agent": p.agent_name,
                        "role": p.role.value,
                        "uncertainty": p.uncertainty,
                        "evidence_refs": p.evidence_refs,
                        "test_artifacts": p.test_artifacts,
                        "claims_count": len(p.claims),
                        "contradictions": p.contradictions,
                    }
                    for p in packets
                ],
                "verdicts": [v.__dict__ for v in verdicts],
                "scores": scores,
                "decision": decision,
            }
            trace["rounds"].append(round_trace)

            if decision["passed"]:
                # End-to-end answer tournament (composer competition).
                composer_candidates = self._compose_final_candidates(
                    query=query,
                    packets=packets,
                    routing_weights=routing_weights,
                    repair_context=repair_context,
                )
                winner, tmeta = self._tournament_select(query, composer_candidates)
                trace["rounds"][-1]["composer_candidates"] = [
                    {"agent": c.agent_name, "latency_s": c.latency_s, "claims_count": len(c.claims)}
                    for c in composer_candidates
                ]
                trace["rounds"][-1]["tournament"] = tmeta["tournament"]
                for p in packets:
                    if p.agent_name not in self.performance_memory:
                        self.performance_memory[p.agent_name] = {"wins": 0.0, "fails": 0.0}
                    self.performance_memory[p.agent_name]["wins"] += 1.0
                final_response = winner.proposal_text
                elapsed = time.time() - total_start
                metrics = {
                    "latency_s": round(elapsed, 3),
                    "rounds_used": round_idx,
                    "veto_count": 0.0,
                    "evidence_completeness": 1.0 if len(winner.evidence_refs) > 0 else 0.0,
                    "composer_samples": float(self.composer_samples),
                }
                return ClusterResult(
                    response=final_response,
                    passed=True,
                    veto_reasons=[],
                    rounds_used=round_idx,
                    metrics=metrics,
                    trace=trace,
                )

            veto_reasons = decision["veto_reasons"]
            repair_context = "Repair the output to resolve these veto reasons: " + "; ".join(veto_reasons)

        # Update evolutionary memory for next runs.
        for p in trace["rounds"][-1]["packets"] if trace["rounds"] else []:
            name = p["agent"]
            if name not in self.performance_memory:
                self.performance_memory[name] = {"wins": 0.0, "fails": 0.0}
            self.performance_memory[name]["fails"] += 1.0

        elapsed = time.time() - total_start
        metrics = {
            "latency_s": round(elapsed, 3),
            "rounds_used": self.max_repair_loops + 1,
            "veto_count": float(len(veto_reasons)),
            "evidence_completeness": 0.0,
        }
        return ClusterResult(
            response="Unable to satisfy strict quality constitution. Returning fail-closed result.",
            passed=False,
            veto_reasons=veto_reasons,
            rounds_used=self.max_repair_loops + 1,
            metrics=metrics,
            trace=trace,
        )
