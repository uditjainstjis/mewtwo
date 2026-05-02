from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dynamic_mlx_inference import DynamicEngine
from hf_trained_router import HFTrainedRouter


@dataclass
class RouterDecision:
    thinking: str
    experts: list[str]
    raw_text: str
    latency_s: float


@dataclass
class ExpertBranch:
    expert: str
    answer: str
    latency_s: float
    mode: str
    verifier_score: float | None = None


@dataclass
class CollaborativeReasoningResult:
    query: str
    router: RouterDecision
    branches: list[ExpertBranch]
    final_answer: str
    router_latency_s: float
    branch_latency_s: float
    verifier_latency_s: float
    total_latency_s: float
    parallel_workers: int
    selected_expert: str | None = None
    route_candidates: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "router": asdict(self.router),
            "branches": [asdict(branch) for branch in self.branches],
            "final_answer": self.final_answer,
            "router_latency_s": self.router_latency_s,
            "branch_latency_s": self.branch_latency_s,
            "verifier_latency_s": self.verifier_latency_s,
            "total_latency_s": self.total_latency_s,
            "parallel_workers": self.parallel_workers,
            "selected_expert": self.selected_expert,
            "route_candidates": self.route_candidates or [],
        }

def _trim_to_word_budget(text: str, max_words: int = 50) -> str:
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


def _build_expert_prompt(query: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a domain expert inside a collaborative reasoning system. "
        "Answer the full user question as directly as you can using your active domain expertise. "
        "If another domain is required, state one uncertainty instead of guessing. "
        "Answer in under 50 words. No XML, no bullet list, no preamble, no extra explanation.<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class NaturalLanguageReasoningRouter:
    def __init__(self, registry_path: str | Path, base_engine: DynamicEngine):
        self.base_engine = base_engine
        with open(registry_path, "r") as f:
            self.registry = json.load(f)
        self.domains = list(self.registry.keys())
        self.domain_list_str = ", ".join(f"[{domain}]" for domain in self.domains)

    def route(self, query: str, max_experts: int = 2, max_tokens: int = 120) -> RouterDecision:
        max_experts = max(1, int(max_experts))
        prompt = (
            "<|im_start|>system\n"
            "You are the TCAR reasoning router for a collaborative expert system.\n"
            "Analyze the user's request, identify the deductive subproblems, then select the smallest set "
            f"of experts needed to solve it. Choose between 1 and {max_experts} experts.\n"
            f"Available experts: {self.domain_list_str}\n"
            "Return exactly this format and nothing else:\n"
            "<thinking>\n"
            "- at most 8 words\n"
            "- at most 8 words\n"
            "</thinking>\n"
            "The thinking block must be 2 bullets maximum and under 20 total words.\n"
            "<experts>[DOMAIN_A],[DOMAIN_B]</experts><|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        raw_text, latency_s = self.base_engine.generate(prompt, routing_weights=None, max_tokens=max_tokens)
        experts = self._extract_experts(raw_text, max_experts=max_experts)
        thinking = self.extract_thinking(raw_text)
        return RouterDecision(thinking=thinking, experts=experts, raw_text=raw_text, latency_s=latency_s)

    def _extract_experts(self, text: str, max_experts: int) -> list[str]:
        seen: list[str] = []
        upper = text.upper()
        for domain in self.domains:
            if f"[{domain}]" in upper or re.search(rf"\b{re.escape(domain)}\b", upper):
                seen.append(domain)
        deduped: list[str] = []
        for domain in seen:
            if domain not in deduped:
                deduped.append(domain)
        if not deduped:
            return [self.domains[0]]
        return deduped[:max_experts]

    @staticmethod
    def extract_thinking(text: str) -> str:
        match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


class CollaborativeReasoner:
    def __init__(
        self,
        base_engine: DynamicEngine,
        registry_path: str | Path,
        *,
        model_path: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        parallel_workers: int = 1,
        backend_dir: str | Path | None = None,
        router_model_name_or_path: str | None = None,
        router_adapter_path: str | Path | None = None,
        router_num_samples: int = 4,
        router_temperature: float = 0.7,
        router_top_p: float = 0.9,
    ):
        self.base_engine = base_engine
        self.registry_path = Path(registry_path)
        self.model_path = model_path
        # Single-engine MLX execution only. The value is preserved in results for
        # compatibility, but execution intentionally stays single-process.
        self.parallel_workers = 1
        self.backend_dir = Path(backend_dir) if backend_dir is not None else Path.cwd()
        self.router_num_samples = max(1, int(router_num_samples))
        self.router_temperature = float(router_temperature)
        self.router_top_p = float(router_top_p)
        if router_adapter_path:
            self.router = HFTrainedRouter(
                self.registry_path,
                router_model_name_or_path or "Qwen/Qwen2.5-0.5B-Instruct",
                router_adapter_path,
            )
        else:
            self.router = NaturalLanguageReasoningRouter(self.registry_path, self.base_engine)
        with open(self.registry_path, "r") as f:
            self.registry = json.load(f)

    def shutdown(self) -> None:
        return None

    def run(
        self,
        query: str,
        *,
        max_experts: int = 2,
        router_max_tokens: int = 120,
        expert_max_tokens: int = 140,
        refine_max_tokens: int = 220,
    ) -> CollaborativeReasoningResult:
        total_start = time.time()
        if hasattr(self.router, "sample_unique_routes") and self.router_num_samples > 1:
            sampled = self.router.sample_unique_routes(
                query,
                num_samples=self.router_num_samples,
                max_experts=max_experts,
                max_tokens=router_max_tokens,
                temperature=self.router_temperature,
                top_p=self.router_top_p,
            )
            result = self.run_with_router_candidates(
                query,
                [
                    RouterDecision(
                        thinking=s.thinking,
                        experts=list(s.experts),
                        raw_text=s.raw_text,
                        latency_s=float(s.latency_s),
                    )
                    for s in sampled
                ],
                expert_max_tokens=expert_max_tokens,
                refine_max_tokens=refine_max_tokens,
            )
            result.total_latency_s = time.time() - total_start
            return result
        router_decision = self.router.route(query, max_experts=max_experts, max_tokens=router_max_tokens)
        result = self.run_with_router_decision(
            query,
            router_decision,
            expert_max_tokens=expert_max_tokens,
            refine_max_tokens=refine_max_tokens,
        )
        result.total_latency_s = time.time() - total_start
        return result

    def run_with_experts(
        self,
        query: str,
        experts: list[str],
        *,
        expert_max_tokens: int = 140,
        refine_max_tokens: int = 220,
        router_thinking: str = "oracle experts supplied by benchmark metadata",
    ) -> CollaborativeReasoningResult:
        router_decision = RouterDecision(
            thinking=router_thinking,
            experts=list(experts),
            raw_text="",
            latency_s=0.0,
        )
        return self.run_with_router_decision(
            query,
            router_decision,
            expert_max_tokens=expert_max_tokens,
            refine_max_tokens=refine_max_tokens,
        )

    def run_with_router_candidates(
        self,
        query: str,
        router_decisions: list[RouterDecision],
        *,
        expert_max_tokens: int = 140,
        refine_max_tokens: int = 220,
    ) -> CollaborativeReasoningResult:
        if not router_decisions:
            raise ValueError("Expected at least one router candidate.")
        total_router_latency = sum(float(d.latency_s) for d in router_decisions)
        total_branch_latency = 0.0
        total_verifier_latency = 0.0
        best_result: CollaborativeReasoningResult | None = None
        best_score = float("-inf")
        candidate_rows: list[dict[str, Any]] = []
        for decision in router_decisions:
            candidate = self.run_with_router_decision(
                query,
                decision,
                expert_max_tokens=expert_max_tokens,
                refine_max_tokens=refine_max_tokens,
            )
            total_branch_latency += candidate.branch_latency_s
            total_verifier_latency += candidate.verifier_latency_s
            chosen_score = float("-inf")
            for branch in candidate.branches:
                if branch.expert == candidate.selected_expert and branch.verifier_score is not None:
                    chosen_score = float(branch.verifier_score)
                    break
            candidate_rows.append(
                {
                    "experts": list(decision.experts),
                    "thinking": decision.thinking,
                    "latency_s": round(float(decision.latency_s), 3),
                    "selected_expert": candidate.selected_expert,
                    "selected_score": None if chosen_score == float("-inf") else round(chosen_score, 4),
                }
            )
            if chosen_score > best_score:
                best_score = chosen_score
                best_result = candidate
        assert best_result is not None
        best_result.router_latency_s = total_router_latency
        best_result.branch_latency_s = total_branch_latency
        best_result.verifier_latency_s = total_verifier_latency
        best_result.total_latency_s = total_router_latency + total_branch_latency + total_verifier_latency
        best_result.route_candidates = candidate_rows
        return best_result

    def run_with_router_decision(
        self,
        query: str,
        router_decision: RouterDecision,
        *,
        expert_max_tokens: int = 140,
        refine_max_tokens: int = 220,
    ) -> CollaborativeReasoningResult:
        branches, branch_latency_s = self._run_branches(query, router_decision.experts, expert_max_tokens)
        final_answer, verifier_latency_s, selected_expert = self._select_best_branch(query, branches)
        return CollaborativeReasoningResult(
            query=query,
            router=router_decision,
            branches=branches,
            final_answer=final_answer,
            router_latency_s=router_decision.latency_s,
            branch_latency_s=branch_latency_s,
            verifier_latency_s=verifier_latency_s,
            total_latency_s=router_decision.latency_s + branch_latency_s + verifier_latency_s,
            parallel_workers=self.parallel_workers,
            selected_expert=selected_expert,
        )

    def _run_branches(self, query: str, experts: list[str], expert_max_tokens: int) -> tuple[list[ExpertBranch], float]:
        if not experts:
            return [], 0.0
        start = time.time()
        shared_prompt = _build_expert_prompt(query)
        prompt_cache, decode_prompt_tokens = self.base_engine.prepare_prompt_cache(shared_prompt)
        branch_rows = [
            self._run_branch_from_shared_prefill(
                expert,
                prompt_cache=prompt_cache,
                decode_prompt_tokens=decode_prompt_tokens,
                expert_max_tokens=expert_max_tokens,
            )
            for expert in experts
        ]
        branches = [
            ExpertBranch(
                expert=row["expert"],
                answer=row["answer"],
                latency_s=float(row["latency_s"]),
                mode=row["mode"],
            )
            for row in branch_rows
        ]
        return branches, time.time() - start

    def _run_branch_from_shared_prefill(self, expert: str, *, prompt_cache, decode_prompt_tokens, expert_max_tokens: int) -> dict[str, Any]:
        weights = {domain: 0.0 for domain in self.registry}
        if expert in weights:
            weights[expert] = 1.0
        self.base_engine.set_adapter_layer_gate(0, -1)
        answer, latency_s = self.base_engine.generate_from_prompt_cache(
            prompt_cache=prompt_cache,
            decode_prompt_tokens=decode_prompt_tokens,
            routing_weights=weights,
            max_tokens=expert_max_tokens,
        )
        answer = _trim_to_word_budget(answer, max_words=50)
        return {
            "expert": expert,
            "answer": answer,
            "latency_s": latency_s,
            "mode": "shared_prefill_cache",
        }

    def _select_best_branch(self, query: str, branches: list[ExpertBranch]) -> tuple[str, float, str | None]:
        if not branches:
            return "", 0.0, None
        start = time.time()
        verifier_prompt = self._build_verifier_prompt(query)
        best_branch: ExpertBranch | None = None
        best_confidence = float("-inf")
        for branch in branches:
            score = self.base_engine.score_completion(verifier_prompt, branch.answer, routing_weights={})
            branch.verifier_score = float(score["confidence"])
            if branch.verifier_score > best_confidence:
                best_confidence = branch.verifier_score
                best_branch = branch
        elapsed = time.time() - start
        if best_branch is None:
            return "", elapsed, None
        return best_branch.answer, elapsed, best_branch.expert

    @staticmethod
    def _build_verifier_prompt(query: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a strict verifier. The best answer is directly responsive, factually grounded, and concise. "
            "Answer the user's question in under 50 words with no preamble.<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
