from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


ROUTER_SYSTEM_PROMPT = (
    "You are the TCAR routing model. Analyze the user's request, plan the required reasoning steps, "
    "and output the exact expert tags needed to solve the task.\n"
    "Return exactly this format:\n"
    "<thinking>\n"
    "- short bullet\n"
    "- short bullet\n"
    "</thinking>\n"
    "<experts>[DOMAIN_A],[DOMAIN_B]</experts>"
)


@dataclass
class RouterDecisionLite:
    thinking: str
    experts: list[str]
    raw_text: str
    latency_s: float


class HFTrainedRouter:
    def __init__(
        self,
        registry_path: str | Path,
        model_name_or_path: str,
        adapter_path: str | Path,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with open(registry_path, "r") as f:
            self.registry = json.load(f)
        self.domains = list(self.registry.keys())
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        resolved_model = self._resolve_model_name_or_path(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "mps" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, query: str) -> str:
        return (
            f"<|im_start|>system\n{ROUTER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _generate_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        do_sample: bool,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        start = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency_s = time.time() - start
        completion = output[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=False).strip(), latency_s

    def route(self, query: str, max_experts: int = 2, max_tokens: int = 120):
        prompt = self.build_prompt(query)
        raw_text, latency_s = self._generate_completion(
            prompt,
            max_tokens=max_tokens,
            do_sample=False,
        )
        experts = self._extract_experts(raw_text, max_experts=max_experts)
        thinking = self.extract_thinking(raw_text)
        return RouterDecisionLite(
            thinking=thinking,
            experts=experts,
            raw_text=raw_text,
            latency_s=latency_s,
        )

    def sample_routes(
        self,
        query: str,
        *,
        num_samples: int = 4,
        max_experts: int = 2,
        max_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list:
        samples = []
        prompt = self.build_prompt(query)
        for _ in range(max(1, int(num_samples))):
            raw_text, latency_s = self._generate_completion(
                prompt,
                max_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            experts = self._extract_experts(raw_text, max_experts=max_experts)
            thinking = self.extract_thinking(raw_text)
            samples.append(
                RouterDecisionLite(
                    thinking=thinking,
                    experts=experts,
                    raw_text=raw_text,
                    latency_s=latency_s,
                )
            )
        return samples

    def sample_unique_routes(
        self,
        query: str,
        *,
        num_samples: int = 4,
        max_experts: int = 2,
        max_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_attempts: int = 16,
    ) -> list[RouterDecisionLite]:
        unique: list[RouterDecisionLite] = []
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(unique) < max(1, int(num_samples)) and attempts < max_attempts:
            attempts += 1
            decision = self.sample_routes(
                query,
                num_samples=1,
                max_experts=max_experts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )[0]
            key = tuple(decision.experts)
            if key in seen:
                continue
            seen.add(key)
            unique.append(decision)
        if not unique:
            unique.append(self.route(query, max_experts=max_experts, max_tokens=max_tokens))
        return unique

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
        return deduped[: max(1, int(max_experts))]

    @staticmethod
    def extract_thinking(text: str) -> str:
        match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _resolve_model_name_or_path(model_name_or_path: str) -> str:
        path = Path(model_name_or_path).expanduser()
        if path.exists():
            return str(path)
        if "/" not in model_name_or_path:
            return model_name_or_path
        org, name = model_name_or_path.split("/", 1)
        hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"
        snapshots_dir = hub_dir / "snapshots"
        if not snapshots_dir.exists():
            return model_name_or_path
        snapshots = sorted(snapshots_dir.iterdir())
        if not snapshots:
            return model_name_or_path
        return str(snapshots[-1])
