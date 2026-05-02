#!/usr/bin/env python3
"""
Gate-Conditioned LoRI-MoE Inference Engine

The novel inference path:
  1. Base Nemotron processes input tokens  
  2. Internal MoE routing signals are captured via hooks
  3. GC-LoRI Router reads signals + hidden states
  4. Per-token adapter composition weights are computed
  5. Selected adapter is loaded and used for generation

This module implements THREE inference modes for ablation:
  A. gc_lori:    Gate-conditioned routing (THE INNOVATION)
  B. blind:      Standard external router (CONTROL)
  C. oracle:     Ground-truth domain selection (CEILING)

Usage:
    python -m src.lori_moe.inference.gc_compose \
        --model_path ./models/nemotron \
        --adapter_dir ./adapters/nemotron_30b/adapters \
        --gc_router_path ./adapters/nemotron_30b/gc_router/best/gc_router.pt \
        --prompt "Solve step by step: What is the integral of x^2 * e^x?"
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

from src.lori_moe.model.gc_router import GateConditionedRouter
from src.lori_moe.model.internal_hook import NemotronRouterHook


@dataclass
class GCLoRIResult:
    """Result from a single GC-LoRI inference."""
    response: str
    selected_domain: str
    routing_weights: Dict[str, float]
    internal_entropy_mean: Optional[float] = None
    internal_entropy_std: Optional[float] = None
    latency_ms: float = 0.0
    mode: str = "gc_lori"  # gc_lori, blind, oracle
    num_input_tokens: int = 0
    num_output_tokens: int = 0


class GCLoRIComposer:
    """
    Gate-Conditioned LoRI-MoE composition engine.
    
    Supports three modes:
    - gc_lori: Routes using internal MoE signals + hidden states
    - blind: Routes using only hidden states (control baseline)
    - oracle: Uses ground-truth domain label (ceiling)
    """

    def __init__(
        self,
        model_path: str,
        adapter_dir: str,
        gc_router_path: Optional[str] = None,
        blind_router_path: Optional[str] = None,
        domains: List[str] = None,
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        self.device = device
        self.domains = domains or ["math", "code", "science"]
        self.adapter_dir = Path(adapter_dir)
        self.model_path = model_path
        self._current_adapter_domain = None

        # Load base model
        logger.info(f"Loading base model: {model_path}")
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info(f"Model loaded. VRAM: {vram:.1f} GB")

        # Install internal router hooks
        self.hooker = NemotronRouterHook(self.model)
        self.hooker.install()
        logger.info(f"Internal hooks: {self.hooker}")

        # Load GC-LoRI Router
        self.gc_router = None
        if gc_router_path and Path(gc_router_path).exists():
            checkpoint = torch.load(gc_router_path, map_location=device, weights_only=True)
            self.gc_router = GateConditionedRouter(**checkpoint["config"])
            self.gc_router.load_state_dict(checkpoint["state_dict"])
            self.gc_router.to(device).eval()
            logger.info(f"GC-LoRI Router loaded from {gc_router_path}")
        else:
            logger.warning("No GC-LoRI router loaded. Only 'blind' and 'oracle' modes available.")

        # Load blind router (control baseline)
        self.blind_router = None
        if blind_router_path and Path(blind_router_path).exists():
            checkpoint = torch.load(blind_router_path, map_location=device, weights_only=True)
            from src.lori_moe.model.router import TokenRouter
            self.blind_router = TokenRouter(
                hidden_dim=checkpoint["config"]["hidden_dim"],
                num_experts=len(self.domains),
                bottleneck_dim=checkpoint["config"].get("bottleneck_dim", 64),
            )
            self.blind_router.load_state_dict(checkpoint["state_dict"])
            self.blind_router.to(device).eval()
            logger.info(f"Blind router loaded from {blind_router_path}")

        # Discover available adapters
        self.available_adapters = {}
        for domain in self.domains:
            for subdir in ["dare_sparsified", "best", "final"]:
                path = self.adapter_dir / domain / subdir
                if path.exists() and (path / "adapter_config.json").exists():
                    self.available_adapters[domain] = path
                    break
        
        logger.info(f"Available adapters: {list(self.available_adapters.keys())}")
        logger.info(f"GC-LoRI Composer ready. Modes: {self.available_modes}")

    @property
    def available_modes(self) -> List[str]:
        """List available inference modes."""
        modes = ["oracle"]
        if self.blind_router is not None or True:  # blind fallback always available
            modes.append("blind")
        if self.gc_router is not None:
            modes.append("gc_lori")
        return modes

    def _prepare_input(self, prompt: str) -> dict:
        """Format prompt with chat template and tokenize."""
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_hidden_and_internal_signals(
        self, inputs: dict
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Run forward pass and extract hidden states + internal routing signals."""
        self.hooker.clear()

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # (B, S, D)
        internal_signal = self.hooker.get_aggregated_signal()

        return hidden, internal_signal

    def _route_gc_lori(
        self, hidden: torch.Tensor, internal_signal: Dict[str, torch.Tensor]
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """Route using GC-LoRI (the innovation)."""
        if self.gc_router is None:
            raise RuntimeError("GC-LoRI router not loaded")

        with torch.no_grad():
            adapter_weights, _ = self.gc_router(hidden, internal_signal, return_aux_loss=False)

        # Use last-token routing for autoregressive generation
        last_weights = adapter_weights[:, -1, :].squeeze(0)  # (num_experts,)
        domain_weights = {d: last_weights[i].item() for i, d in enumerate(self.domains)}
        selected = max(domain_weights, key=domain_weights.get)

        # Internal signal stats
        entropy_stats = {}
        if internal_signal:
            ent = internal_signal["entropy"]
            entropy_stats = {
                "mean": ent.mean().item(),
                "std": ent.std().item(),
            }

        return selected, domain_weights, entropy_stats

    def _route_blind(self, hidden: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """Route using only hidden states (control baseline)."""
        with torch.no_grad():
            # Average pool hidden states
            pooled = hidden.mean(dim=1)  # (B, D)

            if self.blind_router is not None:
                weights, _ = self.blind_router(hidden)
                last_weights = weights[:, -1, :].squeeze(0)
            else:
                # Fallback: simple norm-based heuristic
                # This shouldn't happen in practice — train a blind router first
                last_weights = torch.ones(len(self.domains), device=self.device)
                last_weights = last_weights / last_weights.sum()

        domain_weights = {d: last_weights[i].item() for i, d in enumerate(self.domains)}
        selected = max(domain_weights, key=domain_weights.get)
        return selected, domain_weights

    def _generate_with_adapter(
        self,
        inputs: dict,
        domain: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using a specific domain adapter."""
        adapter_path = self.available_adapters.get(domain)

        if adapter_path and self._current_adapter_domain != domain:
            # Unload current adapter if any
            if self._current_adapter_domain is not None:
                try:
                    self.model = self.model.base_model  # unwrap PeftModel
                except Exception:
                    pass
                torch.cuda.empty_cache()

            # Load new adapter
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), is_trainable=False
            )
            self.model.eval()
            self._current_adapter_domain = domain

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response, output_ids.shape[1] - inputs["input_ids"].shape[1]

    def generate(
        self,
        prompt: str,
        mode: str = "gc_lori",
        domain: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> GCLoRIResult:
        """
        Generate with GC-LoRI composition.

        Args:
            prompt: User prompt
            mode: 'gc_lori' (innovation), 'blind' (control), 'oracle' (ceiling)
            domain: Required for oracle mode — ground-truth domain
            max_new_tokens: Max tokens to generate
            temperature: 0 = greedy, >0 = sampling
        """
        t0 = time.time()

        inputs = self._prepare_input(prompt)
        num_input_tokens = inputs["input_ids"].shape[1]

        if mode == "oracle":
            if domain is None:
                raise ValueError("Oracle mode requires domain= argument")
            selected = domain
            domain_weights = {d: 1.0 if d == domain else 0.0 for d in self.domains}
            entropy_stats = {}

        elif mode == "gc_lori":
            hidden, internal_signal = self._get_hidden_and_internal_signals(inputs)
            if internal_signal is None:
                logger.warning("No internal signals captured. Falling back to blind routing.")
                selected, domain_weights = self._route_blind(hidden)
                entropy_stats = {}
                mode = "blind_fallback"
            else:
                selected, domain_weights, entropy_stats = self._route_gc_lori(
                    hidden, internal_signal
                )

        elif mode == "blind":
            hidden, _ = self._get_hidden_and_internal_signals(inputs)
            selected, domain_weights = self._route_blind(hidden)
            entropy_stats = {}

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'gc_lori', 'blind', or 'oracle'.")

        # Generate with selected adapter
        response, num_output_tokens = self._generate_with_adapter(
            inputs, selected, max_new_tokens, temperature
        )

        latency = (time.time() - t0) * 1000  # ms

        return GCLoRIResult(
            response=response,
            selected_domain=selected,
            routing_weights=domain_weights,
            internal_entropy_mean=entropy_stats.get("mean"),
            internal_entropy_std=entropy_stats.get("std"),
            latency_ms=latency,
            mode=mode,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
        )

    def run_ablation(
        self,
        prompts: List[Dict],
        modes: List[str] = None,
        max_new_tokens: int = 256,
    ) -> Dict[str, List[GCLoRIResult]]:
        """
        Run full ablation across modes on a prompt set.

        Args:
            prompts: List of {"text": ..., "domain": ..., "expected_answer": ...}
            modes: Which modes to test (default: all available)
        """
        modes = modes or self.available_modes
        results = {mode: [] for mode in modes}

        for i, prompt_info in enumerate(prompts):
            text = prompt_info["text"]
            domain = prompt_info.get("domain")

            for mode in modes:
                try:
                    if mode == "oracle" and domain is None:
                        continue

                    result = self.generate(
                        text,
                        mode=mode,
                        domain=domain,
                        max_new_tokens=max_new_tokens,
                    )
                    results[mode].append(result)

                    logger.info(
                        f"  [{i+1}/{len(prompts)}] {mode:10s} → {result.selected_domain} "
                        f"({result.latency_ms:.0f}ms)"
                    )
                except Exception as e:
                    logger.error(f"  [{i+1}/{len(prompts)}] {mode} FAILED: {e}")

        return results

    def cleanup(self):
        """Release resources."""
        self.hooker.remove()
        if self._current_adapter_domain is not None:
            try:
                self.model = self.model.base_model
            except Exception:
                pass
        del self.model
        torch.cuda.empty_cache()
        logger.info("GC-LoRI Composer cleaned up")


def main():
    parser = argparse.ArgumentParser(description="GC-LoRI Inference")
    parser.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "models" / "nemotron"))
    parser.add_argument("--adapter_dir", type=str, default=str(PROJECT_ROOT / "adapters" / "nemotron_30b"))
    parser.add_argument("--gc_router_path", type=str, default=str(PROJECT_ROOT / "adapters" / "nemotron_30b" / "gc_router" / "best" / "gc_router.pt"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="gc_lori", choices=["gc_lori", "blind", "oracle"])
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    composer = GCLoRIComposer(
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        gc_router_path=args.gc_router_path,
    )

    if args.prompt:
        result = composer.generate(
            args.prompt, mode=args.mode, domain=args.domain,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\n{'='*60}")
        print(f"Mode:    {result.mode}")
        print(f"Domain:  {result.selected_domain}")
        print(f"Weights: {result.routing_weights}")
        print(f"Entropy: mean={result.internal_entropy_mean}, std={result.internal_entropy_std}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"{'='*60}")
        print(f"\n{result.response}\n")

    elif args.interactive:
        print("\n🧬 GC-LoRI Interactive Mode")
        print(f"Available modes: {composer.available_modes}")
        print(f"Available adapters: {list(composer.available_adapters.keys())}")
        print("Type 'quit' to exit.\n")

        while True:
            prompt = input(">>> ").strip()
            if prompt.lower() == "quit":
                break

            result = composer.generate(prompt, mode=args.mode)
            print(f"\n[{result.mode} → {result.selected_domain}] ({result.latency_ms:.0f}ms)")
            print(f"Weights: {result.routing_weights}")
            print(f"\n{result.response}\n")

    composer.cleanup()


if __name__ == "__main__":
    main()
