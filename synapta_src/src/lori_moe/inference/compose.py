#!/usr/bin/env python3
"""
LoRI-MoE Inference Engine

Loads the base model, all trained domain adapters, and the router to perform
dynamic token-level adapter composition at inference time.

Usage:
    python -m src.lori_moe.inference.compose \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_dir adapters/lori_moe/qwen2.5_1.5b \
        --prompt "Solve x^2 + 5x + 6 = 0"
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "science", "legal", "medical"]


class RouterMLP(nn.Module):
    """Mirror of the trained router architecture."""

    def __init__(self, hidden_dim: int, num_domains: int = 5, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, num_domains),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class LoRIMoEComposer:
    """Composes multiple LoRI adapters using a trained router."""

    def __init__(self, base_model_name: str, adapter_dir: str, device: str = "cuda"):
        self.device = device
        self.adapter_dir = Path(adapter_dir)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Discover and load available adapters
        self.available_domains = []
        for domain in DOMAINS:
            adapter_path = self.adapter_dir / domain / "dare_sparsified"
            if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
                self.available_domains.append(domain)
                logger.info(f"Found adapter: {domain}")
            else:
                logger.warning(f"Missing adapter: {domain}")

        # Load router if available
        router_path = self.adapter_dir / "router" / "best" / "router.pt"
        self.router = None
        if router_path.exists():
            checkpoint = torch.load(router_path, map_location=device, weights_only=True)
            config = checkpoint["config"]
            self.router = RouterMLP(
                hidden_dim=config["hidden_dim"],
                num_domains=config["num_domains"],
                bottleneck=config.get("bottleneck", 128),
            )
            self.router.load_state_dict(checkpoint["router_state_dict"])
            self.router.to(device)
            self.router.eval()
            logger.info(f"Router loaded (trained acc: {checkpoint.get('accuracy', 'N/A')}%)")
        else:
            logger.warning("No router found — will use uniform adapter weighting or manual domain selection")

        logger.info(f"LoRI-MoE Composer ready. Domains: {self.available_domains}")

    def route_prompt(self, prompt: str) -> dict:
        """Use the router to determine domain weights for a prompt."""
        if self.router is None:
            # Uniform fallback
            n = len(self.available_domains)
            return {d: 1.0 / n for d in self.available_domains}

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1].float()  # (1, S, D)

            # Average pool
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, D)

            logits = self.router(pooled.unsqueeze(1)).squeeze(1)  # (1, num_domains)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        weights = {}
        for i, domain in enumerate(DOMAINS):
            if domain in self.available_domains:
                weights[domain] = probs[i].item()

        return weights

    def generate(self, prompt: str, domain: str = None, max_new_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text using a specific domain adapter or auto-routed."""

        if domain is None:
            # Auto-route
            weights = self.route_prompt(prompt)
            domain = max(weights, key=weights.get)
            logger.info(f"Auto-routed to '{domain}' (weights: {weights})")

        # Load specific adapter
        adapter_path = self.adapter_dir / domain / "dare_sparsified"
        if not adapter_path.exists():
            logger.error(f"Adapter not found: {adapter_path}")
            return ""

        # Load adapter dynamically
        model = PeftModel.from_pretrained(
            self.model, str(adapter_path), is_trainable=False
        )
        model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Unload adapter to free memory
        model.unload()
        del model
        torch.cuda.empty_cache()

        return response

    def benchmark_routing(self, prompts: list) -> dict:
        """Benchmark routing accuracy across test prompts."""
        results = []
        for item in prompts:
            prompt = item["prompt"]
            expected = item.get("domain", None)
            weights = self.route_prompt(prompt)
            predicted = max(weights, key=weights.get)
            results.append({
                "prompt": prompt[:80] + "...",
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted if expected else None,
                "weights": {k: f"{v:.3f}" for k, v in weights.items()},
            })

        correct = sum(1 for r in results if r["correct"] is True)
        total = sum(1 for r in results if r["correct"] is not None)
        accuracy = correct / max(total, 1) * 100

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="LoRI-MoE Inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str,
                        default="/home/learner/Desktop/mewtwo/adapters/lori_moe/qwen2.5_1.5b")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None, choices=DOMAINS)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    composer = LoRIMoEComposer(args.base_model, args.adapter_dir)

    if args.prompt:
        response = composer.generate(args.prompt, domain=args.domain)
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'='*60}")
        print(f"\n{response}\n")

    elif args.interactive:
        print("\n🧠 LoRI-MoE Interactive Mode")
        print(f"Available domains: {composer.available_domains}")
        print("Type 'quit' to exit, 'route:<prompt>' to see routing weights\n")

        while True:
            prompt = input(">>> ").strip()
            if prompt.lower() == "quit":
                break
            if prompt.startswith("route:"):
                weights = composer.route_prompt(prompt[6:])
                print(f"  Routing: {weights}")
                continue

            response = composer.generate(prompt, domain=args.domain)
            print(f"\n{response}\n")


if __name__ == "__main__":
    main()
