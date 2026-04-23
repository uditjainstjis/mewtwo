"""
Nemotron Internal Router Hook Extractor

Hooks into Nemotron's internal MoE routers (NemotronHTopkRouter) to extract
per-token routing signals. These signals are fed to GC-LoRI for conditioning.

Usage:
    hooker = NemotronRouterHook(model)
    hooker.install()

    # Run a forward pass as normal
    outputs = model(**inputs, output_hidden_states=True)

    # Extract captured signals
    signals = hooker.get_aggregated_signal()
    # signals = {'top_k_weights': (B, S, K), 'entropy': (B, S)}

    hooker.clear()   # clear between batches
    hooker.remove()  # cleanup when done

Note: All hooks use .detach() to prevent gradient flow back through the
      internal router. This is critical — we observe, not modify.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NemotronRouterHook:
    """
    Extracts internal MoE routing signals from Nemotron's TopkRouter layers.

    The hook captures:
    - top_k_weights: the normalized weights assigned to selected experts
    - top_k_indices: which experts were selected
    - entropy: routing entropy (uncertainty) per token
    """

    # Known Nemotron router class names
    ROUTER_CLASS_NAMES = {
        "NemotronHTopkRouter",
        "TopkRouter",
    }

    def __init__(
        self,
        model: Any,
        router_pattern: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            model: The loaded Nemotron model
            router_pattern: Override to match specific module class names
            verbose: Log every hook activation
        """
        self.model = model
        self.hooks: List = []
        self.signals: Dict[str, Dict[str, torch.Tensor]] = {}
        self.moe_layer_names: List[str] = []
        self.verbose = verbose

        # Discover router modules
        target_names = {router_pattern} if router_pattern else self.ROUTER_CLASS_NAMES

        for name, module in model.named_modules():
            class_name = type(module).__name__
            # Match by class name or by name pattern
            if class_name in target_names or (
                "router" in name.lower() and hasattr(module, "weight")
            ):
                self.moe_layer_names.append(name)

        logger.info(
            f"NemotronRouterHook: found {len(self.moe_layer_names)} router modules"
        )
        if self.verbose and self.moe_layer_names:
            for n in self.moe_layer_names[:5]:
                logger.info(f"  → {n}")
            if len(self.moe_layer_names) > 5:
                logger.info(f"  ... and {len(self.moe_layer_names) - 5} more")

    def install(self) -> "NemotronRouterHook":
        """Install forward hooks on all discovered router layers."""
        if self.hooks:
            logger.warning("Hooks already installed. Call remove() first.")
            return self

        for name in self.moe_layer_names:
            # Navigate to the module by name
            module = self._get_module_by_name(name)
            if module is not None:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        logger.info(f"Installed {len(self.hooks)} router hooks")
        return self

    def _get_module_by_name(self, name: str):
        """Get a module by its dot-separated name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, "__getitem__"):
                module = module[int(part)]
            else:
                logger.warning(f"Could not traverse to module: {name} (stuck at {part})")
                return None
        return module

    def _make_hook(self, layer_name: str):
        """Create a forward hook for a specific router layer."""

        def hook_fn(module, input_args, output):
            try:
                if isinstance(output, tuple) and len(output) >= 2:
                    # Standard Nemotron TopkRouter output: (weights, indices)
                    weights = output[0].detach().float()
                    indices = output[1].detach()

                    # Compute entropy from weights
                    probs = weights.clamp(min=1e-8)
                    entropy = -(probs * probs.log()).sum(dim=-1)

                    self.signals[layer_name] = {
                        "top_k_weights": weights,
                        "top_k_indices": indices,
                        "entropy": entropy,
                    }

                elif hasattr(output, "detach"):
                    # Fallback: treat output as raw logits
                    logits = output.detach().float()
                    probs = F.softmax(logits, dim=-1)
                    k = min(8, probs.shape[-1])
                    top_k_weights, top_k_indices = probs.topk(k, dim=-1)
                    entropy = -(
                        probs.clamp(min=1e-8) * probs.clamp(min=1e-8).log()
                    ).sum(-1)

                    self.signals[layer_name] = {
                        "top_k_weights": top_k_weights,
                        "top_k_indices": top_k_indices,
                        "entropy": entropy,
                    }

                if self.verbose:
                    sig = self.signals.get(layer_name, {})
                    w_shape = sig.get("top_k_weights", torch.empty(0)).shape
                    logger.debug(f"Hook {layer_name}: weights shape={w_shape}")

            except Exception as e:
                logger.warning(f"Hook failed for {layer_name}: {e}")

        return hook_fn

    def get_signals(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all captured routing signals, keyed by layer name."""
        return self.signals

    def get_aggregated_signal(self, batch_size: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate routing signals across all MoE layers into a single
        conditioning tensor suitable for GC-LoRI.

        Args:
            batch_size: Optional expected batch size to reshape flattened signals.

        Returns:
            Dict with:
              'top_k_weights': mean top-K weights across layers (B, S, K)
              'entropy': mean entropy across layers (B, S)
            or None if no signals captured.
        """
        if not self.signals:
            logger.warning("No routing signals captured. Did you run a forward pass?")
            return None

        all_weights = []
        all_entropy = []

        for layer_name, sig in self.signals.items():
            w = sig["top_k_weights"]
            e = sig["entropy"]

            # Handle flattened MoE signals (B*S, K) -> (B, S, K)
            if batch_size is not None and w.dim() == 2 and w.shape[0] != batch_size:
                S = w.shape[0] // batch_size
                w = w.view(batch_size, S, -1)
                e = e.view(batch_size, S)
            
            # Ensure consistent batch/seq dims for raw unflattened signals
            if w.dim() == 2:
                w = w.unsqueeze(0)  # add batch dim
            if e.dim() == 1:
                e = e.unsqueeze(0)

            all_weights.append(w)
            all_entropy.append(e)

        # Find minimum K across layers for stacking
        min_k = min(w.shape[-1] for w in all_weights)

        # Find minimum seq_len across layers (should be identical)
        min_s = min(w.shape[-2] for w in all_weights)

        # Stack and average
        stacked_w = torch.stack(
            [w[..., :min_s, :min_k] for w in all_weights], dim=0
        )
        stacked_e = torch.stack(
            [e[..., :min_s] for e in all_entropy], dim=0
        )

        return {
            "top_k_weights": stacked_w.mean(dim=0),  # (B, S, K)
            "entropy": stacked_e.mean(dim=0),  # (B, S)
        }

    def get_layer_entropy_profile(self) -> Dict[str, float]:
        """
        Get per-layer mean entropy — useful for understanding which layers
        have uncertain routing (and thus benefit most from external adapters).
        """
        profile = {}
        for name, sig in self.signals.items():
            profile[name] = sig["entropy"].mean().item()
        return profile

    def get_entropy_stats(self) -> Dict[str, float]:
        """Get aggregate entropy statistics."""
        if not self.signals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        all_entropy = []
        for sig in self.signals.values():
            all_entropy.append(sig["entropy"].mean().item())

        import statistics

        return {
            "mean": statistics.mean(all_entropy),
            "std": statistics.stdev(all_entropy) if len(all_entropy) > 1 else 0.0,
            "min": min(all_entropy),
            "max": max(all_entropy),
            "num_layers": len(all_entropy),
        }

    def clear(self):
        """Clear captured signals between forward passes."""
        self.signals.clear()

    def remove(self):
        """Remove all hooks and cleanup."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.signals.clear()
        logger.info("All router hooks removed")

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *args):
        self.remove()

    def __repr__(self):
        return (
            f"NemotronRouterHook("
            f"layers={len(self.moe_layer_names)}, "
            f"hooks_active={len(self.hooks)}, "
            f"signals_captured={len(self.signals)})"
        )
