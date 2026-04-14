"""
LoRI-MoE Full Model Wrapper

Orchestrates:
  1. Loading the frozen base model (Qwen2.5-3B-Instruct)
  2. Loading/creating the shared B projection
  3. Loading/creating per-domain LoRI expert adapters
  4. Injecting LoRIMoELinear modules at target layers
  5. Managing token-level routers across all layers
  6. Full forward pass with routing + loss computation
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from src.lori_moe.shared_projection import get_shared_projection
from src.lori_moe.lori_adapter import LoRIExpertAdapter, LoRIExpertBank
from src.lori_moe.model.router import MultiLayerRouter
from src.lori_moe.model.lori_moe_linear import LoRIMoELinear
from src.lori_moe.model.losses import LoRIMoELoss
from src.lori_moe.config import LoRIMoEConfig

logger = logging.getLogger(__name__)


class LoRIMoEModel(nn.Module):
    """
    Full LoRI-MoE model combining base LLM + orthogonal experts + token-level routing.

    Training modes:
      - ADAPTER_TRAINING: Train one domain's A matrix (B frozen, router frozen/absent)
      - ROUTER_TRAINING: Train routers (base + all adapters frozen)
      - INFERENCE: Everything frozen, full pipeline active

    Memory budget on RTX 5090 (32GB):
      - Base model (3B, BF16): ~6GB
      - Shared B (rank=32): ~0.25MB per target module (~1.8MB total)
      - 5 Expert A matrices (80% sparse): ~5MB per expert (~25MB total)
      - Routers (64-dim bottleneck, 36 layers): ~0.5MB
      - KV Cache + activations: ~10-15GB
      - Total: ~21GB → 11GB headroom for batching
    """

    def __init__(self, config: LoRIMoEConfig):
        super().__init__()
        self.config = config
        self.expert_names = list(config.domains.keys())
        self.num_experts = len(self.expert_names)

        # These will be populated by setup methods
        self.base_model = None
        self.tokenizer = None
        self.shared_projection = None
        self.routers = None
        self.expert_banks = nn.ModuleDict()  # per-module expert banks
        self.loss_fn = None
        self._target_module_names = []
        self._injected = False
        self._layer_hooks = []
        self._last_routing_by_layer = {}
        self._module_name_to_key = {}
        self._module_key_to_name = {}
        self._module_shared_B = {}
        
        # Ablations
        self._ablation_prompt_routing = False
        self._cached_prompt_routing = {}

    @property
    def device(self) -> torch.device:
        if self.base_model is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return next(self.base_model.parameters()).device

    def load_base_model(self, device_map: str = "auto"):
        """Load the frozen base model and tokenizer."""
        logger.info(f"Loading base model: {self.config.model.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, self.config.model.torch_dtype)
        self.torch_dtype = dtype
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Freeze base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Detect hidden size from model config
        model_config = self.base_model.config
        self.config.model.hidden_size = model_config.hidden_size
        self.config.model.num_layers = model_config.num_hidden_layers

        # Count frozen params
        total_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(
            f"Base model loaded: {total_params / 1e9:.2f}B params (all frozen), "
            f"hidden_size={self.config.model.hidden_size}, "
            f"num_layers={self.config.model.num_layers}"
        )

    def setup_shared_projection(self):
        """
        Initialize or load the fallback shared B projection matrix.

        This only matches modules whose input dimension equals the model hidden size.
        Modules with other input dimensions need a module-specific projection.
        """
        self.shared_projection = get_shared_projection(
            hidden_size=self.config.model.hidden_size,
            rank=self.config.adapter.rank,
            seed=self.config.adapter.shared_b_seed,
            save_path=self.config.paths.shared_b_path,
            device=str(self.device),
            dtype=getattr(torch, self.config.model.torch_dtype),
        )

        # Verify orthogonality
        ortho_stats = self.shared_projection.verify_orthogonality()
        logger.info(f"Orthogonality check: {ortho_stats}")

    def _get_projection_tensor(self, module_name: str, dtype: torch.dtype) -> torch.Tensor:
        """
        Return a frozen shared projection matching a target module.
        Priority:
        1. Explicitly loaded B from PEFT checkpoint (stored in self._module_shared_B[module_key])
        2. Hidden-size matching global shared projection
        3. Module-specific generated projection
        """
        module_key = self._module_key(module_name)
        
        # 1. Check if we have a specifically loaded B for this module (e.g. from PEFT)
        if module_key in self._module_shared_B:
            return self._module_shared_B[module_key]

        # 2. Fallback to input_dim based generic projections
        parent, leaf = self._resolve_parent_module(module_name)
        module = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        input_dim = module.in_features

        if input_dim not in self._module_shared_B:
            if input_dim == self.config.model.hidden_size and self.shared_projection is not None:
                projection = self.shared_projection
            else:
                stem = self.config.paths.shared_b_path.stem
                suffix = self.config.paths.shared_b_path.suffix
                path = self.config.paths.shared_b_path.with_name(f"{stem}_{input_dim}{suffix}")
                projection = get_shared_projection(
                    hidden_size=input_dim,
                    rank=self.config.adapter.rank,
                    seed=self.config.adapter.shared_b_seed,
                    save_path=path,
                    device=str(self.device),
                    dtype=getattr(torch, self.config.model.torch_dtype),
                )
            return projection.B.to(device=self.device, dtype=dtype)
        
        return self._module_shared_B[input_dim]

    def setup_routers(self):
        """Initialize token-level routers for all layers."""
        self.routers = MultiLayerRouter(
            hidden_dim=self.config.model.hidden_size,
            num_experts=self.num_experts,
            num_layers=self.config.model.num_layers,
            bottleneck_dim=self.config.router.bottleneck_dim,
            top_k=self.config.router.top_k,
            noise_std=self.config.router.noise_std,
        )
        self.routers.to(device=self.device, dtype=self.torch_dtype)

        router_params = self.routers.get_total_params()
        logger.info(f"Routers initialized: {router_params:,} trainable params")

    def setup_loss(self):
        """Initialize loss function."""
        self.loss_fn = LoRIMoELoss(
            num_experts=self.num_experts,
            load_balance_weight=self.config.router.load_balance_weight,
        )

    def create_expert_for_domain(self, domain_name: str) -> Dict[str, LoRIExpertAdapter]:
        """
        Create LoRI expert adapters for a single domain across all target modules.
        Used during Phase 1 (adapter training).
        """
        experts = {}
        for module_name in self.config.adapter.target_modules:
            # Find all instances of this module type in the base model
            for name, module in self.base_model.named_modules():
                if name.endswith(module_name) and isinstance(module, nn.Linear):
                    expert = LoRIExpertAdapter(
                        in_features=module.out_features,  # A: out × rank
                        rank=self.config.adapter.rank,
                        alpha=self.config.adapter.alpha,
                        sparsity=self.config.adapter.sparsity_level,
                        domain_name=domain_name,
                    )
                    experts[name] = expert

        logger.info(
            f"Created {len(experts)} expert modules for domain '{domain_name}'"
        )
        return experts

    def get_trainable_params(self, mode: str = "router") -> List[nn.Parameter]:
        """
        Get parameters to train based on current training mode.

        Args:
            mode: "adapter" for Phase 1, "router" for Phase 3
        """
        if mode == "router":
            return list(self.routers.parameters())
        elif mode == "adapter":
            # Only current domain's A matrices
            params = []
            for bank in self.expert_banks.values():
                for expert in bank.experts.values():
                    if isinstance(expert, LoRIExpertAdapter):
                        params.append(expert.A)
            return params
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory usage breakdown."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2),
            }
        return {}

    def print_status(self):
        """Print a comprehensive status of the model."""
        print("\n" + "=" * 60)
        print("LoRI-MoE Model Status")
        print("=" * 60)
        print(f"  Base model:      {self.config.model.base_model}")
        print(f"  Hidden size:     {self.config.model.hidden_size}")
        print(f"  Num layers:      {self.config.model.num_layers}")
        print(f"  Adapter rank:    {self.config.adapter.rank}")
        print(f"  Sparsity:        {self.config.adapter.sparsity_level:.0%}")
        print(f"  Num experts:     {self.num_experts}")
        print(f"  Expert names:    {self.expert_names}")
        print(f"  Router Top-K:    {self.config.router.top_k}")

        if self.routers:
            print(f"  Router params:   {self.routers.get_total_params():,}")
            entropies = self.routers.get_all_entropies()
            avg_entropy = sum(entropies.values()) / len(entropies) if entropies else 0
            print(f"  Avg entropy:     {avg_entropy:.4f}")
            print(f"  Any collapsed:   {self.routers.any_collapsed()}")

        mem = self.get_memory_stats()
        if mem:
            print(f"  GPU memory:      {mem['allocated_gb']:.1f}GB / {mem['total_gb']:.1f}GB")
            print(f"  GPU free:        {mem['free_gb']:.1f}GB")

        print("=" * 60 + "\n")

    def _get_transformer_layers(self):
        """Resolve the decoder layer stack for common HuggingFace causal LM layouts."""
        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before resolving layers")

        candidates = [
            ("model.layers", getattr(getattr(self.base_model, "model", None), "layers", None)),
            (
                "model.model.layers",
                getattr(getattr(getattr(self.base_model, "model", None), "model", None), "layers", None),
            ),
            ("transformer.h", getattr(getattr(self.base_model, "transformer", None), "h", None)),
        ]
        for label, value in candidates:
            if value is not None:
                logger.info("Using transformer layers from %s", label)
                return value
        raise ValueError("Could not locate transformer layers in base model")

    def _resolve_parent_module(self, module_path: str) -> Tuple[nn.Module, str]:
        """Return the parent module and attribute name for a dotted module path."""
        parts = module_path.split(".")
        parent = self.base_model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        return parent, parts[-1]

    def _set_module_by_name(self, module_path: str, module: nn.Module) -> None:
        parent, leaf = self._resolve_parent_module(module_path)
        if leaf.isdigit():
            parent[int(leaf)] = module
        else:
            setattr(parent, leaf, module)

    def _iter_target_linear_modules(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(name.endswith(target) for target in self.config.adapter.target_modules):
                yield name, module

    @staticmethod
    def _module_key(module_name: str) -> str:
        return module_name.replace(".", "__")

    def initialize_random_experts(self) -> None:
        """Create fresh domain experts for each target linear module."""
        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before creating experts")

        self.expert_banks = nn.ModuleDict()
        self._module_shared_B = {}
        target_names = []
        for module_name, module in self._iter_target_linear_modules():
            module_key = self._module_key(module_name)
            expert_configs = {name: {} for name in self.expert_names}
            bank = LoRIExpertBank(
                expert_configs=expert_configs,
                in_features=module.out_features,
                rank=self.config.adapter.rank,
                alpha=self.config.adapter.alpha,
                sparsity=self.config.adapter.sparsity_level,
            )
            self.expert_banks[module_key] = bank
            self._module_name_to_key[module_name] = module_key
            self._module_key_to_name[module_key] = module_name
            if module.in_features == self.config.model.hidden_size and self.shared_projection is not None:
                shared_B = self.shared_projection.B
            else:
                shared_B = torch.randn(
                    self.config.adapter.rank,
                    module.in_features,
                    device=self.device,
                    dtype=module.weight.dtype,
                ) / (self.config.adapter.rank ** 0.5)
            self._module_shared_B[module_key] = shared_B.to(device=self.device, dtype=module.weight.dtype)
            target_names.append(module_name)

        self._target_module_names = target_names
        logger.info("Initialized random experts for %d target modules", len(target_names))

    def load_experts(self, adapters_root: Optional[Path] = None) -> None:
        """
        Load previously saved experts for each target linear module.

        Supports either:
          - native LoRI expert banks under adapters_root/<sanitized_module_name>/*_adapter.pt
          - PEFT checkpoints under adapters_root/<domain>/<checkpoint>/adapter_model.safetensors
        """
        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before loading experts")

        adapters_root = Path(adapters_root or self.config.paths.adapters_dir)
        if self._load_peft_experts(adapters_root):
            return

        loaded_banks = nn.ModuleDict()
        self._module_shared_B = {}
        target_names = []

        for module_name, _ in self._iter_target_linear_modules():
            module_key = self._module_key(module_name)
            module_dir = adapters_root / module_key
            if not module_dir.exists():
                logger.debug("No expert bank found for %s at %s", module_name, module_dir)
                continue

            bank = LoRIExpertBank.load_experts(
                module_dir,
                device=str(self.device),
                dtype=getattr(torch, self.config.model.torch_dtype),
            )
            loaded_banks[module_key] = bank
            self._module_name_to_key[module_name] = module_key
            self._module_key_to_name[module_key] = module_name
            module = dict(self.base_model.named_modules())[module_name]
            if module.in_features == self.config.model.hidden_size and self.shared_projection is not None:
                self._module_shared_B[module_key] = self.shared_projection.B.to(
                    device=self.device,
                    dtype=module.weight.dtype,
                )
            target_names.append(module_name)

        if not loaded_banks:
            raise FileNotFoundError(f"No LoRI expert banks found under {adapters_root}")

        self.expert_banks = loaded_banks
        self._target_module_names = target_names
        logger.info("Loaded expert banks for %d target modules", len(target_names))

    def _normalize_peft_weight_name(self, key: str) -> Optional[Tuple[str, str]]:
        for suffix in (
            ".lora_A.default.weight",
            ".lora_B.default.weight",
            ".lora_A.weight",
            ".lora_B.weight",
        ):
            if key.endswith(suffix):
                module_name = key[: -len(suffix)]
                for prefix in ("base_model.model.", "model."):
                    if module_name.startswith(prefix):
                        module_name = module_name[len(prefix):]
                        break
                component = "lora_A" if "lora_A" in suffix else "lora_B"
                return module_name, component
        return None

    def _load_adapter_state_dict(self, adapter_path: Path) -> Dict[str, torch.Tensor]:
        safetensors_path = adapter_path / "adapter_model.safetensors"
        if safetensors_path.exists():
            return load_safetensors(str(safetensors_path))

        bin_path = adapter_path / "adapter_model.bin"
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu")

        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    def _load_peft_experts(self, adapters_root: Path) -> bool:
        """
        Load PEFT LoRA checkpoints and reinterpret them as LoRI expert banks.

        This bridges the current training path, which saves PEFT adapters instead of
        native LoRI expert-bank checkpoints.
        """
        module_lookup = dict(self.base_model.named_modules())
        selected_paths = {}
        for domain_name in self.expert_names:
            domain_dir = adapters_root / domain_name
            for subdir in ("dare_sparsified", "best", "final"):
                candidate = domain_dir / subdir
                if (candidate / "adapter_config.json").exists():
                    selected_paths[domain_name] = candidate
                    break

        if not selected_paths:
            return False

        loaded_banks = nn.ModuleDict()
        module_shared_B = {}
        target_names = set()
        available_domains = []
        dtype = getattr(torch, self.config.model.torch_dtype)

        for domain_name, adapter_path in selected_paths.items():
            state_dict = self._load_adapter_state_dict(adapter_path)
            loaded_count = 0

            for key, value in state_dict.items():
                parsed = self._normalize_peft_weight_name(key)
                if parsed is None:
                    continue

                module_name, component = parsed
                print(f"Parsed module_name: {module_name}, component: {component}")
                if module_name not in module_lookup:
                    print(f"module_name {module_name} not in module_lookup")
                    continue
                if not any(module_name.endswith(target) for target in self.config.adapter.target_modules):
                    print(f"module_name {module_name} not in target modules")
                    continue

                module = module_lookup[module_name]
                module_key = self._module_key(module_name)

                if component == "lora_A":
                    module_shared_B[module_key] = value.to(device=self.device, dtype=module.weight.dtype)
                    continue

                if module_key not in loaded_banks:
                    loaded_banks[module_key] = nn.ModuleDict()

                expert = LoRIExpertAdapter(
                    in_features=module.out_features,
                    rank=value.shape[1],
                    alpha=self.config.adapter.alpha,
                    sparsity=self.config.adapter.sparsity_level,
                    domain_name=domain_name,
                ).to(device=self.device, dtype=dtype)
                with torch.no_grad():
                    weight = value.to(device=self.device, dtype=dtype)
                    expert.A.copy_(weight)
                    expert.sparse_mask.copy_((weight != 0).to(device=weight.device, dtype=weight.dtype))
                for param in expert.parameters():
                    param.requires_grad = False

                loaded_banks[module_key][domain_name] = expert
                self._module_name_to_key[module_name] = module_key
                self._module_key_to_name[module_key] = module_name
                target_names.add(module_name)
                loaded_count += 1

            if loaded_count:
                available_domains.append(domain_name)

        if not loaded_banks:
            return False

        self.expert_names = available_domains
        self.num_experts = len(self.expert_names)
        self.expert_banks = loaded_banks
        self._module_shared_B = module_shared_B
        self._target_module_names = sorted(target_names)
        logger.info("Loaded PEFT-backed experts for %d target modules", len(self._target_module_names))
        return True

    def inject_lori_layers(self) -> None:
        """Replace target linear modules with LoRIMoE wrappers."""
        if self.base_model is None:
            raise RuntimeError("Base model must be loaded before injection")
        if self.shared_projection is None:
            raise RuntimeError("Shared projection must be set up before injection")
        if not self.expert_banks:
            raise RuntimeError("Expert banks must be initialized or loaded before injection")
        if self._injected:
            logger.info("LoRI-MoE layers already injected")
            return

        injected = 0
        for module_name in self._target_module_names:
            module_key = self._module_name_to_key[module_name]
            parent, leaf = self._resolve_parent_module(module_name)
            base_layer = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
            wrapped = LoRIMoELinear(
                base_layer=base_layer,
                shared_B=self._get_projection_tensor(module_name, base_layer.weight.dtype),
                expert_A_matrices=self.expert_banks[module_key],
                expert_names=self.expert_names,
                scaling=self.config.adapter.alpha / self.config.adapter.rank,
            )
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            injected += 1

        self._register_layer_hooks()
        self._injected = True
        logger.info("Injected LoRI-MoE wrappers into %d target modules", injected)

    def _clear_layer_hooks(self) -> None:
        for hook in self._layer_hooks:
            hook.remove()
        self._layer_hooks = []

    def _register_layer_hooks(self) -> None:
        """Attach per-layer hooks that compute and distribute token routing weights."""
        self._clear_layer_hooks()
        layers = self._get_transformer_layers()

        for layer_idx, layer in enumerate(layers):
            def pre_hook(module, inputs, idx=layer_idx):
                hidden_states = inputs[0]
                
                # Prompt-Level Routing Ablation
                if self._ablation_prompt_routing:
                    # Prefill pass (S > 1)
                    if hidden_states.shape[1] > 1 or idx not in self._cached_prompt_routing:
                        router = self.routers.get_router(idx)
                        if next(router.parameters()).device != hidden_states.device:
                            router.to(device=hidden_states.device, dtype=hidden_states.dtype)
                        routing_weights, _ = router(hidden_states, return_logits=False)
                        
                        # Use the last token of the prompt to avoid left-padding bias 
                        # and to capture the full causal context of the sequence
                        pooled_weights = routing_weights[:, -1:, :] # (B, 1, K)
                        
                        # Top-1 force inside prompt routing:
                        if router.top_k == 1:
                            # Re-apply top-1 to the *pooled* distribution
                            max_indices = pooled_weights.argmax(dim=-1, keepdim=True)
                            pooled_weights = torch.zeros_like(pooled_weights).scatter_(-1, max_indices, 1.0)

                        self._cached_prompt_routing[idx] = pooled_weights
                        
                        # Expand to match sequence length
                        routing_weights = pooled_weights.expand(-1, hidden_states.size(1), -1)
                    else:
                        # Auto-regressive decode pass (S == 1)
                        pooled_weights = self._cached_prompt_routing[idx]
                        routing_weights = pooled_weights.expand(-1, hidden_states.size(1), -1)
                        
                else:
                    # Default Token-level Routing
                    router = self.routers.get_router(idx)
                    if next(router.parameters()).device != hidden_states.device:
                        router.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    routing_weights, _ = router(hidden_states, return_logits=False)
                
                self._last_routing_by_layer[idx] = routing_weights
                for submodule in module.modules():
                    if isinstance(submodule, LoRIMoELinear):
                        submodule.set_current_routing(routing_weights)

            def post_hook(module, inputs, output):
                for submodule in module.modules():
                    if isinstance(submodule, LoRIMoELinear):
                        submodule.clear_current_routing()
                return output

            self._layer_hooks.append(layer.register_forward_pre_hook(pre_hook))
            self._layer_hooks.append(layer.register_forward_hook(post_hook))

    def build(
        self,
        device_map: Optional[str] = None,
        load_experts: bool = False,
        adapters_root: Optional[Path] = None,
    ) -> "LoRIMoEModel":
        """End-to-end assembly helper for the LoRI-MoE runtime model."""
        self.load_base_model(device_map=device_map or self.config.model.device_map)
        self.setup_shared_projection()
        self.setup_routers()
        self.setup_loss()
        if load_experts:
            self.load_experts(adapters_root=adapters_root)
        else:
            self.initialize_random_experts()
        self.inject_lori_layers()
        return self

    def get_router_state_summary(self) -> Dict[str, float]:
        if not self._last_routing_by_layer:
            return {}

        summary = {}
        for layer_idx, weights in self._last_routing_by_layer.items():
            probs = weights.detach().float().mean(dim=(0, 1))
            for expert_idx, expert_name in enumerate(self.expert_names):
                summary[f"layer_{layer_idx}_{expert_name}"] = probs[expert_idx].item()
        return summary

    @contextmanager
    def inference_ablation(
        self,
        top_k: Optional[int] = None,
        noise_std: Optional[float] = None,
        prompt_routing: Optional[bool] = None,
    ):
        """Temporarily override router settings for inference-time ablations."""
        if self.routers is None:
            raise RuntimeError("Routers must be initialized before running ablations")

        original_top_k = [router.top_k for router in self.routers.routers]
        original_noise_std = [router.noise_std for router in self.routers.routers]
        original_prompt_routing = self._ablation_prompt_routing
        
        try:
            if top_k is not None:
                self.routers.set_top_k(top_k)
            if noise_std is not None:
                self.routers.set_noise_std(noise_std)
            if prompt_routing is not None:
                self._ablation_prompt_routing = prompt_routing
                self._cached_prompt_routing.clear()
            yield self
        finally:
            for router, value in zip(self.routers.routers, original_top_k):
                router.top_k = value
            for router, value in zip(self.routers.routers, original_noise_std):
                router.noise_std = value
            self._ablation_prompt_routing = original_prompt_routing
            self._cached_prompt_routing.clear()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.base_model is None or not self._injected:
            raise RuntimeError("Call build() or inject_lori_layers() before forward()")

        self._last_routing_by_layer = {}
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        if self.loss_fn is not None and labels is not None and self._last_routing_by_layer:
            layer_losses = []
            layer_stats = {}
            for layer_idx, routing_weights in self._last_routing_by_layer.items():
                lb_loss, lb_stats = self.loss_fn.lb_loss(routing_weights, attention_mask)
                layer_losses.append(lb_loss)
                layer_stats[f"layer_{layer_idx}_routing_entropy"] = self.routers.get_router(layer_idx).routing_entropy
                layer_stats[f"layer_{layer_idx}_balance_ratio"] = lb_stats["balance_ratio"]

            avg_lb_loss = torch.stack(layer_losses).mean()
            outputs.loss = outputs.loss + avg_lb_loss if outputs.loss is not None else avg_lb_loss
            outputs.load_balance_loss = avg_lb_loss
            outputs.routing_stats = layer_stats
            outputs.routing_summary = self.get_router_state_summary()

        return outputs

    def generate(self, *args, **kwargs):
        if self.base_model is None or not self._injected:
            raise RuntimeError("Call build() or inject_lori_layers() before generate()")
        return self.base_model.generate(*args, **kwargs)
