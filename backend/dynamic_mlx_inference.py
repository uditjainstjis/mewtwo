import os
import re
import time
import copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, stream_generate
from mlx_lm.models import cache as mlx_cache
from safetensors import safe_open

# Note: Clamp mode is managed by the DynamicEngine instance via set_clamp_mode().
# Recommended defaults: mode="weight_cap", value=0.5

class RoutedLoRALinear(nn.Module):
    def __init__(self, base_layer, in_features, out_features, alpha=16.0, layer_id: int = -1):
        super().__init__()
        self.base_layer = base_layer
        self.alpha = alpha
        # Transformer block index (0..L-1). -1 = unknown / apply everywhere.
        self.layer_id = layer_id
        # If >= 0: only apply LoRA when layer_id >= adapter_min_layer (late-layer injection).
        self.adapter_min_layer = 0
        # If >= 0: only apply LoRA when layer_id <= adapter_max_layer (early-layer-only injection).
        self.adapter_max_layer = -1
        self.adapters = {}
        self.routing_weights = {}
        self.token_routing = {}
        
        # Injected settings from DynamicEngine
        self.clamp_mode = "weight_cap"
        self.clamp_value = 0.5
        
    def add_adapter(self, name, A, B):
        A_arr = mx.array(A)
        self.adapters[name] = {"A": A_arr, "B": mx.array(B)}
        self.routing_weights[name] = mx.array([0.0])
        
        rank = A_arr.shape[-1]
        self.adapters[name]["scale"] = self.alpha / rank
        
    def update_routing_weights(self, weights_dict):
        self.token_routing = {}
        for name in self.routing_weights.keys():
            raw = weights_dict.get(name, 0.0)
            if isinstance(raw, (list, tuple, np.ndarray)):
                # Token-level schedule: one scalar per token position.
                self.token_routing[name] = np.asarray(raw, dtype=np.float32).reshape(-1)
                self.routing_weights[name] = mx.array([0.0])
            else:
                self.routing_weights[name] = mx.array([float(raw)])

    def _resolve_weight(self, name, x):
        schedule = self.token_routing.get(name)
        if schedule is None:
            return self.routing_weights[name]

        # x shape is typically [B, T, H]. If no token axis, fallback to scalar mean.
        if len(x.shape) < 2:
            return mx.array([float(schedule.mean())])

        seq_len = int(x.shape[1])
        if seq_len <= 0:
            return mx.array([0.0])

        if schedule.shape[0] >= seq_len:
            trimmed = schedule[:seq_len]
        else:
            pad_value = schedule[-1] if schedule.shape[0] > 0 else 0.0
            trimmed = np.pad(schedule, (0, seq_len - schedule.shape[0]), constant_values=pad_value)

        # Broadcast to [1, T, 1] so it scales every token position.
        return mx.array(trimmed.reshape(1, seq_len, 1))
            
    def __call__(self, x):
        base_out = self.base_layer(x)

        # Layer band: [adapter_min_layer, adapter_max_layer] if max is set; else [min_layer, inf).
        if self.layer_id >= 0:
            if self.layer_id < self.adapter_min_layer:
                return base_out
            if self.adapter_max_layer >= 0 and self.layer_id > self.adapter_max_layer:
                return base_out

        if self.clamp_mode == "norm_ratio":
            # v2b: Per-layer activation norm-ratio clamp
            # γ_l = min(1, c * ||z_l||_2 / (||m_l||_2 + ε))
            # h_out = z_l + γ_l * m_l
            lora_sum = mx.zeros_like(base_out)
            for name, weights in self.adapters.items():
                w = self._resolve_weight(name, x)
                # Note: in norm_ratio mode, we don't clamp w individually here, 
                # we clamp the total sum norm ratio below.
                adapter_out = (x @ weights["A"]) @ weights["B"]
                lora_sum = lora_sum + w * (adapter_out * weights["scale"])

            m_norm = mx.linalg.norm(lora_sum, axis=-1, keepdims=True)
            z_norm = mx.linalg.norm(base_out, axis=-1, keepdims=True)
            
            # Use self.clamp_value (c)
            gamma = mx.minimum(1.0, self.clamp_value * z_norm / (m_norm + 1e-6))
            return base_out + gamma * lora_sum

        # Default or "weight_cap":
        lora_sum = mx.zeros_like(base_out)
        for name, weights in self.adapters.items():
            w = self._resolve_weight(name, x)
            # Apply weight clamp c (self.clamp_value)
            w_clamped = mx.minimum(w, self.clamp_value)
            adapter_out = (x @ weights["A"]) @ weights["B"]
            lora_sum = lora_sum + w_clamped * (adapter_out * weights["scale"])

        return base_out + lora_sum

def get_linear_dims(layer):
    if hasattr(layer, "group_size"):
        out_features = layer.scales.shape[0]
        in_features = layer.scales.shape[1] * layer.group_size
        return in_features, out_features
    else:
        out_features, in_features = layer.weight.shape
        return in_features, out_features

def set_module_by_path(root_module, path, new_module):
    parts = path.split(".")
    current = root_module
    for part in parts[:-1]:
        if hasattr(current, "__getitem__") and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    last_part = parts[-1]
    if hasattr(current, "__setitem__") and last_part.isdigit():
        current[int(last_part)] = new_module
    else:
        setattr(current, last_part, new_module)

def _parse_layer_id(module_name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", module_name)
    return int(m.group(1)) if m else -1


def apply_routed_lora(module, target_modules=["q_proj", "v_proj"], alpha=16.0):
    to_replace = []
    for name, child in module.named_modules():
        if isinstance(child, (nn.Linear, nn.QuantizedLinear)) and any(t in name for t in target_modules):
            in_f, out_f = get_linear_dims(child)
            layer_id = _parse_layer_id(name)
            routed = RoutedLoRALinear(child, in_f, out_f, alpha, layer_id=layer_id)
            to_replace.append((name, routed))
            
    for name, routed in to_replace:
        set_module_by_path(module, name, routed)

class DynamicEngine:
    def __init__(self, model_path, registry):
        print(f"Loading Base Engine: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        self._num_layers = self._count_transformer_layers()
        self.clamp_mode = "weight_cap"
        self.clamp_value = 0.5
        
        print("Injecting RoutedLoRALinear layers...")
        apply_routed_lora(self.model)
        
        self.registry = registry
        print("Loading adapters into UMA RAM...")
        self.load_all_adapters()
        self.set_adapter_layer_gate(0)
        print("Dynamic Engine fully initialized.")

    def _iter_routed_layers(self):
        for _, child in self.model.named_modules():
            if isinstance(child, RoutedLoRALinear):
                yield child

    def _count_transformer_layers(self) -> int:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return len(m.model.layers)
        if hasattr(m, "layers"):
            return len(m.layers)
        return 0

    def set_clamp_mode(self, mode: str):
        """Set clamp mode: 'weight_cap' or 'norm_ratio'."""
        assert mode in ("weight_cap", "norm_ratio"), f"Unknown clamp mode: {mode}"
        self.clamp_mode = mode
        for child in self._iter_routed_layers():
            child.clamp_mode = mode

    def set_global_clamp(self, value: float):
        """Set the clamp bound (c)."""
        self.clamp_value = float(value)
        for child in self._iter_routed_layers():
            child.clamp_value = float(value)

    def set_adapter_layer_gate(self, min_layer: int, max_layer: int = -1):
        """
        Apply LoRA only on layers with min_layer <= layer_id <= max_layer (inclusive).
        max_layer < 0 means no upper bound.
        """
        for child in self._iter_routed_layers():
            child.adapter_min_layer = int(max(0, min_layer))
            child.adapter_max_layer = int(max_layer) if max_layer is not None and max_layer >= 0 else -1
        
    def load_all_adapters(self):
        for domain, info in self.registry.items():
            path = info["path"]
            with safe_open(path, framework="np", device="cpu") as f:
                adapter_dict = {}
                for k in f.keys():
                    target_module_path = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "").replace(".lora_a", "").replace(".lora_b", "")
                    if target_module_path not in adapter_dict:
                        adapter_dict[target_module_path] = {}
                    if "lora_A" in k or "lora_a" in k:
                        adapter_dict[target_module_path]["A"] = f.get_tensor(k)
                    elif "lora_B" in k or "lora_b" in k:
                        adapter_dict[target_module_path]["B"] = f.get_tensor(k)
                        
                for name, child in self.model.named_modules():
                    if isinstance(child, RoutedLoRALinear):
                        if name in adapter_dict:
                            A = adapter_dict[name].get("A")
                            B = adapter_dict[name].get("B")
                            if A is not None and B is not None:
                                child.add_adapter(domain, A, B)
                
    def generate(self, prompt, routing_weights=None, max_tokens=100):
        if routing_weights is None:
            routing_weights = {}
            
        self._apply_routing_weights(routing_weights)
        
        start = time.time()
        from mlx_lm import generate
        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        duration = time.time() - start
        
        self._clear_routing_weights()
                
        return response, duration

    def generate_sequential_segments(self, prompt, segments, reset_weights_between=True):
        """
        Token-budget sequential generation: each segment uses its own routing_weights.
        segments: list of (routing_weights: dict, max_tokens: int)
        Returns (assistant_text_only, total_seconds).
        Hypothesis: first N tokens with adapter A, next tokens with adapter B (no weight merge).
        """
        assistant_accum = ""
        total_dur = 0.0
        from mlx_lm import generate as mlx_generate

        for seg_idx, (routing_weights, max_tokens) in enumerate(segments):
            rw = routing_weights if routing_weights is not None else {}
            self._apply_routing_weights(rw)
            full_prompt = prompt + assistant_accum
            t0 = time.time()
            chunk = mlx_generate(
                self.model, self.tokenizer, prompt=full_prompt, max_tokens=max_tokens, verbose=False
            )
            total_dur += time.time() - t0
            assistant_accum += chunk
            if reset_weights_between:
                self._clear_routing_weights()
        return assistant_accum, total_dur

    def _completion_nll_stats(self, prompt, ground_truth, routing_weights=None):
        """Return average NLL and perplexity for a completion conditioned on prompt."""
        if routing_weights is None:
            routing_weights = {}

        self._apply_routing_weights(routing_weights)

        full_text = prompt + ground_truth
        tokens = self.tokenizer.encode(full_text)
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        if len(tokens) <= prompt_len:
            self._clear_routing_weights()
            return float("inf"), float("inf")

        input_ids = mx.array(tokens[:-1])[None, :]
        target_ids = mx.array(tokens[1:])

        logits = self.model(input_ids)
        logits = logits[0]  # remove batch dim

        # Only score the ground truth portion (after prompt)
        gt_start = max(0, prompt_len - 1)
        gt_logits = logits[gt_start:]
        gt_targets = target_ids[gt_start:]

        log_probs = mx.log(mx.softmax(gt_logits, axis=-1))
        token_log_probs = mx.take_along_axis(
            log_probs, gt_targets[:, None], axis=-1
        ).squeeze(-1)

        avg_nll = float((-mx.mean(token_log_probs)).item())
        perplexity = float(mx.exp(mx.array(avg_nll)).item())

        self._clear_routing_weights()

        return avg_nll, perplexity

    def compute_perplexity(self, prompt, ground_truth, routing_weights=None):
        """Compute perplexity of ground_truth conditioned on prompt under current routing."""
        _, perplexity = self._completion_nll_stats(prompt, ground_truth, routing_weights=routing_weights)
        return perplexity

    def score_completion(self, prompt, completion, routing_weights=None):
        """
        Score a candidate completion conditioned on prompt.
        Higher confidence corresponds to lower average NLL / perplexity.
        """
        avg_nll, perplexity = self._completion_nll_stats(prompt, completion, routing_weights=routing_weights)
        return {
            "avg_nll": avg_nll,
            "perplexity": perplexity,
            "confidence": -avg_nll,
        }

    def _apply_routing_weights(self, routing_weights):
        weights = routing_weights or {}
        for child in self._iter_routed_layers():
            child.update_routing_weights(weights)

    def _clear_routing_weights(self):
        for child in self._iter_routed_layers():
            child.update_routing_weights({})

    def _encode_prompt_tokens(self, prompt):
        if isinstance(prompt, mx.array):
            return prompt.astype(mx.uint32)
        if isinstance(prompt, (list, tuple, np.ndarray)):
            return mx.array(prompt, dtype=mx.uint32)
        if isinstance(prompt, str):
            add_special_tokens = (
                getattr(self.tokenizer, "bos_token", None) is None
                or not prompt.startswith(getattr(self.tokenizer, "bos_token", ""))
            )
            return mx.array(
                self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens),
                dtype=mx.uint32,
            )
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

    def prepare_prompt_cache(self, prompt, prefill_step_size: int = 2048):
        """
        Prefill the prompt KV-cache once and return a cache plus the final prompt
        token needed to start decoding.
        """
        prompt_tokens = self._encode_prompt_tokens(prompt)
        if int(prompt_tokens.shape[0]) <= 0:
            raise ValueError("Prompt must contain at least one token for KV-cache prefill.")

        prompt_cache = mlx_cache.make_prompt_cache(self.model)
        remaining = prompt_tokens

        while int(remaining.shape[0]) > 1:
            n_to_process = min(prefill_step_size, int(remaining.shape[0]) - 1)
            self.model(remaining[:n_to_process][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            remaining = remaining[n_to_process:]
            mx.clear_cache()

        return prompt_cache, remaining

    def generate_from_prompt_cache(
        self,
        *,
        prompt_cache,
        decode_prompt_tokens,
        routing_weights=None,
        max_tokens=100,
    ):
        """
        Decode using an already-prefilled prompt cache. The provided cache is
        copied so multiple branches can reuse the same prefill state without
        re-running the prompt.
        """
        self._apply_routing_weights(routing_weights)
        local_cache = copy.deepcopy(prompt_cache)
        start = time.time()
        pieces = []
        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=decode_prompt_tokens,
                max_tokens=max_tokens,
                prompt_cache=local_cache,
            ):
                if response.text:
                    pieces.append(response.text)
        finally:
            self._clear_routing_weights()
        return "".join(pieces), time.time() - start
