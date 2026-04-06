import time
from safetensors import safe_open
from runtime_backend import detect_runtime_backend, get_cuda_max_memory, resolve_model_path

try:
    import mlx.nn as _mlx_nn
except Exception:
    _mlx_nn = None

try:
    import torch.nn as _torch_nn
except Exception:
    _torch_nn = None

# Global clamp value for ablation studies
_GLOBAL_CLAMP = 0.5
# Clamp mode: "weight_cap" (original v1/v2) or "norm_ratio" (v2b activation clamp)
_CLAMP_MODE = "weight_cap"

def set_global_clamp(value):
    global _GLOBAL_CLAMP
    _GLOBAL_CLAMP = value

def set_clamp_mode(mode):
    """Set clamp mode: 'weight_cap' (default, per-adapter min(w,c)) or 'norm_ratio' (per-layer γ=min(1, c·‖z‖/‖m‖))."""
    global _CLAMP_MODE
    assert mode in ("weight_cap", "norm_ratio"), f"Unknown clamp mode: {mode}"
    _CLAMP_MODE = mode

def get_clamp_mode():
    return _CLAMP_MODE

class _MLXModule(_mlx_nn.Module if _mlx_nn is not None else object):
    pass


class _TorchModule(_torch_nn.Module if _torch_nn is not None else object):
    pass


class RoutedLoRALinearMLX(_MLXModule):
    def __init__(self, base_layer, in_features, out_features, alpha=16.0):
        import mlx.core as mx

        super().__init__()
        self.base_layer = base_layer
        self.alpha = alpha
        self.adapters = {}
        self.routing_weights = {}
        self.mx = mx
        
    def add_adapter(self, name, A, B):
        A_arr = self.mx.array(A)
        self.adapters[name] = {"A": A_arr, "B": self.mx.array(B)}
        self.routing_weights[name] = self.mx.array([0.0])
        
        rank = A_arr.shape[-1]
        self.adapters[name]["scale"] = self.alpha / rank
        
    def update_routing_weights(self, weights_dict):
        for name in self.routing_weights.keys():
            self.routing_weights[name] = self.mx.array([weights_dict.get(name, 0.0)])
            
    def __call__(self, x):
        mx = self.mx
        base_out = self.base_layer(x)

        if _CLAMP_MODE == "norm_ratio":
            # v2b: Per-layer activation norm-ratio clamp
            # γ_l = min(1, c * ||z_l||_2 / (||m_l||_2 + ε))
            # h_out = z_l + γ_l * m_l
            lora_sum = mx.zeros_like(base_out)
            for name, adapter in self.adapters.items():
                w = self.routing_weights[name]
                out = (x @ adapter["A"]) @ adapter["B"]
                lora_sum = lora_sum + (w * adapter["scale"] * out)
            # Compute norms over last dimension
            eps = 1e-6
            base_norm = mx.sqrt(mx.sum(base_out * base_out, axis=-1, keepdims=True) + eps)
            lora_norm = mx.sqrt(mx.sum(lora_sum * lora_sum, axis=-1, keepdims=True) + eps)
            gamma = mx.minimum(mx.array([1.0]), mx.array([_GLOBAL_CLAMP]) * base_norm / lora_norm)
            return base_out + gamma * lora_sum
        else:
            # Original v1/v2: Per-adapter weight cap min(w, c)
            lora_out = 0.0
            for name, adapter in self.adapters.items():
                w = mx.minimum(self.routing_weights[name], mx.array([_GLOBAL_CLAMP]))
                out = (x @ adapter["A"]) @ adapter["B"]
                lora_out = lora_out + (w * adapter["scale"] * out)
            return base_out + lora_out


class RoutedLoRALinearTorch(_TorchModule):
    def __init__(self, base_layer, in_features, out_features, alpha=16.0):
        import torch
        import torch.nn as tnn

        super().__init__()
        self.base_layer = base_layer
        self.alpha = alpha
        self.adapters = {}
        self.routing_weights = {}
        self.torch = torch
        self.tnn = tnn

    def add_adapter(self, name, A, B):
        torch = self.torch
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        A_tensor = torch.as_tensor(A, device=device, dtype=dtype)
        B_tensor = torch.as_tensor(B, device=device, dtype=dtype)
        self.adapters[name] = {"A": A_tensor, "B": B_tensor}
        self.routing_weights[name] = torch.tensor([0.0], device=device, dtype=dtype)

        rank = A_tensor.shape[0]
        self.adapters[name]["scale"] = self.alpha / rank

    def update_routing_weights(self, weights_dict):
        torch = self.torch
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        for name in self.routing_weights.keys():
            self.routing_weights[name] = torch.tensor([weights_dict.get(name, 0.0)], device=device, dtype=dtype)

    def forward(self, x):
        torch = self.torch
        F = self.torch.nn.functional
        base_out = self.base_layer(x)

        if _CLAMP_MODE == "norm_ratio":
            lora_sum = torch.zeros_like(base_out)
            for name, adapter in self.adapters.items():
                w = self.routing_weights[name]
                out = F.linear(F.linear(x, adapter["A"]), adapter["B"])
                lora_sum = lora_sum + (w * adapter["scale"] * out)
            eps = 1e-6
            base_norm = torch.sqrt(torch.sum(base_out * base_out, dim=-1, keepdim=True) + eps)
            lora_norm = torch.sqrt(torch.sum(lora_sum * lora_sum, dim=-1, keepdim=True) + eps)
            gamma = torch.minimum(
                torch.ones(1, device=base_out.device, dtype=base_out.dtype),
                torch.tensor([_GLOBAL_CLAMP], device=base_out.device, dtype=base_out.dtype) * base_norm / lora_norm,
            )
            return base_out + gamma * lora_sum

        lora_out = torch.zeros_like(base_out)
        for name, adapter in self.adapters.items():
            w = torch.minimum(
                self.routing_weights[name],
                torch.tensor([_GLOBAL_CLAMP], device=base_out.device, dtype=base_out.dtype),
            )
            out = F.linear(F.linear(x, adapter["A"]), adapter["B"])
            lora_out = lora_out + (w * adapter["scale"] * out)
        return base_out + lora_out

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

def apply_routed_lora(module, target_modules=["q_proj", "v_proj"], alpha=16.0, backend=None):
    backend = detect_runtime_backend(backend)
    to_replace = []
    if backend == "mlx":
        import mlx.nn as nn

        for name, child in module.named_modules():
            if isinstance(child, (nn.Linear, nn.QuantizedLinear)) and any(t in name for t in target_modules):
                in_f, out_f = get_linear_dims(child)
                routed = RoutedLoRALinearMLX(child, in_f, out_f, alpha)
                to_replace.append((name, routed))
    else:
        import torch.nn as nn

        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                in_f, out_f = get_linear_dims(child)
                routed = RoutedLoRALinearTorch(child, in_f, out_f, alpha)
                to_replace.append((name, routed))
            
    for name, routed in to_replace:
        set_module_by_path(module, name, routed)

class DynamicEngine:
    def __init__(self, model_path, registry, backend=None):
        self.backend = detect_runtime_backend(backend)
        self.model_path = resolve_model_path(model_path, self.backend)
        print(f"Loading Base Engine [{self.backend}]: {self.model_path}...")
        self._load_runtime()
        print("Injecting RoutedLoRALinear layers...")
        apply_routed_lora(self.model, backend=self.backend)
        
        self.registry = registry
        print("Loading adapters into UMA RAM...")
        self.load_all_adapters()
        print("Dynamic Engine fully initialized.")

    def _load_runtime(self):
        if self.backend == "mlx":
            from mlx_lm import load

            self.model, self.tokenizer = load(self.model_path)
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            max_memory=get_cuda_max_memory() if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
        )
        self.model.eval()
        
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
                        
                routed_types = (RoutedLoRALinearMLX, RoutedLoRALinearTorch)
                for name, child in self.model.named_modules():
                    if isinstance(child, routed_types):
                        if name in adapter_dict:
                            A = adapter_dict[name].get("A")
                            B = adapter_dict[name].get("B")
                            if A is not None and B is not None:
                                child.add_adapter(domain, A, B)
                
    def generate(self, prompt, routing_weights=None, max_tokens=100):
        if routing_weights is None:
            routing_weights = {}
            
        for name, child in self.model.named_modules():
            if isinstance(child, (RoutedLoRALinearMLX, RoutedLoRALinearTorch)):
                child.update_routing_weights(routing_weights)
        
        start = time.time()
        if self.backend == "mlx":
            from mlx_lm import generate

            response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        else:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        duration = time.time() - start
        
        for name, child in self.model.named_modules():
            if isinstance(child, (RoutedLoRALinearMLX, RoutedLoRALinearTorch)):
                child.update_routing_weights({})
                
        return response, duration

    def compute_perplexity(self, prompt, ground_truth, routing_weights=None):
        """Compute perplexity of ground_truth conditioned on prompt under current routing."""
        if routing_weights is None:
            routing_weights = {}

        for name, child in self.model.named_modules():
            if isinstance(child, (RoutedLoRALinearMLX, RoutedLoRALinearTorch)):
                child.update_routing_weights(routing_weights)

        if self.backend == "mlx":
            import mlx.core as mx

            full_text = prompt + ground_truth
            tokens = self.tokenizer.encode(full_text)
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)

            if len(tokens) <= prompt_len:
                return float("inf")

            input_ids = mx.array(tokens[:-1])[None, :]
            target_ids = mx.array(tokens[1:])

            logits = self.model(input_ids)
            logits = logits[0]

            gt_start = max(0, prompt_len - 1)
            gt_logits = logits[gt_start:]
            gt_targets = target_ids[gt_start:]

            log_probs = mx.log(mx.softmax(gt_logits, axis=-1))
            token_log_probs = mx.take_along_axis(log_probs, gt_targets[:, None], axis=-1).squeeze(-1)

            avg_nll = -mx.mean(token_log_probs).item()
            perplexity = float(mx.exp(mx.array(avg_nll)).item())
        else:
            import torch

            full_text = prompt + ground_truth
            tokens = self.tokenizer(full_text, return_tensors="pt")
            prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            input_ids = tokens["input_ids"].to(next(self.model.parameters()).device)
            if input_ids.shape[1] <= prompt_len:
                return float("inf")
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0, :-1, :]
            target_ids = input_ids[0, 1:]
            gt_start = max(0, prompt_len - 1)
            gt_logits = logits[gt_start:]
            gt_targets = target_ids[gt_start:]
            log_probs = torch.log_softmax(gt_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, gt_targets.unsqueeze(-1)).squeeze(-1)
            avg_nll = -token_log_probs.mean().item()
            perplexity = float(torch.exp(torch.tensor(avg_nll)).item())

        for name, child in self.model.named_modules():
            if isinstance(child, (RoutedLoRALinearMLX, RoutedLoRALinearTorch)):
                child.update_routing_weights({})

        return perplexity
