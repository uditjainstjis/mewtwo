import os
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from safetensors import safe_open

# Global clamp value for ablation studies
_GLOBAL_CLAMP = 0.5

def set_global_clamp(value):
    global _GLOBAL_CLAMP
    _GLOBAL_CLAMP = value

class RoutedLoRALinear(nn.Module):
    def __init__(self, base_layer, in_features, out_features, alpha=16.0):
        super().__init__()
        self.base_layer = base_layer
        self.alpha = alpha
        self.adapters = {}
        self.routing_weights = {}
        
    def add_adapter(self, name, A, B):
        A_arr = mx.array(A)
        self.adapters[name] = {"A": A_arr, "B": mx.array(B)}
        self.routing_weights[name] = mx.array([0.0])
        
        rank = A_arr.shape[-1]
        self.adapters[name]["scale"] = self.alpha / rank
        
    def update_routing_weights(self, weights_dict):
        for name in self.routing_weights.keys():
            self.routing_weights[name] = mx.array([weights_dict.get(name, 0.0)])
            
    def __call__(self, x):
        base_out = self.base_layer(x)
        lora_out = 0.0
        for name, adapter in self.adapters.items():
            w = mx.minimum(self.routing_weights[name], mx.array([_GLOBAL_CLAMP]))
            out = (x @ adapter["A"]) @ adapter["B"]
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

def apply_routed_lora(module, target_modules=["q_proj", "v_proj"], alpha=16.0):
    to_replace = []
    for name, child in module.named_modules():
        if isinstance(child, (nn.Linear, nn.QuantizedLinear)) and any(t in name for t in target_modules):
            in_f, out_f = get_linear_dims(child)
            routed = RoutedLoRALinear(child, in_f, out_f, alpha)
            to_replace.append((name, routed))
            
    for name, routed in to_replace:
        set_module_by_path(module, name, routed)

class DynamicEngine:
    def __init__(self, model_path, registry):
        print(f"Loading Base Engine: {model_path}...")
        self.model, self.tokenizer = load(model_path)
        
        print("Injecting RoutedLoRALinear layers...")
        apply_routed_lora(self.model)
        
        self.registry = registry
        print("Loading adapters into UMA RAM...")
        self.load_all_adapters()
        print("Dynamic Engine fully initialized.")
        
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
            
        for name, child in self.model.named_modules():
            if isinstance(child, RoutedLoRALinear):
                child.update_routing_weights(routing_weights)
        
        start = time.time()
        from mlx_lm import generate
        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        duration = time.time() - start
        
        for name, child in self.model.named_modules():
            if isinstance(child, RoutedLoRALinear):
                child.update_routing_weights({})
                
        return response, duration

    def compute_perplexity(self, prompt, ground_truth, routing_weights=None):
        """Compute perplexity of ground_truth conditioned on prompt under current routing."""
        if routing_weights is None:
            routing_weights = {}

        for name, child in self.model.named_modules():
            if isinstance(child, RoutedLoRALinear):
                child.update_routing_weights(routing_weights)

        full_text = prompt + ground_truth
        tokens = self.tokenizer.encode(full_text)
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        if len(tokens) <= prompt_len:
            return float('inf')

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

        avg_nll = -mx.mean(token_log_probs).item()
        perplexity = float(mx.exp(mx.array(avg_nll)).item())

        for name, child in self.model.named_modules():
            if isinstance(child, RoutedLoRALinear):
                child.update_routing_weights({})

        return perplexity
