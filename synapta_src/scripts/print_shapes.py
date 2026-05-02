import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_path = "/home/learner/Desktop/mewtwo/models/nemotron"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello, my name is", return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
from transformers_modules.nemotron.modeling_nemotron_h import HybridMambaAttentionDynamicCache

test_cache = HybridMambaAttentionDynamicCache(
    config=model.config,
    batch_size=1,
    dtype=torch.bfloat16,
    device=model.device,
)
print("Initial test_cache layer 3 key shape:", test_cache.key_cache[3].shape)

class PrintHook:
    def __init__(self):
        self.call_count = 0
    def __call__(self, module, input, output):
        print(f"Attention Layer {module.layer_idx} | Call {self.call_count}")
        self.call_count += 1
        print(f"  test_cache AFTER update in layer 3: {test_cache.key_cache[3].shape}")

model.model.layers[3].mixer.register_forward_hook(PrintHook())
model.model.layers[7].mixer.register_forward_hook(PrintHook())

with torch.no_grad():
    out = model.generate(
        **inputs.to(model.device),
        past_key_values=test_cache,
        use_cache=True,
        max_new_tokens=2,
        do_sample=False,
    )
