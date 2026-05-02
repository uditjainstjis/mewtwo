import torch
from transformers_modules.nemotron.modeling_nemotron_h import HybridMambaAttentionDynamicCache

class DummyConfig:
    num_hidden_layers = 10
    mamba_num_heads = 32
    mamba_head_dim = 128
    ssm_state_size = 16
    conv_kernel = 4
    hybrid_override_pattern = "MAMBA" * 2

cache = HybridMambaAttentionDynamicCache(DummyConfig(), 1, torch.float16, "cpu")
print("Cache length:", cache.get_seq_length())
