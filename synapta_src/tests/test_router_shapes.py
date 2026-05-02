import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

from src.lori_moe.model.gc_router import GateConditionedRouter
from src.lori_moe.model.internal_hook import NemotronRouterHook

def test_router_batching():
    print("Testing GCRouter batching...")
    hidden_dim = 256
    router = GateConditionedRouter(
        hidden_dim=hidden_dim,
        num_external_experts=3,
        internal_top_k=8,
        bottleneck_dim=64,
        top_k=2
    )
    router.cuda().eval()

    # Case 1: Batch size 4 (Training)
    B, S, K = 4, 16, 6
    hidden = torch.randn(B, S, hidden_dim).cuda()
    
    # Simulate internal routing signals (B, S, K)
    internal_routing = {
        "top_k_weights": torch.randn(B, S, K).cuda().softmax(-1),
        "entropy": torch.randn(B, S).cuda().abs()
    }
    
    print(f"  Training batch: hidden {hidden.shape}, signals {internal_routing['top_k_weights'].shape}")
    weights, _ = router(hidden, internal_routing)
    print(f"  Output weights: {weights.shape}")
    assert weights.shape == (B, S, 3)
    print("  ✅ Training batch pass OK")

    # Case 2: Batch size 1 (Generation)
    B_inf, S_inf = 1, 1
    hidden_inf = torch.randn(B_inf, S_inf, hidden_dim).cuda()
    internal_routing_inf = {
        "top_k_weights": torch.randn(B_inf, S_inf, K).cuda().softmax(-1),
        "entropy": torch.randn(B_inf, S_inf).cuda().abs()
    }
    print(f"  Inference batch: hidden {hidden_inf.shape}, signals {internal_routing_inf['top_k_weights'].shape}")
    weights_inf, _ = router(hidden_inf, internal_routing_inf)
    print(f"  Output weights: {weights_inf.shape}")
    assert weights_inf.shape == (B_inf, S_inf, 3)
    print("  ✅ Inference batch pass OK")

def test_hook_reshaping():
    print("\nTesting Hook reshaping...")
    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = nn.Module()
            self.config.hidden_size = 256
    
    model = MockModel()
    hooker = NemotronRouterHook(model)
    
    # Simulate a flattened signal (B*S, K) as happens in some MoE layers
    B, S, K = 4, 16, 8
    flattened_weights = torch.randn(B * S, K)
    flattened_entropy = torch.randn(B * S)
    
    # Inject directly into signals dict
    hooker.signals["layer1"] = {
        "top_k_weights": flattened_weights,
        "entropy": flattened_entropy
    }
    
    print(f"  Captured signal: {flattened_weights.shape} (flattened B*S={B*S})")
    agg = hooker.get_aggregated_signal(batch_size=B)
    print(f"  Aggregated weights: {agg['top_k_weights'].shape}")
    
    assert agg['top_k_weights'].shape == (B, S, K)
    assert agg['entropy'].shape == (B, S)
    print("  ✅ Hook reshaping pass OK")

if __name__ == "__main__":
    with torch.no_grad():
        test_router_batching()
        test_hook_reshaping()
    print("\nALL SHAPE TESTS PASSED! 🚀")
