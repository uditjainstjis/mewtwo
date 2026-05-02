#!/usr/bin/env python3
"""
Train Neural Router on Nemotron-30B Hidden States
"""
import sys, logging
import torch
import torch.nn as nn
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
DATA_PATH = PROJECT / "data" / "neural_router_train_data.pt"
SAVE_PATH = PROJECT / "adapters" / "routers" / "neural_mlp_router.pt"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_router")

class SimpleNeuralRouter(nn.Module):
    def __init__(self, hidden_dim=2688, bottleneck=256, num_classes=3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(self.norm(x))

def main():
    log.info("Loading training data...")
    if not DATA_PATH.exists():
        log.error(f"Data file not found at {DATA_PATH}")
        sys.exit(1)
        
    data = torch.load(DATA_PATH)
    states = torch.tensor(data["states"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    
    log.info(f"Data shape: {states.shape}, Labels shape: {labels.shape}")
    
    router = SimpleNeuralRouter().to("cuda")
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Shuffle
    idx = torch.randperm(states.size(0))
    states, labels = states[idx].to("cuda"), labels[idx].to("cuda")
    
    epochs = 150
    batch_size = 64
    
    router.train()
    for ep in range(epochs):
        total_loss = 0
        correct = 0
        for i in range(0, states.size(0), batch_size):
            batch_x = states[i:i+batch_size]
            batch_y = labels[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = router(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            
        acc = correct / states.size(0)
        if ep % 25 == 0 or ep == epochs - 1:
            log.info(f"Epoch {ep:3d} | Loss: {total_loss/states.size(0):.4f} | Acc: {acc:.1%}")
            
    # Save
    router.eval()
    torch.save(router.state_dict(), SAVE_PATH)
    log.info(f"✅ Router saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
