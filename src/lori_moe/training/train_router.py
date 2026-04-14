#!/usr/bin/env python3
"""
LoRI-MoE Router Training (Phase 3)

Trains a lightweight token-level router that dynamically selects which domain
adapter(s) to apply for each token in the sequence.

Architecture:
    hidden_state (d_model) -> MLP -> softmax -> domain weights (5 dims)

The router is trained on a mixed-domain dataset where each example has a
known ground-truth domain label. We use cross-entropy on the router's
per-token domain predictions.

Usage:
    python -m src.lori_moe.training.train_router \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_dir checkpoints/lori_moe/qwen2.5_1.5b \
        --output_dir checkpoints/lori_moe/qwen2.5_1.5b/router \
        --epochs 5 --batch_size 8 --lr 1e-4
"""

import os
import sys
import json
import time
import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "science", "legal", "medical"]
PROJECT_ROOT = "/home/learner/Desktop/mewtwo"


class RouterMLP(nn.Module):
    """Lightweight token-level domain router.

    Takes hidden states from the base model and produces per-token
    domain routing weights.
    """

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len, num_domains)
        """
        return self.net(hidden_states)


class MixedDomainDataset(Dataset):
    """Loads examples from ALL domains with domain labels."""

    def __init__(self, data_dir: str, tokenizer, max_length: int = 512,
                 max_samples_per_domain: int = 2000):
        self.examples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for domain_idx, domain in enumerate(DOMAINS):
            data_path = Path(data_dir) / f"{domain}_train.jsonl"
            if not data_path.exists():
                logger.warning(f"Missing data for {domain}: {data_path}")
                continue

            count = 0
            with open(data_path) as f:
                for line in f:
                    if count >= max_samples_per_domain:
                        break
                    row = json.loads(line.strip())
                    encoded = tokenizer(
                        row["text"],
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    self.examples.append({
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                    })
                    self.labels.append(domain_idx)
                    count += 1

            logger.info(f"Loaded {count} examples for domain '{domain}' (idx={domain_idx})")

        logger.info(f"Total router training examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "domain_label": self.labels[idx],
        }


def train_router(
    base_model_name: str,
    adapter_dir: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    max_seq_length: int = 512,
    max_samples_per_domain: int = 2000,
):
    """Train the token-level router on mixed-domain data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model (frozen, no adapters — we just want hidden states)
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get hidden dim from model config
    hidden_dim = model.config.hidden_size
    logger.info(f"Hidden dim: {hidden_dim}")

    # Create router
    from src.lori_moe.model.router import MultiLayerRouter
    router = MultiLayerRouter(
        hidden_dim=hidden_dim,
        num_experts=len(DOMAINS),
        num_layers=model.config.num_hidden_layers,
    )
    router = router.to("cuda", dtype=torch.float32)  # Router trains in fp32

    # Dataset
    data_dir = f"{PROJECT_ROOT}/data/lori_moe"
    dataset = MixedDomainDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_samples_per_domain=max_samples_per_domain,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"\n{'='*60}")
    logger.info(f"Router Training (MultiLayer)")
    logger.info(f"{'='*60}")
    logger.info(f"  Domains:     {DOMAINS}")
    logger.info(f"  Dataset:     {len(dataset)} examples")
    logger.info(f"  Batch size:  {batch_size}")
    logger.info(f"  Epochs:      {epochs}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  LR:          {lr}")
    logger.info(f"  Hidden dim:  {hidden_dim}")
    logger.info(f"  Output:      {output_path}")
    logger.info(f"{'='*60}\n")

    best_acc = 0.0
    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        router.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(dataloader, desc=f"[Router] Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            domain_labels = batch["domain_label"].to("cuda")  # (batch,)

            # Get hidden states from frozen base model
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                
            # We train on ALL layers
            total_batch_loss = 0
            batch_correct = 0
            batch_tokens = 0
            
            # Sub-sample layers to speed up training if needed, but 1.5B is fast
            for layer_idx in range(len(router.routers)):
                # hidden_states[layer_idx] is the output of that layer
                h = outputs.hidden_states[layer_idx].float() # (B, S, D)
                
                # Get router for this layer
                r = router.get_router(layer_idx)
                _, logits = r(h, return_logits=True) # (B, S, K)
                
                # Expand labels to (B, S)
                # In this dataset, all tokens in the sequence share the same domain label
                target_labels = domain_labels.unsqueeze(1).expand(-1, h.size(1)) # (B, S)
                
                # Only compute loss on non-pad tokens
                flat_logits = logits.view(-1, len(DOMAINS))
                flat_labels = target_labels.reshape(-1)
                flat_mask = attention_mask.reshape(-1)
                
                # Masked loss
                loss = criterion(flat_logits[flat_mask == 1], flat_labels[flat_mask == 1])
                total_batch_loss += loss
                
                with torch.no_grad():
                    preds = flat_logits.argmax(dim=-1)
                    correct = ((preds == flat_labels) & (flat_mask == 1)).sum().item()
                    batch_correct += correct
                    batch_tokens += flat_mask.sum().item()

            avg_batch_loss = total_batch_loss / len(router.routers)
            
            optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Stats
            epoch_correct += batch_correct
            epoch_total += batch_tokens
            epoch_loss += avg_batch_loss.item()
            global_step += 1

            if global_step % 5 == 0:
                acc = epoch_correct / max(epoch_total, 1) * 100
                pbar.set_postfix(loss=f"{avg_batch_loss.item():.4f}", acc=f"{acc:.1f}%")
                logger.info(
                    f"[Router] Epoch {epoch+1}/{epochs} | Step {step+1} ({global_step}) | "
                    f"Loss: {avg_batch_loss.item():.4f} | Acc: {acc:.1f}% | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        # Epoch summary
        epoch_acc = epoch_correct / max(epoch_total, 1) * 100
        avg_loss = epoch_loss / len(dataloader)
        logger.info(
            f"\n[Router] Epoch {epoch+1} complete — "
            f"Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.1f}%"
        )

        # Save best
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = output_path / "best"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "router_state_dict": router.state_dict(),
                "config": {
                    "hidden_dim": hidden_dim,
                    "num_domains": len(DOMAINS),
                    "domains": DOMAINS,
                    "bottleneck": 128,
                },
                "epoch": epoch + 1,
                "accuracy": epoch_acc,
                "loss": avg_loss,
            }, save_path / "router.pt")
            logger.info(f"  New best! Saved to {save_path} (acc={epoch_acc:.1f}%)")

    elapsed = (time.time() - start_time) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f"Router training complete")
    logger.info(f"  Best accuracy: {best_acc:.1f}%")
    logger.info(f"  Time:          {elapsed:.1f} min")
    logger.info(f"  Output:        {output_path}")
    logger.info(f"{'='*60}\n")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="LoRI-MoE Router Training")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples_per_domain", type=int, default=2000)
    args = parser.parse_args()

    Path(f"{PROJECT_ROOT}/logs/lori_moe").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{PROJECT_ROOT}/logs/lori_moe/train_router.log"),
        ],
    )

    train_router(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        max_samples_per_domain=args.max_samples_per_domain,
    )


if __name__ == "__main__":
    main()
