#!/usr/bin/env python3
"""
GC-LoRI Router Training Pipeline

Trains the Gate-Conditioned Router using:
  1. Hidden states from the frozen Nemotron base model
  2. Internal MoE routing signals extracted via hooks
  3. Ground-truth domain labels from the training data

The key innovation: this router sees BOTH the hidden state AND the internal
routing signal. The control baseline (blind router) only sees hidden states.

Training produces two checkpoints:
  - gc_router/best/gc_router.pt     — the innovation (gate-conditioned)
  - gc_router/blind/blind_router.pt — the control (hidden-only)

Usage:
    python -m src.lori_moe.training.train_gc_router \
        --base_model ./models/nemotron \
        --data_dir ./data/nemotron \
        --output_dir ./adapters/nemotron_30b/gc_router \
        --epochs 5 --lr 5e-4 --use_4bit
"""

import os
import sys
import json
import time
import argparse
import logging
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT_ROOT))

from src.lori_moe.model.gc_router import GateConditionedRouter
from src.lori_moe.model.internal_hook import NemotronRouterHook
from src.lori_moe.model.router import TokenRouter

logger = logging.getLogger(__name__)

DOMAINS = ["math", "code", "science"]  # Nemotron track: 3 reasoning domains


class MixedDomainDataset(Dataset):
    """Loads examples from all domains with domain labels for router training."""

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        domains: List[str],
        max_length: int = 512,
        max_samples_per_domain: int = 2000,
    ):
        self.examples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for domain_idx, domain in enumerate(domains):
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
                    self.examples.append(row["text"])
                    self.labels.append(domain_idx)
                    count += 1

            logger.info(f"Loaded {count} examples for '{domain}' (idx={domain_idx})")

        logger.info(f"Total router training examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "domain_label": self.labels[idx],
        }


def train_gc_router(
    base_model_name: str,
    data_dir: str,
    output_dir: str,
    domains: List[str] = None,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-4,
    max_seq_length: int = 512,
    max_samples_per_domain: int = 2000,
    use_4bit: bool = True,
    internal_top_k: int = 8,
    bottleneck_dim: int = 128,
    router_top_k: int = 2,
):
    """
    Train both GC-LoRI Router (innovation) and Blind Router (control).
    
    The training loop:
    1. Forward pass through frozen Nemotron → hidden states
    2. Internal hooks capture MoE routing signals
    3. GC-LoRI router sees (hidden + internal signals) → predict domain
    4. Blind router sees (hidden only) → predict domain
    5. Cross-entropy loss against ground-truth domain labels
    """
    domains = domains or DOMAINS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load frozen base model
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    logger.info(f"Hidden dim: {hidden_dim}")

    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Base model VRAM: {vram:.1f} GB")

    # Install internal router hooks
    hooker = NemotronRouterHook(model)
    hooker.install()
    logger.info(f"Internal hooks installed: {hooker}")

    # Create both routers
    gc_router = GateConditionedRouter(
        hidden_dim=hidden_dim,
        num_external_experts=len(domains),
        internal_top_k=internal_top_k,
        bottleneck_dim=bottleneck_dim,
        top_k=router_top_k,
        noise_std=0.1,
        load_balance_weight=0.01,
    ).to("cuda", dtype=torch.float32)

    blind_router = TokenRouter(
        hidden_dim=hidden_dim,
        num_experts=len(domains),
        bottleneck_dim=bottleneck_dim // 2,
        top_k=router_top_k,
        noise_std=0.1,
    ).to("cuda", dtype=torch.float32)

    logger.info(f"GC-LoRI Router params: {sum(p.numel() for p in gc_router.parameters()):,}")
    logger.info(f"Blind Router params: {sum(p.numel() for p in blind_router.parameters()):,}")

    # Dataset
    dataset = MixedDomainDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        domains=domains,
        max_length=max_seq_length,
        max_samples_per_domain=max_samples_per_domain,
    )
    
    # 80/20 train/val split
    from torch.utils.data import random_split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Optimizers (separate for each router)
    gc_optimizer = torch.optim.AdamW(gc_router.parameters(), lr=lr, weight_decay=0.01)
    blind_optimizer = torch.optim.AdamW(blind_router.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    gc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gc_optimizer, T_max=total_steps)
    blind_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(blind_optimizer, T_max=total_steps)

    criterion = nn.CrossEntropyLoss()

    logger.info(f"\n{'='*60}")
    logger.info(f"GC-LoRI Router Training")
    logger.info(f"{'='*60}")
    logger.info(f"  Domains:          {domains}")
    logger.info(f"  Dataset:          {len(dataset)} examples (Train: {len(train_dataset)}, Val: {len(val_dataset)})")
    logger.info(f"  Batch size:       {batch_size}")
    logger.info(f"  Epochs:           {epochs}")
    logger.info(f"  Total steps:      {total_steps}")
    logger.info(f"  LR:               {lr}")
    logger.info(f"  Hidden dim:       {hidden_dim}")
    logger.info(f"  Output:           {output_path}")
    logger.info(f"{'='*60}\n")

    best_gc_val_acc = 0.0
    best_blind_val_acc = 0.0
    global_step = 0
    start_time = time.time()
    training_log = []

    for epoch in range(epochs):
        gc_router.train()
        blind_router.train()

        gc_epoch_loss = 0.0
        blind_epoch_loss = 0.0
        gc_correct_train = 0
        blind_correct_train = 0
        total_tokens_train = 0
        internal_signals_captured = 0

        pbar = tqdm(train_loader, desc=f"[Router] Epoch {epoch+1}/{epochs} (Train)")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            domain_labels = batch["domain_label"].to("cuda")

            hooker.clear()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hidden = outputs.hidden_states[-1].float()
            internal_signal = hooker.get_aggregated_signal(batch_size=input_ids.shape[0])

            if internal_signal is not None:
                internal_signals_captured += 1

            target_labels = domain_labels.unsqueeze(1).expand(-1, hidden.size(1))
            flat_labels = target_labels.reshape(-1)
            flat_mask = attention_mask.reshape(-1)

            # GC-LoRI
            if internal_signal is not None:
                gc_weights, aux_loss = gc_router(hidden, internal_signal, return_aux_loss=True)
                gc_logits = torch.log(gc_weights.clamp(min=1e-8))
                flat_gc_logits = gc_logits.view(-1, len(domains))

                gc_loss = criterion(flat_gc_logits[flat_mask == 1], flat_labels[flat_mask == 1])
                if aux_loss is not None:
                    gc_loss = gc_loss + aux_loss

                gc_optimizer.zero_grad()
                gc_loss.backward()
                torch.nn.utils.clip_grad_norm_(gc_router.parameters(), 1.0)
                gc_optimizer.step()
                gc_scheduler.step()

                gc_epoch_loss += gc_loss.item()
                with torch.no_grad():
                    gc_preds = flat_gc_logits.argmax(dim=-1)
                    gc_correct_train += ((gc_preds == flat_labels) & (flat_mask == 1)).sum().item()

            # Blind Router
            blind_weights, blind_logits_raw = blind_router(hidden.detach(), return_logits=True)
            flat_blind_logits = blind_logits_raw.view(-1, len(domains))

            blind_loss = criterion(flat_blind_logits[flat_mask == 1], flat_labels[flat_mask == 1])

            blind_optimizer.zero_grad()
            blind_loss.backward()
            torch.nn.utils.clip_grad_norm_(blind_router.parameters(), 1.0)
            blind_optimizer.step()
            blind_scheduler.step()

            blind_epoch_loss += blind_loss.item()
            with torch.no_grad():
                blind_preds = flat_blind_logits.argmax(dim=-1)
                blind_correct_train += ((blind_preds == flat_labels) & (flat_mask == 1)).sum().item()

            total_tokens_train += flat_mask.sum().item()
            global_step += 1

            if global_step % 10 == 0:
                gc_acc = gc_correct_train / max(total_tokens_train, 1) * 100
                blind_acc = blind_correct_train / max(total_tokens_train, 1) * 100
                pbar.set_postfix(gc_acc=f"{gc_acc:.1f}%", blind_acc=f"{blind_acc:.1f}%")

        # Validation Loop
        gc_router.eval()
        blind_router.eval()
        gc_correct_val = 0
        blind_correct_val = 0
        total_tokens_val = 0

        pbar_val = tqdm(val_loader, desc=f"[Router] Epoch {epoch+1}/{epochs} (Val)")
        for batch in pbar_val:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            domain_labels = batch["domain_label"].to("cuda")

            hooker.clear()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1].float()
                internal_signal = hooker.get_aggregated_signal(batch_size=input_ids.shape[0])

                target_labels = domain_labels.unsqueeze(1).expand(-1, hidden.size(1))
                flat_labels = target_labels.reshape(-1)
                flat_mask = attention_mask.reshape(-1)

                if internal_signal is not None:
                    gc_weights, _ = gc_router(hidden, internal_signal, return_aux_loss=False)
                    gc_preds = gc_weights.argmax(dim=-1).view(-1)
                    gc_correct_val += ((gc_preds == flat_labels) & (flat_mask == 1)).sum().item()

                blind_weights, _ = blind_router(hidden, return_logits=True)
                blind_preds = blind_weights.argmax(dim=-1).view(-1)
                blind_correct_val += ((blind_preds == flat_labels) & (flat_mask == 1)).sum().item()
                total_tokens_val += flat_mask.sum().item()

        gc_train_acc = gc_correct_train / max(total_tokens_train, 1) * 100
        blind_train_acc = blind_correct_train / max(total_tokens_train, 1) * 100
        gc_val_acc = gc_correct_val / max(total_tokens_val, 1) * 100
        blind_val_acc = blind_correct_val / max(total_tokens_val, 1) * 100

        logger.info(f"\n[Epoch {epoch+1} Train] GC: {gc_train_acc:.1f}%, Blind: {blind_train_acc:.1f}%")
        logger.info(f"[Epoch {epoch+1} Val]   GC: {gc_val_acc:.1f}%, Blind: {blind_val_acc:.1f}%")

        training_log.append({
            "epoch": epoch + 1,
            "gc_train_acc": gc_train_acc,
            "blind_train_acc": blind_train_acc,
            "gc_val_acc": gc_val_acc,
            "blind_val_acc": blind_val_acc,
        })

        # Save best GC-LoRI based on Validation Acc
        if gc_val_acc > best_gc_val_acc:
            best_gc_val_acc = gc_val_acc
            save_path = output_path / "best"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": gc_router.state_dict(),
                "config": gc_router.get_config(),
                "epoch": epoch + 1,
                "val_accuracy": gc_val_acc,
                "domains": domains,
            }, save_path / "gc_router.pt")
            logger.info(f"  New best GC-LoRI Val Acc: {gc_val_acc:.1f}%! Saved.")

        # Save best Blind based on Validation Acc
        if blind_val_acc > best_blind_val_acc:
            best_blind_val_acc = blind_val_acc
            save_path = output_path / "blind"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": blind_router.state_dict(),
                "config": {"hidden_dim": hidden_dim, "num_experts": len(domains), "bottleneck_dim": bottleneck_dim // 2},
                "epoch": epoch + 1,
                "val_accuracy": blind_val_acc,
                "domains": domains,
            }, save_path / "blind_router.pt")
            logger.info(f"  New best Blind Val Acc: {blind_val_acc:.1f}%! Saved.")

    # Save training log
    elapsed = (time.time() - start_time) / 60
    log_data = {
        "best_gc_val_acc": best_gc_val_acc,
        "best_blind_val_acc": best_blind_val_acc,
        "delta_best_val": best_gc_val_acc - best_blind_val_acc,
        "total_time_min": elapsed,
        "training_log": training_log,
    }
    with open(output_path / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Cleanup
    hooker.remove()
    del model
    torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"Router training complete")
    logger.info(f"  Best GC-LoRI Val Acc:  {best_gc_val_acc:.1f}%")
    logger.info(f"  Best Blind Val Acc:    {best_blind_val_acc:.1f}%")
    logger.info(f"  Δ(GC - Blind):         {best_gc_val_acc - best_blind_val_acc:+.1f}%")
    logger.info(f"  Time:                  {elapsed:.1f} min")
    logger.info(f"  Output:                {output_path}")
    logger.info(f"{'='*60}\n")

    if best_gc_val_acc > best_blind_val_acc + 2.0:
        logger.info("✅ GC-LoRI shows meaningful improvement over blind routing!")
    elif best_gc_val_acc > best_blind_val_acc:
        logger.info("⚠️  Small GC-LoRI improvement. May not be significant.")
    else:
        logger.info("❌ GC-LoRI did NOT outperform blind routing. Internal signals may not help.")

    return log_data


def main():
    parser = argparse.ArgumentParser(description="Train GC-LoRI Router")
    parser.add_argument("--base_model", type=str, default=str(PROJECT_ROOT / "models" / "nemotron"))
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "nemotron"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "adapters" / "nemotron_30b" / "gc_router"))
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples_per_domain", type=int, default=2000)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--internal_top_k", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--router_top_k", type=int, default=2)
    args = parser.parse_args()

    Path(str(PROJECT_ROOT / "logs" / "nemotron")).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(PROJECT_ROOT / "logs" / "nemotron" / "train_gc_router.log")),
        ],
    )

    train_gc_router(
        base_model_name=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        domains=args.domains,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        max_samples_per_domain=args.max_samples_per_domain,
        use_4bit=args.use_4bit,
        internal_top_k=args.internal_top_k,
        bottleneck_dim=args.bottleneck_dim,
        router_top_k=args.router_top_k,
    )


if __name__ == "__main__":
    main()
