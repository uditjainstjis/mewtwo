"""
Instrumented LoRA Trainer — Synapta v2.0

This is the core of the grokking study. It trains LoRA adapters while logging
dense internal metrics at every checkpoint:
  - SVD spectrum of adapter weight matrices (ΔW = B @ A)
  - Effective rank (exponential of Shannon entropy of normalized singular values)
  - Weight Frobenius norm
  - Gradient norm
  - Training & validation loss
  - Optional: domain MMLU accuracy at periodic intervals

The goal: detect grokking-like phase transitions in LoRA adapter training.
"""

import os
import json
import math
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset


@dataclass
class AdapterCheckpointMetrics:
    """Metrics captured at each checkpoint during adapter training."""
    step: int
    timestamp: float
    train_loss: float
    eval_loss: float
    learning_rate: float
    grad_norm: float

    # SVD metrics per layer (aggregated across all target modules)
    svd_spectra: Dict[str, List[float]]       # layer_name -> list of singular values
    effective_ranks: Dict[str, float]          # layer_name -> effective rank
    frobenius_norms: Dict[str, float]          # layer_name -> ||ΔW||_F
    weight_cosine_to_prev: Dict[str, float]    # layer_name -> cosine sim to previous checkpoint

    # Aggregate metrics
    mean_effective_rank: float
    total_frobenius_norm: float
    mean_weight_drift: float                   # avg cosine distance from previous step

    # Optional domain accuracy
    domain_mmlu_accuracy: Optional[float] = None


def compute_svd_metrics(model, prev_weights: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """
    Compute SVD-based metrics for all LoRA adapter weight matrices.
    
    For each LoRA layer, computes ΔW = B @ A and then:
      - Full SVD spectrum σ₁ ≥ σ₂ ≥ ... ≥ σ_r
      - Effective rank r_eff = exp(H(σ̂)) where σ̂ = σ/sum(σ) and H is Shannon entropy
      - Frobenius norm ||ΔW||_F
      - Cosine similarity to previous checkpoint (weight drift rate)
    
    Returns:
        metrics: dict of per-layer metrics
        current_weights: dict of current weight matrices (for next comparison)
    """
    metrics = {
        'svd_spectra': {},
        'effective_ranks': {},
        'frobenius_norms': {},
        'weight_cosine_to_prev': {},
    }
    current_weights = {}
    
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            continue
    
    # Collect paired A/B matrices
    lora_pairs = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            base_name = name.replace('.lora_A.default.weight', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['A'] = param.detach().cpu().float()
        elif 'lora_B' in name:
            base_name = name.replace('.lora_B.default.weight', '')
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]['B'] = param.detach().cpu().float()
    
    effective_ranks_list = []
    frobenius_list = []
    cosine_list = []
    
    for layer_name, pair in lora_pairs.items():
        if 'A' not in pair or 'B' not in pair:
            continue
        
        # ΔW = B @ A
        delta_w = pair['B'] @ pair['A']
        current_weights[layer_name] = delta_w.clone()
        
        # SVD
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        singular_values = S.numpy().tolist()
        
        # Effective rank via Shannon entropy of normalized singular values
        s_normalized = S / (S.sum() + 1e-12)
        s_normalized = s_normalized[s_normalized > 1e-12]  # filter zeros
        entropy = -(s_normalized * torch.log(s_normalized)).sum().item()
        eff_rank = math.exp(entropy)
        
        # Frobenius norm
        frob_norm = torch.norm(delta_w, p='fro').item()
        
        # Cosine similarity to previous checkpoint
        cosine_sim = 0.0
        if prev_weights and layer_name in prev_weights:
            prev = prev_weights[layer_name].flatten()
            curr = delta_w.flatten()
            cosine_sim = (torch.dot(prev, curr) / (torch.norm(prev) * torch.norm(curr) + 1e-12)).item()
        
        metrics['svd_spectra'][layer_name] = singular_values
        metrics['effective_ranks'][layer_name] = eff_rank
        metrics['frobenius_norms'][layer_name] = frob_norm
        metrics['weight_cosine_to_prev'][layer_name] = cosine_sim
        
        effective_ranks_list.append(eff_rank)
        frobenius_list.append(frob_norm)
        cosine_list.append(cosine_sim)
    
    metrics['mean_effective_rank'] = np.mean(effective_ranks_list) if effective_ranks_list else 0.0
    metrics['total_frobenius_norm'] = sum(frobenius_list)
    metrics['mean_weight_drift'] = np.mean(cosine_list) if cosine_list else 0.0
    
    return metrics, current_weights


class InstrumentedLoRATrainer:
    """
    Trains a LoRA adapter with dense checkpoint instrumentation.
    
    At every `checkpoint_every` steps:
      1. Saves the full adapter weights
      2. Computes SVD spectrum of all ΔW = B @ A matrices
      3. Logs effective rank, Frobenius norm, weight drift
      4. Periodically evaluates domain MMLU accuracy
    
    This produces the training dynamics data needed for the grokking analysis.
    """
    
    def __init__(self,
                 base_model_id: str,
                 train_dataset: Dataset,
                 eval_dataset: Dataset,
                 domain_name: str,
                 lora_rank: int = 32,
                 lora_alpha: Optional[int] = None,
                 target_modules: Optional[List[str]] = None,
                 weight_decay: float = 0.01,
                 learning_rate: float = 2e-4,
                 max_steps: int = 2000,
                 batch_size: int = 4,
                 gradient_accumulation: int = 4,
                 checkpoint_every: int = 25,
                 svd_every: int = 25,
                 mmlu_every: int = 100,
                 output_dir: str = "./adapters",
                 wandb_project: str = "synapta-v2",
                 dtype: torch.dtype = torch.bfloat16):
        
        self.base_model_id = base_model_id
        self.domain_name = domain_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha or (2 * lora_rank)
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_every = checkpoint_every
        self.svd_every = svd_every
        self.mmlu_every = mmlu_every
        self.dtype = dtype
        
        # Output directory structure
        model_short = base_model_id.split("/")[-1]
        self.run_name = f"{domain_name}_r{lora_rank}_wd{weight_decay}_{model_short}"
        self.output_dir = Path(output_dir) / self.run_name
        self.metrics_dir = self.output_dir / "dynamics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Will be populated during training
        self.all_metrics: List[Dict] = []
        self.prev_weights: Optional[Dict] = None
    
    def _load_model_and_tokenizer(self):
        """Load base model with LoRA config."""
        print(f"Loading {self.base_model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        
        self.model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _create_training_args(self) -> TrainingArguments:
        """Build HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=str(self.output_dir / "hf_checkpoints"),
            num_train_epochs=999,  # We use max_steps instead
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=self.weight_decay,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=self.checkpoint_every,
            save_strategy="steps",
            save_steps=self.checkpoint_every,
            save_total_limit=None,  # Keep ALL checkpoints for dynamics analysis
            report_to="wandb",
            run_name=self.run_name,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
    
    def _instrument_checkpoint(self, step: int, train_loss: float, eval_loss: float,
                                learning_rate: float, grad_norm: float):
        """
        Capture dense metrics at this checkpoint.
        This is the heart of the grokking study.
        """
        svd_metrics, current_weights = compute_svd_metrics(self.model, self.prev_weights)
        self.prev_weights = current_weights
        
        checkpoint_data = {
            'step': step,
            'timestamp': time.time(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': learning_rate,
            'grad_norm': grad_norm,
            'mean_effective_rank': svd_metrics['mean_effective_rank'],
            'total_frobenius_norm': svd_metrics['total_frobenius_norm'],
            'mean_weight_drift': svd_metrics['mean_weight_drift'],
            'effective_ranks': svd_metrics['effective_ranks'],
            'frobenius_norms': svd_metrics['frobenius_norms'],
            'weight_cosine_to_prev': svd_metrics['weight_cosine_to_prev'],
            # We store spectra separately as they can be large
        }
        
        self.all_metrics.append(checkpoint_data)
        
        # Save spectra to separate file (can be large)
        spectra_path = self.metrics_dir / f"svd_spectra_step{step:06d}.json"
        with open(spectra_path, 'w') as f:
            json.dump(svd_metrics['svd_spectra'], f)
        
        # Save running metrics
        metrics_path = self.metrics_dir / "training_dynamics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.all_metrics, f, indent=2, default=str)
        
        print(f"  [Step {step}] train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} "
              f"eff_rank={svd_metrics['mean_effective_rank']:.2f} "
              f"||ΔW||_F={svd_metrics['total_frobenius_norm']:.4f}")
    
    def train(self):
        """
        Execute the full instrumented training run.
        """
        print(f"\n{'='*60}")
        print(f"  INSTRUMENTED TRAINING: {self.run_name}")
        print(f"  Domain: {self.domain_name}")
        print(f"  Rank: {self.lora_rank}, Alpha: {self.lora_alpha}")
        print(f"  Weight Decay: {self.weight_decay}")
        print(f"  Max Steps: {self.max_steps}")
        print(f"{'='*60}\n")
        
        self._load_model_and_tokenizer()
        training_args = self._create_training_args()
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True
        )
        
        # Custom callback for instrumented checkpointing
        from transformers import TrainerCallback
        
        trainer_ref = [None]  # Mutable reference for callback
        parent = self
        
        class InstrumentationCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if state.global_step % parent.svd_every == 0:
                    train_loss = state.log_history[-1].get('loss', 0) if state.log_history else 0
                    eval_loss = metrics.get('eval_loss', 0) if metrics else 0
                    lr = state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
                    grad_norm = state.log_history[-1].get('grad_norm', 0) if state.log_history else 0
                    parent._instrument_checkpoint(
                        state.global_step, train_loss, eval_loss, lr, grad_norm
                    )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[InstrumentationCallback()],
        )
        trainer_ref[0] = trainer
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final adapter
        final_path = self.output_dir / "final_adapter"
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        # Save complete training config
        config = {
            'base_model_id': self.base_model_id,
            'domain_name': self.domain_name,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'target_modules': self.target_modules,
            'weight_decay': self.weight_decay,
            'learning_rate': self.learning_rate,
            'max_steps': self.max_steps,
            'batch_size': self.batch_size,
            'gradient_accumulation': self.gradient_accumulation,
        }
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Training complete. Adapter saved to {final_path}")
        print(f"   Dynamics data: {self.metrics_dir}")
        print(f"   Total checkpoints: {len(self.all_metrics)}")
        
        return self.all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Instrumented LoRA Trainer")
    parser.add_argument("--config", type=str, default="configs/training_v2.yaml")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./adapters_v2")
    args = parser.parse_args()
    
    # Load training data
    train_ds = load_dataset("json", data_files=f"{args.data_dir}/train.jsonl", split="train")
    eval_ds = load_dataset("json", data_files=f"{args.data_dir}/val.jsonl", split="train")
    
    trainer = InstrumentedLoRATrainer(
        base_model_id=args.model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        domain_name=args.domain,
        lora_rank=args.rank,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
    )
    trainer.train()
