import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

"""
LoRI Adapter Training Script

Trains a single domain LoRA adapter using the LoRI methodology:
  - Shared frozen B matrix (random Gaussian projection)
  - Trainable sparse A matrix (domain-specific)
  - Standard causal LM loss on domain data

This script trains ONE domain at a time. Run it 5 times (once per domain)
or use train_all_domains.sh for sequential orchestration.

Usage:
    python -m src.lori_moe.training.train_lori_adapter \
        --domain math \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --rank 32 \
        --sparsity 0.8 \
        --epochs 3 \
        --batch_size 8 \
        --lr 2e-4
"""
import sys
import json
import time
import signal
import argparse
import logging
import math
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

# ─── Training Dataset ──────────────────────────────────────────────────────────


class DomainDataset(TorchDataset):
    """Loads pre-processed JSONL domain training data."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.encoded_examples = []
        self.assistant_prefix = self._derive_assistant_prefix()
        self._mask_warning_emitted = False

        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                row = json.loads(line.strip())
                self.examples.append(row["text"])

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
        if self.assistant_prefix:
            logger.info(
                "Derived assistant prefix from chat template for loss masking: %r",
                self.assistant_prefix,
            )
        else:
            logger.warning(
                "Could not derive assistant prefix from tokenizer chat template. "
                "Falling back to token-pattern masking."
            )

        # Pretokenize once so the GPU is not starved by repeated CPU tokenization.
        batch_size = 512
        for start in range(0, len(self.examples), batch_size):
            texts = self.examples[start:start + batch_size]
            encoded_batch = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )
            for text, input_ids, attention_mask in zip(
                texts, encoded_batch["input_ids"], encoded_batch["attention_mask"]
            ):
                input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
                attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
                labels = input_ids_tensor.clone()

                mask_until = self._find_assistant_start(text, input_ids_tensor)
                if mask_until is None:
                    mask_until = 0
                else:
                    mask_until = max(0, min(mask_until, input_ids_tensor.shape[0]))
                    labels[:mask_until] = -100

                self.encoded_examples.append(
                    {
                        "input_ids": input_ids_tensor,
                        "attention_mask": attention_mask_tensor,
                        "labels": labels,
                    }
                )

    def _derive_assistant_prefix(self) -> Optional[str]:
        """Infer the exact assistant-content prefix from the tokenizer chat template."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return None

        system = "__system__"
        user = "__user__"
        sentinel = "__assistant_sentinel__"

        try:
            prompt_only = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            with_assistant = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": sentinel},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            logger.warning("Chat template inspection failed: %s", exc)
            return None

        sentinel_start = with_assistant.find(sentinel)
        if sentinel_start == -1 or not with_assistant.startswith(prompt_only):
            return None

        prefix = with_assistant[len(prompt_only):sentinel_start]
        return prefix or None

    def _find_assistant_start(self, text: str, input_ids: torch.Tensor) -> Optional[int]:
        """Find where assistant response begins, using the tokenizer's own template when possible."""
        if self.assistant_prefix:
            prefix_pos = text.find(self.assistant_prefix)
            if prefix_pos != -1:
                prefix_text = text[:prefix_pos + len(self.assistant_prefix)]
                prefix_ids = self.tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_attention_mask=False,
                )["input_ids"]
                return min(len(prefix_ids), input_ids.shape[0])

        # Fallback: preserve old behavior for Qwen-like templates if dynamic detection fails.
        assistant_start_seq = [151644, 77091, 198]
        for i in range(len(input_ids) - len(assistant_start_seq)):
            if input_ids[i:i + len(assistant_start_seq)].tolist() == assistant_start_seq:
                return i + len(assistant_start_seq)

        if not self._mask_warning_emitted:
            logger.warning(
                "Could not locate assistant span in one or more examples. "
                "Those examples will keep full-token loss."
            )
            self._mask_warning_emitted = True
        return None

    def __len__(self):
        return len(self.encoded_examples)

    def __getitem__(self, idx):
        encoded = self.encoded_examples[idx]
        input_ids = encoded["input_ids"].clone()
        attention_mask = encoded["attention_mask"].clone()
        labels = encoded["labels"].clone()
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DynamicPaddingCollator:
    """Pad each batch to the longest sequence instead of max_length every time."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(feature["input_ids"].shape[0] for feature in features)
        if self.pad_to_multiple_of:
            max_len = int(
                math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )

        pad_token_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            seq_len = feature["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids.append(
                torch.nn.functional.pad(
                    feature["input_ids"], (0, pad_len), value=pad_token_id
                )
            )
            attention_mask.append(
                torch.nn.functional.pad(
                    feature["attention_mask"], (0, pad_len), value=0
                )
            )
            labels.append(
                torch.nn.functional.pad(
                    feature["labels"], (0, pad_len), value=-100
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


# ─── LoRI-specific PEFT wrapper ────────────────────────────────────────────────


def create_lori_model(
    base_model_name: str,
    rank: int = 32,
    alpha: float = 64.0,
    target_modules: list = None,
    shared_b_seed: int = 42,
    device: str = "cuda",
    use_gradient_checkpointing: bool = False,
    use_4bit: bool = False,
):
    """
    Load base model and apply LoRA with LoRI modifications.

    Strategy: We use HuggingFace PEFT's standard LoRA, then post-hoc:
      1. Replace each LoRA down-projection with the shared frozen projection
      2. Keep the up-projection trainable as the domain-specific LoRI factor
    
    This gives us PEFT compatibility while enforcing LoRI constraints.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] # No MoE gate/router targets per user instruction

    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        logger.info("Loading base model in 4-bit precision (BitsAndBytes)...")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **kwargs
    )
    
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    

    # Apply LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    if use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing (non-reentrant for LoRI compatibility).")
        model.enable_input_require_grads()  # Must come AFTER get_peft_model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False  # Mutually exclusive with checkpointing

    # ─── LoRI Modification: Freeze B matrices and share them ───────────────
    logger.info("Applying LoRI constraints: freezing B, sharing projection...")

    # Generate the shared B values from deterministic seed
    generator = torch.Generator(device="cpu")
    generator.manual_seed(shared_b_seed)

    lori_stats = {"modules_modified": 0, "b_frozen": 0}

    for name, module in model.named_modules():
        # In PEFT, lora_A is the down-projection (rank x in_features)
        # and lora_B is the up-projection (out_features x rank).
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Get the actual linear layers from the module dict
            for adapter_name in module.lora_A:
                lora_a = module.lora_A[adapter_name]
                lora_b = module.lora_B[adapter_name]

                # Freeze the shared random down-projection on lora_A.
                with torch.no_grad():
                    # LoRI scaling fix: use in_features (shape[1]) for variance preservation
                    # instead of rank. Original rank-based scaling was ~32x too volatile.
                    in_features = lora_a.weight.shape[1]
                    shared_b = torch.randn(
                        lora_a.weight.shape,
                        generator=generator,
                        dtype=torch.bfloat16,
                    ) / (in_features ** 0.5)
                    lora_a.weight.copy_(shared_b)
                lora_a.weight.requires_grad = False
                lori_stats["b_frozen"] += 1

                # The domain-specific up-projection stays trainable.
                lora_b.weight.requires_grad = True
                lori_stats["modules_modified"] += 1

    logger.info(
        f"LoRI applied: {lori_stats['modules_modified']} modules modified, "
        f"{lori_stats['b_frozen']} B matrices frozen and shared"
    )

    # Free up memory during backward pass (already handled conditionally above)
    # model.gradient_checkpointing_enable()

    # Print trainable params
    model.print_trainable_parameters()

    return model, tokenizer


# ─── Training Loop ─────────────────────────────────────────────────────────────


def train_adapter(
    domain: str,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    data_dir: str = "/home/learner/Desktop/mewtwo/data/lori_moe",
    output_dir: str = "/home/learner/Desktop/mewtwo/checkpoints/lori_moe/adapters",
    rank: int = 32,
    alpha: float = 64.0,
    sparsity: float = 0.8,
    shared_b_seed: int = 42,
    lr: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_seq_length: int = 1024,
    max_train_samples: Optional[int] = None,
    warmup_ratio: float = 0.1,
    log_every: int = 10,
    save_every: int = 500,
    gradient_checkpointing: bool = False,
    optimizer_backend: str = "auto",
    compile_mode: Optional[str] = None,
    vram_fraction: float = None,
    use_4bit: bool = False,
    target_modules: str = "",
):
    """Train a single domain LoRI adapter."""

    output_path = Path(output_dir) / domain
    output_path.mkdir(parents=True, exist_ok=True)
    interrupted = False
    interrupt_signal = None

    model, tokenizer = create_lori_model(
        base_model_name=base_model_name,
        rank=rank,
        alpha=alpha,
        shared_b_seed=shared_b_seed,
        device="cuda",
        use_gradient_checkpointing=gradient_checkpointing,
        use_4bit=use_4bit,
        target_modules=target_modules.split(",") if target_modules else None,
    )

    # ── Load dataset ──
    data_path = Path(data_dir) / f"{domain}_train.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run: python -m src.lori_moe.data.prepare_datasets first!"
        )

    dataset = DomainDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_samples=max_train_samples,
    )

    num_workers = min(8, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DynamicPaddingCollator(tokenizer),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )

    # ── Optimizer & Scheduler ──
    # Only optimize A matrices (B is frozen by LoRI constraint)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_backend = (optimizer_backend or "auto").lower()
    optimizer_name = "torch_adamw_fused"
    if optimizer_backend in {"auto", "paged_adamw_8bit"} and bnb is not None:
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        optimizer_name = "bnb_paged_adamw_8bit"
    elif optimizer_backend == "paged_adamw_32bit" and bnb is not None:
        optimizer = bnb.optim.PagedAdamW32bit(
            trainable_params,
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        optimizer_name = "bnb_paged_adamw_32bit"
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available(),
        )

    total_steps = max(1, (len(dataloader) * epochs) // grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Training LoRI adapter: {domain}")
    logger.info(f"{'='*60}")
    logger.info(f"  Dataset:       {len(dataset)} examples")
    logger.info(f"  Batch size:    {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Total steps:   {total_steps}")
    logger.info(f"  Warmup steps:  {warmup_steps}")
    logger.info(f"  LR:            {lr}")
    logger.info(f"  Rank:          {rank}")
    logger.info(f"  Sparsity:      {sparsity}")
    logger.info(f"  Seq length:    {max_seq_length}")
    logger.info(f"  Optimizer:     {optimizer_name}")
    logger.info(f"  Output:        {output_path}")
    logger.info(f"{'='*60}\n")

    model.train()
    train_model = model
    global_step = 0
    total_loss = 0
    best_loss = float("inf")
    start_time = time.time()
    initial_global_step = 0
    training_log = []

    def persist_training_snapshot(
        save_path: Path,
        interrupted_run: bool,
        save_tokenizer: bool = False,
        include_full_log: bool = False,
    ) -> None:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        if save_tokenizer:
            tokenizer.save_pretrained(str(save_path))
        with open(save_path / "training_state.json", "w") as f:
            json.dump({
                "domain": domain,
                "base_model": base_model_name,
                "global_step": global_step,
                "best_loss": best_loss,
                "interrupted": interrupted_run,
                "signal": interrupt_signal,
                "config": {
                    "rank": rank,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "shared_b_seed": shared_b_seed,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "max_seq_length": max_seq_length,
                    "max_train_samples": max_train_samples,
                    "warmup_ratio": warmup_ratio,
                    "save_every": save_every,
                    "gradient_checkpointing": gradient_checkpointing,
                    "optimizer_backend": optimizer_name,
                    "compile_mode": compile_mode,
                },
                "log": training_log if include_full_log else [],
            }, f, indent=2)

    def handle_interrupt(signum, _frame):
        nonlocal interrupted, interrupt_signal
        interrupted = True
        interrupt_signal = signal.Signals(signum).name
        logger.warning(
            "Received %s. Training will stop after the current optimizer step and save an interrupt checkpoint.",
            interrupt_signal,
        )

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    def emergency_save(signum, frame):
        logger.info(f"\n[EMERGENCY SAVE] SIGUSR1 received at step {global_step}. Saving checkpoint...")
        emergency_path = output_path / f"emergency-manual-step-{global_step}"
        persist_training_snapshot(
            emergency_path,
            interrupted_run=False,
            save_tokenizer=True,
            include_full_log=False,
        )
        logger.info(f"[EMERGENCY SAVE] Done. Training continues.")

    signal.signal(signal.SIGUSR1, emergency_save)

    # ── Auto-Resume Logic ──
    latest_ckpt = None
    max_step_found = -1
    
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir() and (item.name.startswith("checkpoint-") or item.name.startswith("emergency-step-")):
                try:
                    step_val = int(item.name.split("-")[-1])
                    if step_val > max_step_found:
                        max_step_found = step_val
                        latest_ckpt = item
                except ValueError:
                    pass

    if latest_ckpt:
        logger.info(f"\n{'*'*60}")
        logger.info(f"⚡ AUTO-RESUME TRIGGERED ⚡")
        logger.info(f"Found latest checkpoint: {latest_ckpt.name}. Resuming from global_step {max_step_found}.")
        
        # Load weights
        from peft import PeftModel
        # model is already a PeftModel, load_adapter updates weights in-place
        model.load_adapter(str(latest_ckpt), "default", is_trainable=True)
        
        # Load state variables
        state_file = latest_ckpt / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state_data = json.load(f)
                global_step = state_data.get("global_step", max_step_found)
                best_loss = state_data.get("best_loss", float('inf'))
                logger.info(f"Restoring state -> Step: {global_step}, Best Loss: {best_loss:.4f}")
        else:
            global_step = max_step_found
            
        # Fast-forward scheduler
        for _ in range(global_step):
            scheduler.step()
            
        logger.info(f"{'*'*60}\n")
    else:
        logger.info("No checkpoint found. Starting from Step 0.")
    
    initial_global_step = global_step

    train_model = model
    if compile_mode:
        try:
            train_model = torch.compile(model, mode=compile_mode)
            logger.info("Compiled training graph enabled (mode=%s).", compile_mode)
        except Exception as exc:
            train_model = model
            logger.warning("torch.compile unavailable, falling back to eager mode: %s", exc)

    try:
        # Determine where to start dynamically
        start_epoch = (global_step * grad_accum) // len(dataloader)
        steps_to_skip_in_first_epoch = (global_step * grad_accum) % len(dataloader)
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0
            epoch_steps = 0
            pbar = tqdm(
                dataloader,
                desc=f"[{domain}] Epoch {epoch+1}/{epochs}",
                dynamic_ncols=True,
            )
            
            for step, batch in enumerate(pbar):
                # Fast-forward dataloader logic (only applies to resumes)
                if epoch == start_epoch and step < steps_to_skip_in_first_epoch:
                    continue
                # Move to GPU
                input_ids = batch["input_ids"].to("cuda", non_blocking=True)
                attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
                labels = batch["labels"].to("cuda", non_blocking=True)

                # Forward pass
                outputs = train_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss / grad_accum

                # Backward — wrapped to catch OOM and save before dying
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM during backward! Emergency saving at step {global_step}...")
                        torch.cuda.empty_cache()
                        emergency_path = output_path / f"emergency-step-{global_step}"
                        persist_training_snapshot(
                            emergency_path,
                            interrupted_run=True,
                            save_tokenizer=False,
                            include_full_log=False,
                        )
                        logger.error(f"Emergency checkpoint saved: {emergency_path}")
                    raise

                epoch_loss += outputs.loss.item()
                total_loss += outputs.loss.item()
                epoch_steps += 1

                # Optimizer step (with gradient accumulation)
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Logging
                    if global_step % log_every == 0:
                        avg_loss = epoch_loss / epoch_steps
                        elapsed = time.time() - start_time
                        steps_completed = global_step - initial_global_step + 1e-6
                        eta = (elapsed / steps_completed) * (total_steps - global_step)
                        gpu_mem = torch.cuda.memory_allocated() / 1e9
                        lr_current = scheduler.get_last_lr()[0]

                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr_current:.2e}",
                            "gpu": f"{gpu_mem:.1f}G",
                            "eta": f"{eta/60:.0f}m",
                        })

                        log_entry = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": avg_loss,
                            "lr": lr_current,
                            "gpu_gb": gpu_mem,
                            "elapsed_min": elapsed / 60,
                        }
                        training_log.append(log_entry)
                        
                        # Physically log to the file so it can be `tail -f` tracked
                        example_text = tokenizer.decode(input_ids[0][:100], skip_special_tokens=True).replace("\n", " ") + "..."
                        logger.info(
                            f"[{domain}] Epoch {epoch+1}/{epochs} | "
                            f"Step {step+1}/{len(dataloader)} ({global_step}) | "
                            f"Loss: {avg_loss:.4f} | LR: {lr_current:.2e} | "
                            f"GPU: {gpu_mem:.1f}GB | ETA: {eta/60:.0f}m"
                        )
                        logger.info(f"Example: {example_text}")

                    # Checkpoint
                    if global_step % save_every == 0:
                        ckpt_path = output_path / f"checkpoint-{global_step}"
                        persist_training_snapshot(
                            ckpt_path,
                            interrupted_run=False,
                            save_tokenizer=False,
                            include_full_log=False,
                        )
                        logger.info(f"Checkpoint saved: {ckpt_path}")
                        with open(output_path / ".latest_checkpoint", "w") as f:
                            f.write(ckpt_path.name)

                    if interrupted:
                        interrupt_path = output_path / f"interrupt-step-{global_step}"
                        persist_training_snapshot(
                            interrupt_path,
                            interrupted_run=True,
                            save_tokenizer=True,
                            include_full_log=True,
                        )
                        logger.warning(f"Interrupt checkpoint saved: {interrupt_path}")
                        import sys
                        sys.exit(130)

            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                persist_training_snapshot(
                    output_path / "best",
                    interrupted_run=False,
                    save_tokenizer=True,
                    include_full_log=False,
                )
                logger.info(f"New best model saved (loss={best_loss:.4f})")

        # ── Save final model ──
        persist_training_snapshot(
            output_path / "final",
            interrupted_run=False,
            save_tokenizer=True,
            include_full_log=True,
        )

        # ── Apply DARE sparsification post-training ──
        logger.info(f"\nApplying DARE sparsification (drop_rate={1-sparsity:.0%})...")
        apply_dare_to_peft_model(model, drop_rate=1-sparsity)
        persist_training_snapshot(
            output_path / "dare_sparsified",
            interrupted_run=False,
            save_tokenizer=True,
            include_full_log=False,
        )

    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

        # ── Save training log ──
        log_file = output_path / "training_log.json"
        with open(log_file, "w") as f:
            json.dump({
                "domain": domain,
                "total_steps": global_step,
                "best_loss": best_loss,
                "total_time_min": (time.time() - start_time) / 60,
                "interrupted": interrupted,
                "signal": interrupt_signal,
                "config": {
                    "rank": rank,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "optimizer_backend": optimizer_name,
                    "compile_mode": compile_mode,
                },
                "log": training_log,
            }, f, indent=2)

    total_time = (time.time() - start_time) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete: {domain}")
    logger.info(f"  Best loss:     {best_loss:.4f}")
    logger.info(f"  Total time:    {total_time:.1f} minutes")
    logger.info(f"  Output:        {output_path}")
    logger.info(f"{'='*60}\n")

    return {"best_loss": best_loss, "total_time_min": total_time}


def apply_dare_to_peft_model(model, drop_rate: float = 0.2):
    """
    Apply DARE (Drop And Rescale) to the trainable PEFT up-projection matrices.
    
    Drops the lowest-magnitude parameters and rescales the remainder
    to preserve expected output magnitude.
    """
    stats = {"total_dropped": 0, "total_params": 0}

    for name, module in model.named_modules():
        if hasattr(module, 'lora_B'):
            for adapter_name in module.lora_B:
                lora_b = module.lora_B[adapter_name]
                if lora_b.weight.requires_grad:
                    with torch.no_grad():
                        w = lora_b.weight.data
                        total = w.numel()
                        
                        # Find threshold for drop_rate
                        threshold = torch.quantile(w.abs().flatten().float(), drop_rate)
                        
                        # Create mask and apply
                        mask = (w.abs() >= threshold).float()
                        dropped = total - mask.sum().item()
                        
                        # Rescale to preserve magnitude
                        scale = total / (mask.sum() + 1e-8)
                        lora_b.weight.data = w * mask * scale

                        stats["total_dropped"] += dropped
                        stats["total_params"] += total

    drop_pct = 100 * stats["total_dropped"] / max(stats["total_params"], 1)
    logger.info(
        f"DARE applied: dropped {stats['total_dropped']:,.0f} / "
        f"{stats['total_params']:,.0f} params ({drop_pct:.1f}%)"
    )


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a single LoRI domain adapter")
    parser.add_argument("--domain", type=str, required=True,
                       choices=["math", "code", "science", "legal", "medical"])
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_dir", type=str, default="/home/learner/Desktop/mewtwo/data/lori_moe")
    parser.add_argument("--output_dir", type=str, default="/home/learner/Desktop/mewtwo/checkpoints/lori_moe/adapters")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--shared_b_seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (saves memory, slower)")
    parser.add_argument("--vram_fraction", type=float, default=None, help="Hard cap VRAM usage (0.0 to 1.0) to prevent OOMing other processes")
    parser.add_argument("--target_modules", type=str, default="", help="Comma separated list of target modules")
    parser.add_argument(
        "--optimizer_backend",
        type=str,
        default="auto",
        choices=["auto", "paged_adamw_8bit", "paged_adamw_32bit", "adamw"],
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--use_4bit", action="store_true", help="Load base model in 4-bit precision to fit huge models (>14B) on standard GPUs.")
    args = parser.parse_args()

    # Move global torch settings out of main logic to avoid scoping issues
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.vram_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, device=0)
        print(f"HARDWARE VRAM CAP SET: Process physically limited to {args.vram_fraction*100}% of GPU.")

    # Ensure log dir exists
    Path("/home/learner/Desktop/mewtwo/logs/lori_moe").mkdir(parents=True, exist_ok=True)

    log_model_name = args.base_model.split("/")[-1].replace("-", "_").lower()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"/home/learner/Desktop/mewtwo/logs/lori_moe/train_{log_model_name}_{args.domain}.log"
            ),
        ],
    )

    train_adapter(
        domain=args.domain,
        base_model_name=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        sparsity=args.sparsity,
        shared_b_seed=args.shared_b_seed,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        log_every=args.log_every,
        save_every=args.save_every,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer_backend=args.optimizer_backend,
        compile_mode=args.compile_mode,
        vram_fraction=args.vram_fraction,
        use_4bit=args.use_4bit,
        target_modules=args.target_modules,
    )


if __name__ == "__main__":
    main()
