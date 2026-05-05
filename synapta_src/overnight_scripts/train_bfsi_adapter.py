#!/usr/bin/env python3
"""Train a BFSI compliance LoRA adapter on Nemotron-30B.

Configuration matches existing math/code/science adapters exactly so it slots
into the routing infrastructure unchanged:
  - Rank 64
  - lora_alpha 128
  - lora_dropout 0.05
  - target_modules: q_proj, v_proj, o_proj
  - 4-bit NF4 quantization on base
  - SFT with paged_adamw_8bit
  - LR 5e-5 (lower than 1e-4 default — 525 examples, want stable convergence)

Output: adapters/nemotron_30b/bfsi/best/
"""
import os, sys, json, time, datetime
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

MODEL_PATH = str(PROJECT / "models" / "nemotron")
TRAIN_DATA = PROJECT / "data" / "rbi_circulars" / "bfsi_train.jsonl"
OUT_DIR = PROJECT / "adapters" / "nemotron_30b" / "bfsi"
LOG_FILE = PROJECT / "logs" / "swarm_8h" / "extras" / "bfsi_train.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    log("=== BFSI adapter training on Nemotron-30B ===")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load training data
    log("Loading training data...")
    examples = []
    with open(TRAIN_DATA) as f:
        for line in f:
            examples.append(json.loads(line))
    log(f"Loaded {len(examples)} training examples")

    # Format as chat messages
    def format_example(ex):
        msgs = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return text

    texts = [format_example(ex) for ex in examples]
    log(f"Sample formatted text (first 600 chars):\n{texts[0][:600]}")

    # Tokenize with truncation/padding
    MAX_LEN = 512

    def tokenize(text):
        out = tok(text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors=None)
        out["labels"] = out["input_ids"].copy()
        return out

    log(f"Tokenizing {len(texts)} examples (max_len={MAX_LEN})...")
    tokenized = [tokenize(t) for t in texts]

    dataset = Dataset.from_list([
        {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"], "labels": t["labels"]}
        for t in tokenized
    ])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    log(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    # Load base model in 4-bit
    log("Loading Nemotron-30B base in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    # LoRA config — match existing adapters exactly
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    log(f"After LoRA. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Training args
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "training"),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # Use simple data collator (texts already padded)
    def collator(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
    )

    log("Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed_min = (time.time() - t0) / 60
    log(f"Training done in {elapsed_min:.1f} min")

    # Save best adapter
    best_dir = OUT_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tok.save_pretrained(str(best_dir))
    log(f"Saved best adapter to {best_dir}")

    # Final eval
    metrics = trainer.evaluate()
    log(f"Final eval: {metrics}")
    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump({"final_metrics": metrics, "elapsed_min": elapsed_min}, f, indent=2)

    log("=== Training complete ===")


if __name__ == "__main__":
    main()
