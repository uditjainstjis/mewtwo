#!/usr/bin/env python3
"""Phase 07: Train BFSI extractive QA LoRA on Nemotron-30B-Nano.

~4477 (context, question, answer) triples from RBI/SEBI master directions.
Loss is masked to the assistant span only. 700-row held-out eval is run
separately by the Phase G script; the 5% in-train split is for early-stop.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

PROJECT = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH = str(PROJECT / "models" / "nemotron")
QA_DIR = PROJECT / "data" / "rbi_corpus" / "qa"
TRAIN_CLEAN = QA_DIR / "train_clean.jsonl"
TRAIN_FALLBACK = QA_DIR / "train.jsonl"

ADAPTER_ROOT = PROJECT / "adapters" / "nemotron_30b" / "bfsi_extract"
BEST_DIR = ADAPTER_ROOT / "best"
TRAIN_DIR = ADAPTER_ROOT / "training"
TRAIN_LOG_JSON = ADAPTER_ROOT / "training_log.json"
LOG_DIR = PROJECT / "logs" / "data_pipeline"
LOG_FILE = LOG_DIR / "07_train.log"
for d in (BEST_DIR, TRAIN_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_LEN = 1024  # token length analysis: max=857, 99th=849; 1024 covers 100% of corpus, halves activation memory
SEED = 42
EVAL_FRAC = 0.05
ASSISTANT_MARKER = "<|im_start|>assistant\n"
SYSTEM_PROMPT = (
    "You are a senior banking and financial regulation expert in India. "
    "Read the provided regulatory context carefully and answer the question "
    "precisely with the specific number, term, rule, or section citation. "
    "Quote directly from the regulation when possible."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(str(LOG_FILE), mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("07_train_bfsi_extract")


def vram_gb() -> float:
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


def load_examples() -> list[dict]:
    src = TRAIN_CLEAN if TRAIN_CLEAN.exists() else TRAIN_FALLBACK
    log.info("Loading training data from %s", src)
    rows: list[dict] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ctx = (ex.get("context") or "").strip()
            q = (ex.get("question") or "").strip()
            a = (ex.get("answer") or "").strip()
            if not (ctx and q and a):
                continue
            rows.append({"context": ctx, "question": q, "answer": a})
    log.info("Loaded %d valid training examples", len(rows))
    return rows


def build_messages(ex: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"REGULATORY CONTEXT:\n{ex['context']}\n\n"
                f"QUESTION: {ex['question']}\n\nANSWER:"
            ),
        },
        {"role": "assistant", "content": ex["answer"]},
    ]


def render_chat(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )


def make_tokenize_fn(tokenizer):
    def _tokenize(ex: dict) -> dict:
        full_text = render_chat(tokenizer, build_messages(ex))
        idx = full_text.rfind(ASSISTANT_MARKER)
        prefix_text = full_text[: idx + len(ASSISTANT_MARKER)] if idx >= 0 else ""
        prefix_len = len(
            tokenizer(prefix_text, add_special_tokens=False, truncation=False)["input_ids"]
        )
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors=None,
        )
        labels = list(enc["input_ids"])
        cut = min(prefix_len, len(labels))
        for i in range(cut):
            labels[i] = -100
        for i, m in enumerate(enc["attention_mask"]):
            if m == 0:
                labels[i] = -100
        enc["labels"] = labels
        enc["_has_supervision"] = int(any(l != -100 for l in labels))
        return enc

    return _tokenize


def collate(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        ),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }


def main() -> None:
    log.info("=" * 72)
    log.info("Phase 07: BFSI extractive LoRA on Nemotron-30B-Nano")
    log.info("Run started at %s", datetime.utcnow().isoformat() + "Z")

    log.info("Loading tokenizer from %s", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        log.info("pad_token missing - falling back to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.eos_token_id is not None, "EOS_TOKEN_ID is None - tokenizer broken"
    log.info(
        "Tokenizer ready. eos_id=%s pad_id=%s", tokenizer.eos_token_id, tokenizer.pad_token_id
    )

    examples = load_examples()
    raw_ds = Dataset.from_list(examples)
    log.info("Tokenizing %d examples (MAX_LEN=%d)...", len(raw_ds), MAX_LEN)
    tokenized = raw_ds.map(
        make_tokenize_fn(tokenizer),
        remove_columns=raw_ds.column_names,
        desc="tokenize+mask",
    )
    before = len(tokenized)
    tokenized = tokenized.filter(lambda r: r["_has_supervision"] == 1)
    dropped = before - len(tokenized)
    if dropped:
        log.warning("Dropped %d examples with no supervised tokens after truncation", dropped)
    tokenized = tokenized.remove_columns(["_has_supervision"])

    sample_text = render_chat(tokenizer, build_messages(examples[0]))
    log.info("Sample formatted example (first 800 chars):\n%s", sample_text[:800])

    splits = tokenized.train_test_split(test_size=EVAL_FRAC, seed=SEED)
    log.info("Split: train=%d eval=%d", len(splits["train"]), len(splits["test"]))

    log.info("VRAM before model load: %.2f GB", vram_gb())
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    log.info("Loading base Nemotron-30B in 4-bit NF4...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    log.info("VRAM after model load: %.2f GB", vram_gb())

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    log.info("VRAM after LoRA wrap: %.2f GB", vram_gb())

    args = TrainingArguments(
        output_dir=str(TRAIN_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,  # GPU budget tight; 1 epoch on cleaner v2 data is sufficient for knowledge injection
        learning_rate=1e-4,
        max_grad_norm=0.3,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=False,  # have ~12GB headroom, recompute is 2x backward overhead
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=collate,
    )

    log.info("Starting training...")
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as oom:
        log.error("CUDA OOM during training: %s | VRAM=%.2f GB", oom, vram_gb())
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Training step failed: %s", exc)
        raise
    elapsed_min = (time.time() - t0) / 60.0
    log.info("Training finished in %.1f min", elapsed_min)

    log.info("Running final eval (700-row held-out eval is separate)...")
    final_metrics = trainer.evaluate()
    log.info("Final eval metrics: %s", final_metrics)

    log.info("Saving best adapter to %s", BEST_DIR)
    model.save_pretrained(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))

    summary = {
        "phase": "07_train_bfsi_extract", "base_model": MODEL_PATH,
        "train_examples": len(splits["train"]), "eval_examples": len(splits["test"]),
        "dropped_truncated": dropped, "max_len": MAX_LEN,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": lora_cfg.target_modules},
        "epochs": 2, "effective_batch_size": 16, "learning_rate": 1e-4,
        "elapsed_minutes": elapsed_min, "final_metrics": final_metrics,
        "best_dir": str(BEST_DIR), "training_dir": str(TRAIN_DIR),
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    TRAIN_LOG_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary to %s", TRAIN_LOG_JSON)
    log.info("=== Phase 07 complete ===")


if __name__ == "__main__":
    main()
