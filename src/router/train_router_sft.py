from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path)
    if "/" not in model_name_or_path:
        return model_name_or_path
    org, name = model_name_or_path.split("/", 1)
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"
    snapshots_dir = hub_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name_or_path
    snapshots = sorted(snapshots_dir.iterdir())
    return str(snapshots[-1]) if snapshots else model_name_or_path


def load_model_and_tokenizer(model_name_or_path: str, *, force_cpu: bool = False):
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        dtype=torch.float32 if force_cpu or not torch.backends.mps.is_available() else torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", default="data/router_reasoning_sft")
    parser.add_argument("--output-dir", default="router_adapters/router_reasoning_sft")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    datasets_cache_dir = PROJECT_ROOT / ".cache" / "huggingface" / "datasets"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    datasets_cache_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, force_cpu=args.force_cpu)
    train_dataset = load_dataset(
        "json",
        data_files=str(data_dir / "train.jsonl"),
        split="train",
        cache_dir=str(datasets_cache_dir),
    )
    eval_dataset = load_dataset(
        "json",
        data_files=str(data_dir / "valid.jsonl"),
        split="train",
        cache_dir=str(datasets_cache_dir),
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        optim="adamw_torch",
        report_to="none",
        max_length=args.max_length,
        dataset_text_field="text",
        gradient_checkpointing=True,
        use_cpu=args.force_cpu or not torch.backends.mps.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
                "task": "router_sft",
                "force_cpu": args.force_cpu,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
