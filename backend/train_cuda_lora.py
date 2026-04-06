import argparse
import json
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class JsonlTextDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj["text"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on CUDA using Transformers + PEFT.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = os.path.join(args.data, "train.jsonl")
    valid_path = os.path.join(args.data, "valid.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    train_dataset = JsonlTextDataset(train_path)
    eval_dataset = JsonlTextDataset(valid_path) if os.path.exists(valid_path) else None
    collator = SFTCollator(tokenizer, max_length=args.max_length)
    output_dir = os.path.join(args.adapter_path, "_trainer_output")
    os.makedirs(args.adapter_path, exist_ok=True)

    grad_accum = max(1, math.ceil(args.iters / max(1, len(train_dataset))))
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=max(1, min(10, args.iters)),
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=max(1, args.iters // 4) if eval_dataset is not None else None,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.adapter_path)


if __name__ == "__main__":
    main()
