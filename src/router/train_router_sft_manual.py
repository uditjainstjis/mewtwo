from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class MaskConfig:
    assistant_prefix_ids: list[int]
    im_end_ids: list[int]


def find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    limit = len(sequence) - len(pattern) + 1
    for idx in range(max(0, start), max(0, limit)):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def build_loss_labels(input_ids: list[int], mask_config: MaskConfig) -> list[int]:
    labels = [-100] * len(input_ids)
    assistant_start = find_subsequence(input_ids, mask_config.assistant_prefix_ids)
    if assistant_start < 0:
        raise ValueError("Could not find assistant start marker in SFT sample.")
    content_start = assistant_start + len(mask_config.assistant_prefix_ids)
    assistant_end = find_subsequence(input_ids, mask_config.im_end_ids, start=content_start)
    content_end = assistant_end if assistant_end >= 0 else len(input_ids)
    if content_end <= content_start:
        raise ValueError("Assistant payload was truncated away; no supervised tokens remain.")
    for idx in range(content_start, content_end):
        labels[idx] = input_ids[idx]
    return labels


def load_model_and_tokenizer(model_name_or_path: str, *, dtype: torch.dtype):
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
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def build_mask_config(tokenizer) -> MaskConfig:
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not assistant_prefix_ids or not im_end_ids:
        raise ValueError("Failed to build assistant/im_end token masks for router SFT.")
    return MaskConfig(assistant_prefix_ids=assistant_prefix_ids, im_end_ids=im_end_ids)


def collate_batch(tokenizer, rows: list[dict], mask_config: MaskConfig, max_length: int, device: str) -> dict[str, torch.Tensor]:
    encoded_rows = [
        tokenizer(
            row["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        for row in rows
    ]
    labels_rows = [build_loss_labels(ids, mask_config) for ids in encoded_rows]
    max_seq_len = max(len(ids) for ids in encoded_rows)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    pad_id = tokenizer.pad_token_id
    for ids, labels in zip(encoded_rows, labels_rows):
        pad_len = max_seq_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }


@torch.no_grad()
def evaluate(model, tokenizer, rows: list[dict], *, mask_config: MaskConfig, batch_size: int, max_length: int, device: str, limit: int = 64) -> float:
    if not rows:
        return 0.0
    model.eval()
    subset = rows[: min(limit, len(rows))]
    losses = []
    for start in range(0, len(subset), batch_size):
        batch_rows = subset[start : start + batch_size]
        batch = collate_batch(tokenizer, batch_rows, mask_config=mask_config, max_length=max_length, device=device)
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


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
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for train_router_sft_manual.py on this stack.")

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(data_dir / "train.jsonl")
    valid_rows = load_rows(data_dir / "valid.jsonl")

    device = "mps"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)
    mask_config = build_mask_config(tokenizer)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    micro_batch_size = max(1, args.per_device_train_batch_size)
    grad_accum = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_rows) / micro_batch_size / grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(args.num_train_epochs * steps_per_epoch)))

    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(math.ceil(args.num_train_epochs) if args.max_steps <= 0 else 10**9):
        if global_step >= total_steps:
            break
        random.shuffle(train_rows)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        for start in range(0, len(train_rows), micro_batch_size):
            batch_rows = train_rows[start : start + micro_batch_size]
            batch = collate_batch(
                tokenizer,
                batch_rows,
                mask_config=mask_config,
                max_length=args.max_length,
                device=device,
            )
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu())
            micro_step += 1

            if micro_step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0 or global_step == 1:
                    elapsed = time.time() - start_time
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "total_steps": total_steps,
                                "train_loss": round(running_loss / max(1, args.logging_steps), 4),
                                "elapsed_s": round(elapsed, 2),
                                "device": device,
                            }
                        ),
                        flush=True,
                    )
                    running_loss = 0.0

                if global_step % args.eval_steps == 0 or global_step == total_steps:
                    eval_loss = evaluate(
                        model,
                        tokenizer,
                        valid_rows,
                        mask_config=mask_config,
                        batch_size=micro_batch_size,
                        max_length=args.max_length,
                        device=device,
                    )
                    print(json.dumps({"step": global_step, "eval_loss": round(eval_loss, 4)}), flush=True)

                if global_step >= total_steps:
                    break
        if micro_step % grad_accum != 0 and global_step < total_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step >= total_steps:
                break

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "data_dir": str(data_dir.relative_to(PROJECT_ROOT)),
                "task": "router_sft_manual",
                "device": device,
                "dtype": args.dtype,
                "total_steps": global_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
