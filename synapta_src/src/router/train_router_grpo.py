from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from collaborative_reasoning import CollaborativeReasoner  # noqa: E402
from dynamic_mlx_inference import DynamicEngine  # noqa: E402

ROUTER_SYSTEM_PROMPT = (
    "You are the TCAR routing model. Analyze the user's request, plan the required reasoning steps, "
    "and output the exact expert tags needed to solve the task.\n"
    "Return exactly this format:\n"
    "<thinking>\n"
    "- short bullet\n"
    "- short bullet\n"
    "</thinking>\n"
    "<experts>[DOMAIN_A],[DOMAIN_B]</experts>"
)


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


def build_prompt(query: str) -> str:
    return (
        f"<|im_start|>system\n{ROUTER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    import re

    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def token_f1(pred: str, ref: str) -> float:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    p_count, r_count = {}, {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in r:
        r_count[t] = r_count.get(t, 0) + 1
    overlap = sum(min(p_count[t], r_count.get(t, 0)) for t in p_count)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_experts(text: str, domains: list[str], max_experts: int) -> list[str]:
    import re

    seen: list[str] = []
    upper = text.upper()
    for domain in domains:
        if f"[{domain}]" in upper or re.search(rf"\b{re.escape(domain)}\b", upper):
            seen.append(domain)
    deduped: list[str] = []
    for domain in seen:
        if domain not in deduped:
            deduped.append(domain)
    if not deduped:
        return [domains[0]]
    return deduped[: max(1, int(max_experts))]


def build_features(tokenizer, prompt: str, completion: str, max_length: int, device: str) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if len(labels) < len(full_ids):
        labels += [-100] * (len(full_ids) - len(labels))
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * len(full_ids)], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def sequence_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid
    return token_logps.sum(dim=-1)


def modality_kl(pred_experts: list[str], gold_experts: list[str], domains: list[str], eps: float = 1e-4) -> float:
    if not pred_experts:
        return 0.0
    p = {d: 0.0 for d in domains}
    for d in pred_experts:
        if d in p:
            p[d] += 1.0 / len(pred_experts)
    q = {d: eps for d in domains}
    if gold_experts:
        mass = max(0.0, 1.0 - eps * len(domains))
        for d in gold_experts:
            if d in q:
                q[d] += mass / len(gold_experts)
    total_q = sum(q.values())
    q = {k: v / total_q for k, v in q.items()}
    kl = 0.0
    for d, p_i in p.items():
        if p_i <= 0:
            continue
        kl += p_i * math.log(max(p_i, eps) / max(q[d], eps))
    return kl


def sample_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = output[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=False).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--registry", default="backend/expert_registry.json")
    parser.add_argument("--data", default="data/multidomain_eval_claude_external_v2_100.json")
    parser.add_argument("--output-dir", default="adapters/routers/router_reasoning_grpo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-experts", type=int, default=2)
    parser.add_argument("--router-max-tokens", type=int, default=64)
    parser.add_argument("--expert-max-tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward-kl-beta", type=float, default=0.15)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for train_router_grpo.py on this stack.")

    device = "mps"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    data_path = PROJECT_ROOT / args.data
    registry_path = PROJECT_ROOT / args.registry
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No GRPO training rows found.")

    with open(registry_path, "r") as f:
        registry = json.load(f)
    domains = list(registry.keys())
    os.chdir(BACKEND_DIR)

    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    policy_model = PeftModel.from_pretrained(base_model, str(PROJECT_ROOT / args.sft_adapter), is_trainable=True)
    policy_model.to(device)
    policy_model.train()

    ref_base = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, str(PROJECT_ROOT / args.sft_adapter), is_trainable=False)
    ref_model.to(device)
    ref_model.eval()

    # Collaborative evaluator uses the current verifier-based TCAR path.
    reasoner = CollaborativeReasoner(
        DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry),
        registry_path,
    )

    optimizer = torch.optim.AdamW((p for p in policy_model.parameters() if p.requires_grad), lr=args.learning_rate)
    start_time = time.time()
    step = 0
    running_reward = 0.0
    running_loss = 0.0

    while step < args.max_steps:
        row = rows[step % len(rows)]
        query = row["question"]
        reference = row["reference_answer"]
        gold_experts = list(row.get("required_adapters") or row.get("domains") or [])
        prompt = build_prompt(query)

        samples = []
        for _ in range(max(1, int(args.samples_per_prompt))):
            completion = sample_completion(
                policy_model,
                tokenizer,
                prompt,
                max_tokens=args.router_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            pred_experts = extract_experts(completion, domains, args.max_experts)
            result = reasoner.run_with_experts(
                query,
                pred_experts,
                expert_max_tokens=args.expert_max_tokens,
                refine_max_tokens=0,
                router_thinking="grpo sampled routing candidate",
            )
            reward_f1 = token_f1(result.final_answer, reference)
            penalty_kl = modality_kl(pred_experts, gold_experts, domains)
            reward = reward_f1 - args.reward_kl_beta * penalty_kl
            features = build_features(tokenizer, prompt, completion, args.max_length, device)
            policy_logp = sequence_logps(policy_model, features)
            with torch.no_grad():
                ref_logp = sequence_logps(ref_model, features)
            samples.append(
                {
                    "completion": completion,
                    "pred_experts": pred_experts,
                    "reward": reward,
                    "reward_f1": reward_f1,
                    "penalty_kl": penalty_kl,
                    "policy_logp": policy_logp,
                    "ref_logp": ref_logp,
                }
            )

        rewards = torch.tensor([s["reward"] for s in samples], dtype=torch.float32, device=device)
        advantages = rewards - rewards.mean()
        adv_std = advantages.std(unbiased=False)
        if float(adv_std.item()) > 1e-6:
            advantages = advantages / adv_std

        losses = []
        for sample, advantage in zip(samples, advantages):
            logp = sample["policy_logp"]
            ref_logp = sample["ref_logp"]
            # Lightweight GRPO: group-relative advantage with a small reference tether.
            losses.append(-(advantage.detach() * logp).mean() + 0.01 * (logp - ref_logp).pow(2).mean())
        loss = torch.stack(losses).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        running_loss += float(loss.detach().cpu())
        running_reward += float(rewards.mean().detach().cpu())

        if step % args.logging_steps == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                json.dumps(
                    {
                        "step": step,
                        "total_steps": args.max_steps,
                        "train_loss": round(running_loss / max(1, args.logging_steps), 4),
                        "mean_reward": round(running_reward / max(1, args.logging_steps), 4),
                        "last_reward_f1": round(float(samples[0]["reward_f1"]), 4),
                        "last_penalty_kl": round(float(samples[0]["penalty_kl"]), 4),
                        "elapsed_s": round(elapsed, 2),
                        "device": device,
                    }
                ),
                flush=True,
            )
            running_loss = 0.0
            running_reward = 0.0

    policy_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "sft_adapter": str((PROJECT_ROOT / args.sft_adapter).relative_to(PROJECT_ROOT)),
                "data": str(data_path.relative_to(PROJECT_ROOT)),
                "task": "router_grpo_manual",
                "device": device,
                "dtype": args.dtype,
                "samples_per_prompt": args.samples_per_prompt,
                "max_steps": args.max_steps,
                "reward_kl_beta": args.reward_kl_beta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
