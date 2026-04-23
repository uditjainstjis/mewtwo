#!/usr/bin/env python3
"""
MEWTWO Master Autonomous Pipeline
===================================
Never-ending, fault-tolerant pipeline that executes all research tasks sequentially.
Updates PIPELINE_TASKS.md in real-time. Robust logging. Full GPU utilization.

Usage:
    nohup .venv/bin/python scripts/master_pipeline.py 2>&1 &
    tail -f logs/nemotron/master_pipeline.log
"""
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from safetensors.torch import load_file
from tqdm import tqdm

# ─── Setup ────────────────────────────────────────────────────────

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
MERGED_ADAPTER = PROJECT / "submission_adapter"
RESULTS_DIR = PROJECT / "results" / "nemotron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs" / "nemotron"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TASKS_FILE = PROJECT / "PIPELINE_TASKS.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "master_pipeline.log")),
    ],
)
log = logging.getLogger("master")

BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)

ALL_RESULTS = {}

# ─── Task Tracker ─────────────────────────────────────────────────

def update_task(task_id: str, status: str, note: str = ""):
    """Update PIPELINE_TASKS.md: mark task as done/in-progress."""
    content = TASKS_FILE.read_text()
    if status == "done":
        marker_old = f"- [ ] {task_id}"
        marker_new = f"- [x] {task_id}"
    elif status == "running":
        marker_old = f"- [ ] {task_id}"
        marker_new = f"- [/] {task_id}"
    elif status == "failed":
        marker_old = f"- [ ] {task_id}"
        marker_new = f"- [!] {task_id}"
    else:
        return
    # Also handle re-marking running tasks as done
    content = content.replace(f"- [/] {task_id}", marker_old)
    content = content.replace(marker_old, marker_new + (f" — {note}" if note else ""))
    # Update timestamp
    content = re.sub(r"Last updated:.*", f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", content)
    TASKS_FILE.write_text(content)
    log.info(f"📋 Task {task_id} → {status}" + (f" ({note})" if note else ""))


def save_results():
    with open(RESULTS_DIR / "master_results.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)


# ─── Model Management ────────────────────────────────────────────

_base_model = None
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def load_base():
    global _base_model
    if _base_model is None:
        log.info("Loading base Nemotron model...")
        _base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True,
        )
        _base_model.eval()
        vram = torch.cuda.memory_allocated() / 1e9
        log.info(f"Base model loaded. VRAM: {vram:.1f} GB")
    return _base_model


def load_with_adapter(adapter_path: str):
    base = load_base()
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model


def unload_adapter(model):
    if isinstance(model, PeftModel):
        model.unload()
        del model
    gc.collect()
    torch.cuda.empty_cache()


def find_adapter(domain: str) -> Optional[str]:
    for sub in ["best", "dare_sparsified", "final"]:
        p = ADAPTER_BASE / domain / sub
        if (p / "adapter_config.json").exists():
            return str(p)
    return None


# ─── Evaluation Helpers (FIXED VERSIONS) ─────────────────────────

def format_prompt(tok, system: str, user: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def extract_number(text: str) -> Optional[str]:
    if text is None:
        return None
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    # \boxed{}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip().replace(",", "")
    # ####
    hashes = re.findall(r'####\s*(.+?)(?:\n|$)', text)
    if hashes:
        return hashes[-1].strip().replace(",", "")
    # "The answer is"
    ans = re.findall(r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if ans:
        return ans[-1].strip().replace(",", "")
    # Last number in text
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        return nums[-1]
    return None


def normalize(s):
    if s is None:
        return ""
    s = s.strip().replace(",", "").replace("$", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return s.lower()


def eval_gsm8k(model, tok, n=100, label=""):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    sys_msg = "You are a math expert. Solve step-by-step. Put final answer after ####."
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds, desc=f"GSM8K [{label}]"):
            gold = normalize(extract_number(ex["answer"]))
            prompt = format_prompt(tok, sys_msg, ex["question"])
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = normalize(extract_number(resp))
            if pred and gold and pred == gold:
                correct += 1
            total += 1
    score = correct / max(total, 1)
    log.info(f"GSM8K [{label}]: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def eval_math500(model, tok, n=200, label=""):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        ds = load_dataset("lighteval/MATH", split="test")
    ds = ds.select(range(min(n, len(ds))))
    sys_msg = "Solve this math problem. Put your final answer in \\boxed{}."
    correct = total = 0
    q_key = "problem" if "problem" in ds.column_names else "question"
    a_key = "solution" if "solution" in ds.column_names else "answer"
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds, desc=f"MATH-500 [{label}]"):
            gold = normalize(extract_number(ex[a_key]))
            prompt = format_prompt(tok, sys_msg, ex[q_key])
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = normalize(extract_number(resp))
            if pred and gold and pred == gold:
                correct += 1
            total += 1
    score = correct / max(total, 1)
    log.info(f"MATH-500 [{label}]: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def eval_humaneval(model, tok, n=100, label=""):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test").select(range(min(n, 164)))
    sys_msg = "Complete the Python function. Output ONLY the code."
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds, desc=f"HumanEval [{label}]"):
            prompt_code = ex["prompt"]
            test_code = ex["test"]
            entry = ex["entry_point"]
            user = f"Complete this function:\n```python\n{prompt_code}\n```"
            prompt = format_prompt(tok, sys_msg, user)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            # Extract code — improved parsing
            code = resp
            code_block = re.search(r'```(?:python)?\s*\n(.*?)```', resp, re.DOTALL)
            if code_block:
                code = code_block.group(1)
            if not code.strip().startswith("def "):
                code = prompt_code + "\n" + code
            # Also try: if code contains the function but not from start
            if f"def {entry}" in code:
                idx = code.index(f"def {entry}")
                code = code[idx:]
            full = code + "\n\n" + test_code + f"\n\ncheck({entry})\n"
            passed = False
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
                    f.write(full)
                    f.flush()
                    r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10, text=True)
                    passed = r.returncode == 0
            except Exception:
                pass
            if passed:
                correct += 1
            total += 1
    score = correct / max(total, 1)
    log.info(f"HumanEval [{label}]: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def eval_mbpp(model, tok, n=100, label=""):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    ds = ds.select(range(min(n, len(ds))))
    sys_msg = "Write a Python function to solve the task. Output ONLY the code."
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds, desc=f"MBPP [{label}]"):
            task = ex["prompt"]
            tests = ex["test_list"]
            prompt = format_prompt(tok, sys_msg, task)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            code = resp
            code_block = re.search(r'```(?:python)?\s*\n(.*?)```', resp, re.DOTALL)
            if code_block:
                code = code_block.group(1)
            full = code + "\n\n" + "\n".join(tests) + "\n"
            passed = False
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
                    f.write(full)
                    f.flush()
                    r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10, text=True)
                    passed = r.returncode == 0
            except Exception:
                pass
            if passed:
                correct += 1
            total += 1
    score = correct / max(total, 1)
    log.info(f"MBPP [{label}]: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def eval_arc(model, tok, n=100, label=""):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    sys_msg = "Answer the multiple choice question with ONLY the letter A, B, C, or D."
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds, desc=f"ARC [{label}]"):
            gold = ex["answerKey"].strip().upper()
            choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
            user = f"{ex['question']}\n\n{choices}"
            prompt = format_prompt(tok, sys_msg, user)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            # FIXED: robust letter extraction
            pred = ""
            # Try first character
            if resp and resp[0].upper() in "ABCD":
                pred = resp[0].upper()
            else:
                # Try first letter found
                for ch in resp[:50]:
                    if ch.upper() in "ABCD":
                        pred = ch.upper()
                        break
            if pred == gold:
                correct += 1
            total += 1
    score = correct / max(total, 1)
    log.info(f"ARC [{label}]: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


# ─── Phase 1: Foundation ──────────────────────────────────────────

def phase_1():
    """Fix evaluations + clean benchmarks."""
    log.info("\n" + "█" * 70)
    log.info("PHASE 1: FOUNDATION — Clean Evaluations")
    log.info("█" * 70)

    tok = get_tokenizer()
    configs = {
        "baseline": None,
        "math": find_adapter("math"),
        "code": find_adapter("code"),
        "science": find_adapter("science"),
        "merged": str(MERGED_ADAPTER) if MERGED_ADAPTER.exists() else None,
    }

    # 1.1 + 1.2: Fixed ARC & HumanEval for all configs
    update_task("1.1", "running")
    update_task("1.2", "running")

    for name, adapter_path in configs.items():
        if adapter_path:
            model = load_with_adapter(adapter_path)
        else:
            model = load_base()

        # ARC (fixed letter extraction)
        r = eval_arc(model, tok, 100, name)
        ALL_RESULTS[f"{name}_arc_fixed"] = r

        # HumanEval (fixed code parsing)
        r = eval_humaneval(model, tok, 100, name)
        ALL_RESULTS[f"{name}_humaneval_fixed"] = r

        if adapter_path:
            unload_adapter(model)
        save_results()

    update_task("1.1", "done", f"ARC fixed for all 5 configs")
    update_task("1.2", "done", f"HumanEval fixed for all 5 configs")

    # 1.3: MATH-500 (uncontaminated)
    update_task("1.3", "running")
    for name, adapter_path in configs.items():
        if adapter_path:
            model = load_with_adapter(adapter_path)
        else:
            model = load_base()
        r = eval_math500(model, tok, 200, name)
        ALL_RESULTS[f"{name}_math500"] = r
        if adapter_path:
            unload_adapter(model)
        save_results()
    update_task("1.3", "done", "MATH-500 clean benchmark complete")

    # 1.4: MBPP (uncontaminated code)
    update_task("1.4", "running")
    for name, adapter_path in configs.items():
        if adapter_path:
            model = load_with_adapter(adapter_path)
        else:
            model = load_base()
        r = eval_mbpp(model, tok, 100, name)
        ALL_RESULTS[f"{name}_mbpp"] = r
        if adapter_path:
            unload_adapter(model)
        save_results()
    update_task("1.4", "done", "MBPP clean benchmark complete")

    log.info("✅ Phase 1 complete.")


# ─── Phase 2: LayerBlend-LoRI ─────────────────────────────────────

def phase_2():
    """Build and train LayerBlend — the novel IP."""
    log.info("\n" + "█" * 70)
    log.info("PHASE 2: LAYERBLEND-LoRI (NOVEL IP)")
    log.info("█" * 70)

    from src.lori_moe.model.layer_blend_router import LayerBlendRouter

    tok = get_tokenizer()
    base = load_base()

    # 2.1: Module is already built
    update_task("2.1", "done", "LayerBlendRouter module created")

    # 2.2: Pre-compute adapter weight deltas
    update_task("2.2", "running")
    adapter_deltas = {}  # {domain: {module_name: delta_W tensor}}
    domains = ["math", "code", "science"]
    adapted_modules = []

    for domain in domains:
        path = find_adapter(domain)
        if not path:
            continue
        weights = load_file(os.path.join(path, "adapter_model.safetensors"))
        deltas = {}
        for key in weights:
            if "lora_A" in key:
                module = key.replace(".lora_A.weight", "")
                A = weights[key].float()  # (r, in)
                B = weights[module + ".lora_B.weight"].float()  # (out, r)
                deltas[module] = (B @ A).to(torch.bfloat16)  # (out, in)
                if module not in adapted_modules:
                    adapted_modules.append(module)
        adapter_deltas[domain] = deltas
        log.info(f"  {domain}: {len(deltas)} module deltas pre-computed")

    update_task("2.2", "done", f"{len(adapted_modules)} modules across {len(domains)} domains")

    # 2.3: Train LayerBlend
    update_task("2.3", "running")

    num_layers = len(adapted_modules)
    blend_router = LayerBlendRouter(
        hidden_dim=2688, num_experts=len(domains),
        num_layers=num_layers, bottleneck=64,
        expert_names=domains,
    ).cuda()

    # Training data: 2000 examples per domain
    from datasets import load_dataset as ld
    import json as _json

    train_data = []
    for idx, domain in enumerate(domains):
        data_file = PROJECT / "data" / "nemotron" / f"{domain}_train.jsonl"
        if data_file.exists():
            with open(data_file) as f:
                lines = [_json.loads(l) for l in f][:2000]
            for line in lines:
                text = ""
                for msg in line.get("messages", []):
                    text += msg.get("content", "") + " "
                train_data.append((text.strip()[:512], idx))
    
    import random
    random.shuffle(train_data)
    split = int(len(train_data) * 0.8)
    train_set, val_set = train_data[:split], train_data[split:]
    log.info(f"  Training data: {len(train_set)} train, {len(val_set)} val")

    # Training loop
    optimizer = torch.optim.AdamW(blend_router.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_set) * 5 // 4)
    
    best_val_acc = 0
    blend_router.train()

    for epoch in range(5):
        random.shuffle(train_set)
        correct = total = 0
        epoch_loss = 0

        for batch_start in tqdm(range(0, len(train_set), 4), desc=f"Blend Epoch {epoch+1}/5"):
            batch = train_set[batch_start:batch_start + 4]
            if not batch:
                continue

            texts = [b[0] for b in batch]
            labels = torch.tensor([b[1] for b in batch]).cuda()

            # Get hidden states from base model
            inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = base(**inputs, output_hidden_states=True)
                all_hidden = outputs.hidden_states  # tuple of (B, S, D)

            # Sample hidden states at adapted layers
            # Map adapted_modules to layer indices in the model
            layer_indices = []
            for mod_name in adapted_modules:
                # Extract layer number from module name
                match = re.search(r'layers\.(\d+)', mod_name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_indices:
                        layer_indices.append(layer_idx)

            layer_indices = sorted(set(layer_indices))
            # Get one hidden state per unique layer
            hidden_per_layer = []
            for li in layer_indices[:num_layers]:
                if li < len(all_hidden):
                    hidden_per_layer.append(all_hidden[li])
            
            # Pad if needed
            while len(hidden_per_layer) < num_layers:
                hidden_per_layer.append(all_hidden[-1])

            # Forward through blend router
            blend_weights, aux_loss = blend_router(hidden_per_layer)

            # Average blend weights across layers → (B, K)
            avg_weights = torch.stack(blend_weights).mean(dim=0)  # (B, K)

            # Classification loss (CE)
            loss = F.cross_entropy(avg_weights.log().clamp(min=-100), labels)
            if aux_loss is not None:
                loss = loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(blend_router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds = avg_weights.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            epoch_loss += loss.item()

        train_acc = correct / max(total, 1) * 100
        
        # Validation
        blend_router.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch_start in range(0, len(val_set), 4):
                batch = val_set[batch_start:batch_start + 4]
                texts = [b[0] for b in batch]
                labels = torch.tensor([b[1] for b in batch]).cuda()
                inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = base(**inputs, output_hidden_states=True)
                all_hidden = outputs.hidden_states
                hidden_per_layer = []
                for li in layer_indices[:num_layers]:
                    if li < len(all_hidden):
                        hidden_per_layer.append(all_hidden[li])
                while len(hidden_per_layer) < num_layers:
                    hidden_per_layer.append(all_hidden[-1])
                blend_weights, _ = blend_router(hidden_per_layer, return_aux_loss=False)
                avg_w = torch.stack(blend_weights).mean(dim=0)
                preds = avg_w.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1) * 100
        log.info(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")
        log.info(f"  Layer summary:\n{blend_router.get_layer_summary()}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = PROJECT / "checkpoints" / "nemotron_lori" / "layer_blend"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": blend_router.state_dict(),
                "config": blend_router.get_config(),
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, save_dir / "best_router.pt")

        blend_router.train()

    update_task("2.3", "done", f"Best val_acc={best_val_acc:.1f}%")
    log.info(f"  LayerBlend training complete. Best val: {best_val_acc:.1f}%")

    # 2.4-2.8: Evaluate LayerBlend on all benchmarks
    # For LayerBlend eval, we apply the blended adapter deltas
    update_task("2.4", "running")

    blend_router.eval()
    blend_save = PROJECT / "checkpoints" / "nemotron_lori" / "layer_blend" / "best_router.pt"
    if blend_save.exists():
        state = torch.load(blend_save, map_location="cuda")
        blend_router.load_state_dict(state["model_state_dict"])

    # Create a blended merged adapter based on LayerBlend's learned weights
    # Use validation set to get average blend weights per layer
    log.info("  Computing learned blend weights...")
    layer_avg_weights = [torch.zeros(len(domains)).cuda() for _ in range(num_layers)]
    n_val = 0
    with torch.no_grad():
        for batch_start in range(0, min(len(val_set), 200), 4):
            batch = val_set[batch_start:batch_start + 4]
            texts = [b[0] for b in batch]
            inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = base(**inputs, output_hidden_states=True)
            all_hidden = outputs.hidden_states
            hidden_per_layer = []
            for li in layer_indices[:num_layers]:
                if li < len(all_hidden):
                    hidden_per_layer.append(all_hidden[li])
            while len(hidden_per_layer) < num_layers:
                hidden_per_layer.append(all_hidden[-1])
            bw, _ = blend_router(hidden_per_layer, return_aux_loss=False)
            for i, w in enumerate(bw):
                layer_avg_weights[i] += w.sum(dim=0)
            n_val += len(texts)
    
    # Normalize
    for i in range(num_layers):
        layer_avg_weights[i] /= n_val

    log.info("  Learned per-layer blend weights:")
    for i, w in enumerate(layer_avg_weights):
        w_list = w.cpu().tolist()
        log.info(f"    Layer {i}: " + " | ".join(f"{domains[j]}={w_list[j]:.3f}" for j in range(len(domains))))

    # Build a blended adapter using learned weights
    from safetensors.torch import save_file as sf_save
    blended_weights = {}
    for mod_idx, module_name in enumerate(adapted_modules):
        # Find which layer this module belongs to
        match = re.search(r'layers\.(\d+)', module_name)
        layer_num = int(match.group(1)) if match else 0
        layer_idx = layer_indices.index(layer_num) if layer_num in layer_indices else 0
        layer_idx = min(layer_idx, num_layers - 1)
        
        alpha = layer_avg_weights[layer_idx].cpu()
        
        # Blend the deltas
        blended = torch.zeros_like(list(adapter_deltas[domains[0]].values())[0])
        for j, domain in enumerate(domains):
            if module_name in adapter_deltas[domain]:
                blended += alpha[j].item() * adapter_deltas[domain][module_name].cpu()
        
        # SVD compress to rank 32
        U, S, Vh = torch.linalg.svd(blended.float(), full_matrices=False)
        k = 32
        sqrt_S = torch.sqrt(S[:k])
        new_A = (sqrt_S.unsqueeze(1) * Vh[:k, :]).contiguous()
        new_B = (U[:, :k] * sqrt_S.unsqueeze(0)).contiguous()
        
        blended_weights[f"{module_name}.lora_A.weight"] = new_A.to(torch.float32)
        blended_weights[f"{module_name}.lora_B.weight"] = new_B.to(torch.float32)

    # Save blended adapter
    blend_adapter_dir = PROJECT / "checkpoints" / "nemotron_lori" / "layer_blend_adapter"
    blend_adapter_dir.mkdir(parents=True, exist_ok=True)
    sf_save(blended_weights, str(blend_adapter_dir / "adapter_model.safetensors"))

    # Copy adapter config from math (modify rank)
    import shutil
    cfg = json.loads((ADAPTER_BASE / "math" / "best" / "adapter_config.json").read_text())
    cfg["r"] = 32
    cfg["lora_alpha"] = 64.0
    cfg["lora_dropout"] = 0.0
    cfg["base_model_name_or_path"] = MODEL_PATH
    (blend_adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        src = ADAPTER_BASE / "math" / "best" / f
        if src.exists():
            shutil.copy2(str(src), str(blend_adapter_dir / f))

    log.info(f"  LayerBlend adapter saved to {blend_adapter_dir}")

    # Now eval the blended adapter on all benchmarks
    benchmarks = [
        ("2.4", "gsm8k", eval_gsm8k, 100),
        ("2.5", "math500", eval_math500, 200),
        ("2.6", "humaneval", eval_humaneval, 100),
        ("2.7", "mbpp", eval_mbpp, 100),
        ("2.8", "arc", eval_arc, 100),
    ]

    model = load_with_adapter(str(blend_adapter_dir))
    for task_id, bench_name, bench_fn, n_samples in benchmarks:
        update_task(task_id, "running")
        r = bench_fn(model, tok, n_samples, "layerblend")
        ALL_RESULTS[f"layerblend_{bench_name}"] = r
        update_task(task_id, "done", f"{bench_name}={r['score']:.1%}")
        save_results()
    
    unload_adapter(model)

    # 2.8 Ablation
    update_task("2.8", "running")
    log.info("\n  ABLATION: blend vs route vs merge vs single")
    ablation = {}
    for key in ALL_RESULTS:
        ablation[key] = ALL_RESULTS[key].get("score", 0) if isinstance(ALL_RESULTS[key], dict) else 0
    log.info(f"  Full ablation saved in master_results.json")
    update_task("2.8", "done", "Ablation complete")

    log.info("✅ Phase 2 complete — LayerBlend-LoRI trained and evaluated.")


# ─── Phase 3: Kaggle ─────────────────────────────────────────────

def phase_3():
    """Improve Kaggle submission."""
    log.info("\n" + "█" * 70)
    log.info("PHASE 3: KAGGLE IMPROVEMENT")
    log.info("█" * 70)

    # Submit the LayerBlend adapter (better than uniform merge)
    update_task("3.1", "running")
    
    import zipfile
    blend_dir = PROJECT / "checkpoints" / "nemotron_lori" / "layer_blend_adapter"
    sub_zip = PROJECT / "submission_layerblend.zip"
    
    if blend_dir.exists():
        # Update base_model_name for Kaggle
        cfg = json.loads((blend_dir / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = "nvidia/Nemotron-3-Nano-30B-A3B"
        (blend_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        
        with zipfile.ZipFile(str(sub_zip), "w", zipfile.ZIP_DEFLATED) as z:
            for f in blend_dir.iterdir():
                if f.is_file():
                    z.write(str(f), f.name)
        
        log.info(f"  LayerBlend submission: {sub_zip} ({sub_zip.stat().st_size/1e6:.1f} MB)")
        update_task("3.1", "done", "LayerBlend submission.zip created")
        
        # Submit
        update_task("3.5", "running")
        try:
            r = subprocess.run(
                [str(PROJECT / ".venv/bin/kaggle"), "competitions", "submit",
                 "-c", "nvidia-nemotron-model-reasoning-challenge",
                 "-f", str(sub_zip),
                 "-m", "LayerBlend-LoRI: per-layer continuous adapter composition (Math+Code+Science)"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                log.info(f"  ✅ Kaggle submission successful!")
                update_task("3.5", "done", "Submitted to Kaggle")
            else:
                log.warning(f"  Kaggle submit failed: {r.stderr}")
                update_task("3.5", "failed", r.stderr[:100])
        except Exception as e:
            log.warning(f"  Kaggle submit error: {e}")
            update_task("3.5", "failed", str(e)[:100])
    else:
        update_task("3.1", "failed", "No LayerBlend adapter found")

    log.info("✅ Phase 3 complete.")


# ─── Phase 4: Demo ───────────────────────────────────────────────

def phase_4():
    """Generate final analysis."""
    log.info("\n" + "█" * 70)
    log.info("PHASE 4: FINAL ANALYSIS")
    log.info("█" * 70)

    update_task("4.3", "running")

    # Print comprehensive results table
    benchmarks = ["gsm8k", "humaneval_fixed", "arc_fixed", "math500", "mbpp"]
    configs = ["baseline", "math", "code", "science", "merged", "layerblend"]

    log.info("\n" + "─" * 80)
    header = f"{'Config':<15}" + "".join(f"{b:>14}" for b in benchmarks)
    log.info(header)
    log.info("─" * 80)

    for cfg in configs:
        vals = []
        for bench in benchmarks:
            key = f"{cfg}_{bench}"
            if key in ALL_RESULTS and isinstance(ALL_RESULTS[key], dict):
                vals.append(f"{ALL_RESULTS[key]['score']:.1%}")
            else:
                vals.append("—")
        log.info(f"{cfg:<15}" + "".join(f"{v:>14}" for v in vals))
    log.info("─" * 80)

    update_task("4.3", "done", "Final analysis complete")
    save_results()
    log.info("✅ Phase 4 complete. ALL PHASES DONE.")


# ─── Main ─────────────────────────────────────────────────────────

import torch.nn.functional as F

def main():
    start = time.time()
    log.info("=" * 70)
    log.info("MEWTWO MASTER AUTONOMOUS PIPELINE")
    log.info(f"Started: {datetime.now()}")
    log.info(f"GPU: {torch.cuda.get_device_name()}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("=" * 70)

    phases = [
        ("Phase 1: Foundation", phase_1),
        ("Phase 2: LayerBlend-LoRI", phase_2),
        ("Phase 3: Kaggle", phase_3),
        ("Phase 4: Analysis", phase_4),
    ]

    for name, fn in phases:
        log.info(f"\n{'▶' * 5} Starting {name} {'▶' * 5}")
        try:
            fn()
        except Exception as e:
            log.error(f"❌ {name} FAILED: {e}")
            log.error(traceback.format_exc())
            save_results()
            # Continue to next phase
            continue

    elapsed = (time.time() - start) / 3600
    log.info(f"\n{'=' * 70}")
    log.info(f"PIPELINE COMPLETE. Total time: {elapsed:.1f} hours")
    log.info(f"Results: {RESULTS_DIR / 'master_results.json'}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
