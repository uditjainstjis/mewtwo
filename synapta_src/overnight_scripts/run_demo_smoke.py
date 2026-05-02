#!/usr/bin/env python3
"""Demo smoke test — runs 20 diverse prompts under 4 inference modes on Nemotron-30B.

Modes:
  base               : no adapter
  single_best        : pick adapter by prompt domain (math/code/science) — heuristic mapping
  token_routing      : LogitsProcessor that swaps adapter every 10 tokens (paradoxical heuristic)
  format_guard       : same as token_routing but freezes adapter inside ```python blocks

Output:
  qa_pairs/demo_smoke_<mode>.jsonl   one row per (prompt, mode)
  gpu_jobs/heartbeats/demo_smoke.txt updated every prompt
  gpu_jobs/logs/demo_smoke.log       full log

Safety:
  - Loads model once, reuses across all prompts.
  - Catches exceptions per-prompt; saves error row instead of crashing.
  - VRAM logged at start + each mode boundary.
  - max_new_tokens conservative.
"""
import os, sys, gc, json, time, re, traceback, datetime
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
OVERNIGHT = PROJECT / "overnight_run"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(OVERNIGHT / "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

from test_prompts import TEST_PROMPTS

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"

QA_DIR = OVERNIGHT / "qa_pairs"
HB_FILE = OVERNIGHT / "gpu_jobs" / "heartbeats" / "demo_smoke.txt"
LOG_FILE = OVERNIGHT / "gpu_jobs" / "logs" / "demo_smoke.log"
QA_DIR.mkdir(parents=True, exist_ok=True)
HB_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg, also_print=True):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def heartbeat(state, mode=None, prompt_id=None, prompt_idx=None, total=None):
    payload = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "state": state, "mode": mode, "prompt_id": prompt_id,
        "idx": prompt_idx, "total": total,
        "vram_used_mb": int(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
    }
    with open(HB_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def save_row(mode, row):
    out = QA_DIR / f"demo_smoke_{mode}.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def vram_used_gb():
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


# ---------- Routers ----------
def heuristic_router(decoded_text: str) -> str:
    """Paradoxical token router from token_router_eval.py."""
    text = decoded_text.lower()
    if re.search(r'```(?:python)?|def |import |class |    \w+', text):
        return "math"
    if re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', text):
        return "code"
    return "code"


class TokenRouterProcessor(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.swap_count = 0
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-50:], skip_special_tokens=True)
            new_ad = heuristic_router(ctx)
            if new_ad != self.current_adapter:
                self.model.set_adapter(new_ad)
                self.current_adapter = new_ad
                self.swap_count += 1
        return scores


class FormatGuardProcessor(LogitsProcessor):
    """Same paradoxical router but locks adapter inside ```python blocks."""
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.swap_count = 0
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-200:], skip_special_tokens=True)
            # Check if we're inside an open code block (odd count of ```)
            ticks = ctx.count("```")
            if ticks % 2 == 1:
                # Locked — favour math (which provides structural code) inside code blocks
                target = "math"
            else:
                target = heuristic_router(ctx)
            if target != self.current_adapter:
                self.model.set_adapter(target)
                self.current_adapter = target
                self.swap_count += 1
        return scores


# ---------- Inference helpers ----------
def get_hybrid_cache(base_model, model, batch_size=1):
    """Build the HybridMambaAttentionDynamicCache the way Nemotron requires."""
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, "HybridMambaAttentionDynamicCache")
    return HybridMambaAttentionDynamicCache(
        base_model.config,
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=model.device,
    )


def format_prompt(tok, user, sys_msg=None):
    if sys_msg is None:
        sys_msg = "You are a helpful assistant. Be concise and accurate."
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def map_domain_to_adapter(domain):
    if domain == "math":
        return "math"
    if domain == "code":
        return "math"  # Code Paradox: math adapter beats code adapter on code synthesis
    if domain == "science":
        return "science"
    if domain.startswith("mixed"):
        return "code"  # Code adapter is the universal hyper-reasoner per Code Paradox
    return "code"


def generate_one(model, base_model, tok, prompt_text, mode, prompt_meta=None, max_new=256):
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])

    swap_count = 0

    if mode == "base":
        # disable adapters
        with model.disable_adapter():
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id, use_cache=True,
                    past_key_values=cache,
                )
    elif mode == "single_best":
        adapter_name = map_domain_to_adapter(prompt_meta["domain"])
        model.set_adapter(adapter_name)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True,
                past_key_values=cache,
            )
    elif mode == "token_routing":
        proc = TokenRouterProcessor(model, tok)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True,
                past_key_values=cache,
                logits_processor=LogitsProcessorList([proc]),
            )
        swap_count = proc.swap_count
    elif mode == "format_guard":
        proc = FormatGuardProcessor(model, tok)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True,
                past_key_values=cache,
                logits_processor=LogitsProcessorList([proc]),
            )
        swap_count = proc.swap_count
    else:
        raise ValueError(f"unknown mode {mode}")

    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded, swap_count


# ---------- Main ----------
def main():
    log("=== Demo smoke test starting ===")
    heartbeat("starting")

    # Reset jsonl files (idempotent fresh run)
    for mode in ("base", "single_best", "token_routing", "format_guard"):
        f = QA_DIR / f"demo_smoke_{mode}.jsonl"
        if f.exists():
            f.unlink()

    log(f"VRAM at script start: {vram_used_gb():.2f} GB")

    log("Loading tokenizer + base model (4bit nf4)...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    base_model.eval()
    log(f"Base loaded. VRAM: {vram_used_gb():.2f} GB")

    log("Loading 3 adapters (math, code, science)...")
    def adapter_path(name):
        b = ADAPTER_BASE / name / "best"
        f = ADAPTER_BASE / name / "final"
        return str(b) if b.exists() else str(f)

    model = PeftModel.from_pretrained(
        base_model, adapter_path("math"), adapter_name="math", is_trainable=False,
    )
    model.load_adapter(adapter_path("code"), adapter_name="code")
    model.load_adapter(adapter_path("science"), adapter_name="science")
    model.eval()
    log(f"All adapters loaded. VRAM: {vram_used_gb():.2f} GB")

    heartbeat("loaded")

    modes = ["base", "single_best", "token_routing", "format_guard"]
    total = len(TEST_PROMPTS) * len(modes)
    counter = 0

    # Iterate prompts × modes
    for pi, p in enumerate(TEST_PROMPTS):
        prompt_text = format_prompt(tok, p["prompt"])
        for mode in modes:
            counter += 1
            heartbeat("running", mode=mode, prompt_id=p["id"], prompt_idx=counter, total=total)
            t0 = time.time()
            try:
                output, swap_count = generate_one(model, base_model, tok, prompt_text, mode, prompt_meta=p)
                elapsed = time.time() - t0

                signal = p["expected_signal"]
                contains_signal = (signal.lower() in output.lower()) if signal else None

                row = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "prompt_id": p["id"], "domain": p["domain"], "mode": mode,
                    "prompt": p["prompt"], "output": output,
                    "elapsed_s": round(elapsed, 2),
                    "swap_count": swap_count,
                    "contains_signal": contains_signal,
                    "expected_signal": signal,
                    "output_len": len(output),
                }
                save_row(mode, row)
                log(f"[{counter}/{total}] {p['id']} | {mode} | {elapsed:.1f}s | sig={contains_signal} | swaps={swap_count} | len={len(output)}")
            except torch.cuda.OutOfMemoryError as e:
                log(f"[{counter}/{total}] {p['id']} | {mode} | OOM: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                save_row(mode, {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "prompt_id": p["id"], "domain": p["domain"], "mode": mode,
                    "prompt": p["prompt"], "output": "",
                    "error": "OOM", "elapsed_s": time.time() - t0,
                })
            except Exception as e:
                log(f"[{counter}/{total}] {p['id']} | {mode} | EXC: {e}")
                log(traceback.format_exc(), also_print=False)
                save_row(mode, {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "prompt_id": p["id"], "domain": p["domain"], "mode": mode,
                    "prompt": p["prompt"], "output": "",
                    "error": str(e), "elapsed_s": time.time() - t0,
                })

            # release temp tensors between prompts
            torch.cuda.empty_cache()

    heartbeat("done", total=total)
    log("=== Demo smoke test complete ===")


if __name__ == "__main__":
    main()
