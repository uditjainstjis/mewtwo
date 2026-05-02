#!/usr/bin/env python3
"""Full HumanEval (n=164) under base + format_guard on Nemotron-30B.

Goal: replace n=25 numbers in the deck with publication-credible n=164 numbers.

Output:
  qa_pairs/humaneval_full_<mode>.jsonl   one row per (task_id, mode)
  qa_pairs/humaneval_full_summary.json   pass@1 per mode at the end
  gpu_jobs/heartbeats/humaneval.txt      every prompt
  gpu_jobs/logs/humaneval.log

Pass@1 is computed by running each generated solution + the canonical test cases
in a subprocess sandbox with 10s timeout (matching openai/human-eval protocol).
"""
import os, sys, gc, json, time, re, traceback, datetime, tempfile, subprocess, signal
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
OVERNIGHT = PROJECT / "overnight_run"
sys.path.insert(0, str(PROJECT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from datasets import load_dataset

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"

QA_DIR = OVERNIGHT / "qa_pairs"
HB_FILE = OVERNIGHT / "gpu_jobs" / "heartbeats" / "humaneval.txt"
LOG_FILE = OVERNIGHT / "gpu_jobs" / "logs" / "humaneval.log"
SUMMARY_FILE = QA_DIR / "humaneval_full_summary.json"
QA_DIR.mkdir(parents=True, exist_ok=True)
HB_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MODES = ["base", "format_guard"]
MAX_NEW = 512
TIMEOUT_SEC = 10


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def heartbeat(state, mode=None, idx=None, total=None, passed=None):
    payload = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "state": state, "mode": mode, "idx": idx, "total": total, "passed": passed,
        "vram_used_mb": int(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
    }
    with open(HB_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def save_row(mode, row):
    out = QA_DIR / f"humaneval_full_{mode}.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------- Format Guard processor (same as smoke test) ----------
def heuristic_router(decoded_text: str) -> str:
    text = decoded_text.lower()
    if re.search(r'```(?:python)?|def |import |class |    \w+', text):
        return "math"
    if re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', text):
        return "code"
    return "code"


class FormatGuardProcessor(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.swap_count = 0
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-200:], skip_special_tokens=True)
            ticks = ctx.count("```")
            if ticks % 2 == 1:
                target = "math"
            else:
                target = heuristic_router(ctx)
            if target != self.current_adapter:
                self.model.set_adapter(target)
                self.current_adapter = target
                self.swap_count += 1
        return scores


def get_hybrid_cache(base_model, model, batch_size=1):
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, "HybridMambaAttentionDynamicCache")
    return HybridMambaAttentionDynamicCache(
        base_model.config, batch_size=batch_size,
        dtype=torch.bfloat16, device=model.device,
    )


def build_prompt(tok, problem):
    """HumanEval prompt: just the function header + docstring; we ask model to complete the body."""
    user = (
        "Complete the following Python function. Only output the completed function body code, "
        "no explanations.\n\n"
        f"```python\n{problem['prompt']}\n```"
    )
    sys_msg = "You are an expert Python programmer. Complete the function precisely."
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_code(text, prompt):
    """Pull the first python code block out of the model output and stitch with prompt prefix.

    Critical: HumanEval prompts include `from typing import List` etc. before the def.
    If the model returns a code block containing only the def, we MUST keep the imports
    from the prompt or the test fails with NameError.
    """
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    body = m.group(1) if m else text

    # Extract the imports/header section from the prompt: everything before the first def/class.
    prompt_lines = prompt.split("\n")
    header_lines = []
    for line in prompt_lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ")):
            break
        header_lines.append(line)
    header = "\n".join(header_lines).strip()

    # Case A: body contains the def — prepend just the header (imports) to keep them.
    prompt_def_match = re.search(r"^def\s+(\w+)", prompt, re.MULTILINE)
    if prompt_def_match and prompt_def_match.group(0) in body:
        return (header + "\n\n" + body) if header else body

    # Case B: body is just the function-body completion — prepend the full prompt.
    return prompt + "\n" + body


def run_test(code, test_code, entry_point):
    """Execute code in a subprocess and check tests."""
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full)
            fpath = f.name
        try:
            r = subprocess.run(
                ["/home/learner/Desktop/mewtwo/.venv/bin/python", fpath],
                timeout=TIMEOUT_SEC, capture_output=True, text=True,
            )
            ok = (r.returncode == 0)
            return ok, r.stderr[:500] if not ok else ""
        finally:
            try:
                os.unlink(fpath)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exec_err: {e}"


def generate_one(model, base_model, tok, prompt_text, mode, max_new=MAX_NEW):
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])
    swap_count = 0

    if mode == "base":
        with model.disable_adapter():
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                )
    elif mode == "format_guard":
        proc = FormatGuardProcessor(model, tok)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                logits_processor=LogitsProcessorList([proc]),
            )
        swap_count = proc.swap_count
    else:
        raise ValueError(f"unknown mode {mode}")

    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded, swap_count


def main():
    log("=== HumanEval n=164 starting ===")
    heartbeat("starting")

    # Reset jsonls
    for mode in MODES:
        f = QA_DIR / f"humaneval_full_{mode}.jsonl"
        if f.exists():
            f.unlink()

    log("Loading HumanEval dataset...")
    ds = load_dataset("openai_humaneval", split="test")
    log(f"Loaded {len(ds)} problems.")

    log("Loading tokenizer + base model (4bit)...")
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
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    def adapter_path(name):
        b = ADAPTER_BASE / name / "best"
        f = ADAPTER_BASE / name / "final"
        return str(b) if b.exists() else str(f)

    log("Loading 3 adapters...")
    model = PeftModel.from_pretrained(
        base_model, adapter_path("math"), adapter_name="math", is_trainable=False,
    )
    model.load_adapter(adapter_path("code"), adapter_name="code")
    model.load_adapter(adapter_path("science"), adapter_name="science")
    model.eval()
    log(f"Adapters loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    heartbeat("loaded")

    summary = {}

    for mode in MODES:
        log(f"\n========== MODE = {mode} ==========")
        passed = 0
        for i, problem in enumerate(ds):
            heartbeat("running", mode=mode, idx=i+1, total=len(ds), passed=passed)
            t0 = time.time()
            try:
                prompt_text = build_prompt(tok, problem)
                output, swap_count = generate_one(model, base_model, tok, prompt_text, mode)
                code = extract_code(output, problem["prompt"])
                ok, err = run_test(code, problem["test"], problem["entry_point"])
                if ok:
                    passed += 1
                row = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "task_id": problem["task_id"], "mode": mode,
                    "passed": ok, "error": err[:300] if err else "",
                    "swap_count": swap_count,
                    "elapsed_s": round(time.time() - t0, 2),
                    "output_len": len(output),
                    "code_len": len(code),
                    # Save raw outputs so we can re-score offline if needed
                    "raw_output": output[:2000],
                    "code_tested": code[:2000],
                }
                save_row(mode, row)
                log(f"[{mode}] {i+1}/164 {problem['task_id']} pass={ok} elapsed={time.time()-t0:.1f}s passed_so_far={passed}")
            except torch.cuda.OutOfMemoryError as e:
                log(f"[{mode}] {i+1}/164 OOM: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                save_row(mode, {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "task_id": problem["task_id"], "mode": mode,
                    "passed": False, "error": "OOM", "elapsed_s": time.time() - t0,
                })
            except Exception as e:
                log(f"[{mode}] {i+1}/164 EXC: {e}")
                save_row(mode, {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "task_id": problem["task_id"], "mode": mode,
                    "passed": False, "error": str(e)[:200], "elapsed_s": time.time() - t0,
                })

            torch.cuda.empty_cache()

        summary[mode] = {"passed": passed, "total": len(ds), "pass_at_1": passed / len(ds)}
        log(f"\n>>> {mode}: pass@1 = {passed}/{len(ds)} = {passed/len(ds):.1%}")
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)

    heartbeat("done", total=len(ds))
    log("=== HumanEval n=164 complete ===")


if __name__ == "__main__":
    main()
