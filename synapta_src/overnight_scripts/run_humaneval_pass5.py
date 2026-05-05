#!/usr/bin/env python3
"""HumanEval pass@5 on Nemotron-30B (n=82, half the set).

Sampling-based: 5 attempts per problem with temperature=0.8.
pass@5 = problem passes if any of 5 attempts pass the test cases.

Scope: n=82 (half of 164) × 5 samples × 2 modes = 820 generations.
Estimated time: ~3 hours on RTX 5090 4-bit.

Output:
  results/bfsi_swarm_extras/humaneval_pass5_results.jsonl
  results/bfsi_swarm_extras/humaneval_pass5_summary.json
"""
import os, sys, json, re, time, datetime, tempfile, subprocess
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from datasets import load_dataset

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
OUT_DIR = PROJECT / "results" / "bfsi_swarm_extras"
LOG_FILE = PROJECT / "logs" / "swarm_8h" / "extras" / "humaneval_pass5.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PROBLEMS = 82  # Half of HumanEval
N_SAMPLES = 5    # pass@5
TEMPERATURE = 0.8
MAX_NEW = 512


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def heuristic_router(decoded_text):
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
            target = "math" if (ticks % 2 == 1) else heuristic_router(ctx)
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


def extract_code_robust(raw_output, prompt, entry_point):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if m:
        body = m.group(1).rstrip()
        if f"def {entry_point}" in body:
            header_lines = []
            for line in prompt.split("\n"):
                if line.strip().startswith(("def ", "class ")):
                    break
                header_lines.append(line)
            header = "\n".join(header_lines).strip()
            return (header + "\n\n" + body) if header else body
        return prompt + "\n" + body
    def_match = re.search(rf"^def\s+{re.escape(entry_point)}\b", raw_output, re.MULTILINE)
    if def_match:
        body = raw_output[def_match.start():].rstrip()
        header_lines = []
        for line in prompt.split("\n"):
            if line.strip().startswith(("def ", "class ")):
                break
            header_lines.append(line)
        header = "\n".join(header_lines).strip()
        return (header + "\n\n" + body) if header else body
    return prompt + "\n    pass\n"


def run_test(code, test_code, entry_point, timeout=10):
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full)
            fpath = f.name
        try:
            r = subprocess.run(
                ["/home/learner/Desktop/mewtwo/.venv/bin/python", fpath],
                timeout=timeout, capture_output=True, text=True,
            )
            return r.returncode == 0
        finally:
            try: os.unlink(fpath)
            except: pass
    except Exception:
        return False


def generate(model, base_model, tok, prompt_text, mode, max_new=MAX_NEW):
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    cache = get_hybrid_cache(base_model, model, batch_size=inputs["input_ids"].shape[0])
    swap_count = 0
    if mode == "base":
        with model.disable_adapter():
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new,
                    do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                    pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                )
    else:
        proc = FormatGuardProcessor(model, tok)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=cache,
                logits_processor=LogitsProcessorList([proc]),
            )
        swap_count = proc.swap_count
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded, swap_count


def main():
    log(f"=== HumanEval pass@{N_SAMPLES} on n={N_PROBLEMS} ===")
    log(f"Estimated total: {N_PROBLEMS * N_SAMPLES * 2} generations")

    ds = load_dataset("openai_humaneval", split="test").select(range(N_PROBLEMS))
    log(f"Loaded {len(ds)} problems")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    base_model.eval()

    def adapter_path(name):
        b = ADAPTER_BASE / name / "best"
        f = ADAPTER_BASE / name / "final"
        return str(b) if b.exists() else str(f)

    model = PeftModel.from_pretrained(base_model, adapter_path("math"), adapter_name="math", is_trainable=False)
    model.load_adapter(adapter_path("code"), adapter_name="code")
    model.load_adapter(adapter_path("science"), adapter_name="science")
    model.eval()
    log(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    sys_msg = "You are an expert Python programmer. Complete the function precisely. Output only the function code."

    out_file = OUT_DIR / "humaneval_pass5_results.jsonl"
    if out_file.exists():
        out_file.unlink()

    results = {"base": [], "format_guard": []}
    for mode in ("base", "format_guard"):
        log(f"\n--- Mode: {mode} ---")
        for i, ex in enumerate(ds):
            t0 = time.time()
            user = f"Complete this function. Output only the function code:\n\n```python\n{ex['prompt']}\n```"
            msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            attempts = []
            any_passed = False
            for s in range(N_SAMPLES):
                try:
                    output, swap_count = generate(model, base_model, tok, prompt_text, mode)
                    code = extract_code_robust(output, ex["prompt"], ex["entry_point"])
                    passed = run_test(code, ex["test"], ex["entry_point"])
                    attempts.append({"sample": s, "passed": passed, "raw_output": output[:1500], "code_tested": code[:1500]})
                    if passed:
                        any_passed = True
                except Exception as e:
                    attempts.append({"sample": s, "error": str(e)[:200]})
                torch.cuda.empty_cache()

            row = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "task_id": ex["task_id"], "mode": mode,
                f"pass_at_{N_SAMPLES}": any_passed,
                "n_passed_of_5": sum(1 for a in attempts if a.get("passed")),
                "elapsed_s": round(time.time() - t0, 2),
                "attempts": attempts,
            }
            results[mode].append(row)
            with open(out_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if (i + 1) % 5 == 0:
                pa5 = sum(1 for r in results[mode] if r.get(f"pass_at_{N_SAMPLES}"))
                log(f"  [{mode}] {i+1}/{N_PROBLEMS} pass@{N_SAMPLES}_so_far={pa5}")

    summary = {}
    for mode, rows in results.items():
        pa5 = sum(1 for r in rows if r.get(f"pass_at_{N_SAMPLES}"))
        n = len(rows)
        summary[mode] = {f"pass_at_{N_SAMPLES}": pa5, "total": n, "rate": pa5 / max(n, 1)}
    with open(OUT_DIR / "humaneval_pass5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n=== SUMMARY ===")
    log(f"base pass@{N_SAMPLES}: {summary['base']['rate']:.1%}")
    log(f"FG pass@{N_SAMPLES}: {summary['format_guard']['rate']:.1%}")
    log(f"DELTA: {(summary['format_guard']['rate'] - summary['base']['rate'])*100:+.1f} pp")


if __name__ == "__main__":
    main()
