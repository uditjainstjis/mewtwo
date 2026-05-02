#!/usr/bin/env python3
"""Code Paradox replication — Qwen-3.5-0.8B at n=200 (matches original Nemotron-30B sample).

Already validated at n=50 (+6.0pp). Scaling to n=200 to match the original n=200 sample
on Nemotron-30B and tighten the cross-family claim.
"""
import os, sys, gc, json, time, re, traceback, datetime
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
OVERNIGHT = PROJECT / "overnight_run"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

ADAPTER_BASE = PROJECT / "hf_kaggle_opensource" / "outputs"
QA_DIR = OVERNIGHT / "qa_pairs"
HB_FILE = OVERNIGHT / "gpu_jobs" / "heartbeats" / "code_paradox_n200.txt"
LOG_FILE = OVERNIGHT / "gpu_jobs" / "logs" / "code_paradox_n200.log"
SUMMARY_FILE = QA_DIR / "code_paradox_qwen_n200_summary.json"
QA_DIR.mkdir(parents=True, exist_ok=True)
HB_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

N = 200
MAX_NEW = 384

CFG = {
    "key": "qwen_0.8b_n200",
    "hf_id": "Qwen/Qwen3.5-0.8B",
    "math_adapter": str(ADAPTER_BASE / "qwen_0.8b_math_SFT_rank128"),
    "code_adapter": str(ADAPTER_BASE / "qwen_0.8b_code_SFT_rank128"),
}


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def heartbeat(state, **kw):
    payload = {"ts": datetime.datetime.utcnow().isoformat(), "state": state,
               "vram_used_mb": int(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
               **kw}
    with open(HB_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def save_row(config, row):
    out = QA_DIR / f"code_paradox_{CFG['key']}_{config}.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def normalize_answer(s):
    if s is None: return ""
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\\frac\{(\-?\d+)\}\{(\-?\d+)\}",
               lambda m: f"{int(m.group(1))/int(m.group(2)):.6f}", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace(" ", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except:
        return s.lower()


def extract_answer(text):
    if not text: return None
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m: return m[-1].strip()
    m = re.findall(r"(?:the answer is|answer\s*[:=])\s*(.+?)(?:[.\n]|$)", text, re.IGNORECASE)
    if m: return m[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def score(prediction, gold):
    return normalize_answer(prediction) == normalize_answer(gold)


def main():
    log(f"=== Qwen-0.8B Code Paradox at n={N} starting ===")
    heartbeat("starting")

    for cfg in ("base", "math_adapter", "code_adapter"):
        f = QA_DIR / f"code_paradox_{CFG['key']}_{cfg}.jsonl"
        if f.exists(): f.unlink()

    log("Loading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [{"problem": ex["problem"], "answer": ex["answer"]} for ex in ds.select(range(N))]
    log(f"{len(problems)} problems loaded.")

    log("Loading Qwen-0.8B...")
    tok = AutoTokenizer.from_pretrained(CFG["hf_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        CFG["hf_id"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base_model.eval()
    log(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    sys_msg = "You are a careful mathematical reasoner. Always put the final answer in \\boxed{}."

    def build(p):
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Solve the problem step by step. Put your final numeric answer in \\boxed{{}}.\n\nProblem: {p['problem']}\n"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen(model, prompt_text):
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                  pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    summary = {}

    # Base
    log(f"\n--- Config: base ---")
    correct = 0
    for i, p in enumerate(problems):
        heartbeat("running", config="base", idx=i+1, total=N, correct=correct)
        try:
            output = gen(base_model, build(p))
            pred = extract_answer(output)
            ok = score(pred, p["answer"])
            if ok: correct += 1
            save_row("base", {"idx": i, "config": "base", "gold": p["answer"], "pred": pred, "passed": ok, "output_len": len(output)})
            if (i+1) % 20 == 0:
                log(f"  [base] {i+1}/{N} correct={correct} ({correct/(i+1):.1%})")
        except Exception as e:
            save_row("base", {"idx": i, "error": str(e)[:200]})
        torch.cuda.empty_cache()
    base_acc = correct / N
    log(f"[base] FINAL: {correct}/{N} = {base_acc:.1%}")
    summary["base"] = {"correct": correct, "total": N, "acc": base_acc}

    # Math + Code adapters
    log("Loading adapters...")
    model = PeftModel.from_pretrained(base_model, CFG["math_adapter"], adapter_name="math", is_trainable=False)
    model.load_adapter(CFG["code_adapter"], adapter_name="code")
    model.eval()
    log(f"Adapters loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    for which in ("math", "code"):
        cfg_name = f"{which}_adapter"
        log(f"\n--- Config: {cfg_name} ---")
        model.set_adapter(which)
        correct = 0
        for i, p in enumerate(problems):
            heartbeat("running", config=cfg_name, idx=i+1, total=N, correct=correct)
            try:
                output = gen(model, build(p))
                pred = extract_answer(output)
                ok = score(pred, p["answer"])
                if ok: correct += 1
                save_row(cfg_name, {"idx": i, "config": cfg_name, "gold": p["answer"], "pred": pred, "passed": ok, "output_len": len(output)})
                if (i+1) % 20 == 0:
                    log(f"  [{cfg_name}] {i+1}/{N} correct={correct} ({correct/(i+1):.1%})")
            except Exception as e:
                save_row(cfg_name, {"idx": i, "error": str(e)[:200]})
            torch.cuda.empty_cache()
        acc = correct / N
        log(f"[{cfg_name}] FINAL: {correct}/{N} = {acc:.1%}")
        summary[cfg_name] = {"correct": correct, "total": N, "acc": acc}
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)

    log(f"\n=== Summary ===")
    log(json.dumps(summary, indent=2))
    delta = summary["code_adapter"]["acc"] - summary["math_adapter"]["acc"]
    log(f"Δ (code - math) = {delta:+.1%}")
    log(f"Code Paradox at n={N}: {'REPLICATES' if delta > 0 else 'DOES NOT REPLICATE'}")

    heartbeat("done")
    log("=== Qwen-0.8B Code Paradox n=200 complete ===")


if __name__ == "__main__":
    main()
