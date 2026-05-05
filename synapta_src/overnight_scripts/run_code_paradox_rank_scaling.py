#!/usr/bin/env python3
"""Code Paradox at varying adapter ranks on Qwen-0.8B.

Question: Is the Code Paradox (code-adapter beats math-adapter on math)
rank-dependent? Tests ranks 8, 128, 1024 on MATH-500 (n=50).

Output: results/bfsi_swarm_extras/code_paradox_rank_scaling.json
        findings/code_paradox_rank_scaling.md
"""
import os, sys, json, re, time, datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

PROJECT = Path("/home/learner/Desktop/mewtwo")
ADAPTERS = PROJECT / "adapters" / "small_models_zoo" / "from_hf_kaggle"
OUT_DIR = PROJECT / "results" / "bfsi_swarm_extras"
LOG_FILE = PROJECT / "logs" / "swarm_8h" / "extras" / "rank_scaling.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

N = 50
MAX_NEW = 256
RANKS = [8, 128, 1024]


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def normalize(s):
    if s is None:
        return ""
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace(" ", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s.lower()


def extract_answer(text):
    if not text:
        return None
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:the answer is|answer\s*[:=])\s*(.+?)(?:[.\n]|$)", text, re.IGNORECASE)
    if m:
        return m[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def main():
    log("=== Code Paradox rank scaling on Qwen-0.8B ===")

    log("Loading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(N))
    problems = [{"problem": ex["problem"], "answer": ex["answer"]} for ex in ds]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("Loading Qwen-0.8B base in bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    log(f"Base loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    sys_msg = "You are a careful mathematical reasoner. Always put the final answer in \\boxed{}."

    def build(p):
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Solve step by step. Put final answer in \\boxed{{}}.\n\nProblem: {p['problem']}\n"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen(model, prompt_text):
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    summary = {}

    # Base (no adapter)
    log("Eval: base, no adapter")
    correct = 0
    for i, p in enumerate(problems):
        try:
            output = gen(base_model, build(p))
            pred = extract_answer(output)
            ok = normalize(pred) == normalize(p["answer"])
            if ok:
                correct += 1
            if (i + 1) % 10 == 0:
                log(f"  base {i+1}/{N} correct={correct}")
        except Exception as e:
            log(f"  base {i+1} EXC: {e}")
        torch.cuda.empty_cache()
    base_acc = correct / N
    summary["base"] = {"correct": correct, "total": N, "acc": base_acc}
    log(f"BASE: {correct}/{N} = {base_acc:.1%}")

    for rank in RANKS:
        for which in ("math", "code"):
            cfg_name = f"{which}_rank{rank}"
            adapter_path = str(ADAPTERS / f"qwen_0.8b_{which}_SFT_rank{rank}")
            if not (ADAPTERS / f"qwen_0.8b_{which}_SFT_rank{rank}").exists():
                log(f"SKIP {cfg_name}: adapter dir not found")
                continue
            log(f"\nEval: {cfg_name}")
            try:
                model = PeftModel.from_pretrained(
                    base_model, adapter_path, adapter_name=which, is_trainable=False,
                )
                model.set_adapter(which)
                model.eval()
                correct = 0
                for i, p in enumerate(problems):
                    try:
                        output = gen(model, build(p))
                        pred = extract_answer(output)
                        ok = normalize(pred) == normalize(p["answer"])
                        if ok:
                            correct += 1
                        if (i + 1) % 10 == 0:
                            log(f"  {cfg_name} {i+1}/{N} correct={correct}")
                    except Exception as e:
                        log(f"  {cfg_name} {i+1} EXC: {e}")
                    torch.cuda.empty_cache()
                acc = correct / N
                summary[cfg_name] = {"correct": correct, "total": N, "acc": acc}
                log(f"{cfg_name}: {correct}/{N} = {acc:.1%}")
                # Unload adapter
                model = base_model  # drop reference; PeftModel kept linked
                torch.cuda.empty_cache()
                # Re-load fresh base for next iteration
                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3.5-0.8B",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                base_model.eval()
            except Exception as e:
                log(f"{cfg_name} fatal: {e}")

    with open(OUT_DIR / "code_paradox_rank_scaling.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")

    # Compute deltas
    log("\nDeltas (code - math):")
    for r in RANKS:
        m = summary.get(f"math_rank{r}", {}).get("acc")
        c = summary.get(f"code_rank{r}", {}).get("acc")
        if m is not None and c is not None:
            log(f"  rank={r}: code {c:.1%} - math {m:.1%} = {(c-m)*100:+.1f} pp")

    log("Done.")


if __name__ == "__main__":
    main()
