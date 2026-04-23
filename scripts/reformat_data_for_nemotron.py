"""
Reformat existing LoRI-MoE training data for Nemotron's chat template.
Reads from data/lori_moe/*.jsonl, writes to data/nemotron/*.jsonl

Usage:
    .venv/bin/python scripts/reformat_data_for_nemotron.py
"""
import json
import re
import sys
from pathlib import Path

INPUT_DIR = Path("/home/learner/Desktop/mewtwo/data/lori_moe")
OUTPUT_DIR = Path("/home/learner/Desktop/mewtwo/data/nemotron")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try multiple model paths
MODEL_CANDIDATES = [
    "/home/learner/Desktop/mewtwo/models/nemotron",
]
# Also check subdirs
base = Path("/home/learner/Desktop/mewtwo/models/nemotron")
if base.exists():
    for sub in base.iterdir():
        if sub.is_dir() and (sub / "config.json").exists():
            MODEL_CANDIDATES.insert(0, str(sub))

MODEL_PATH = None
for c in MODEL_CANDIDATES:
    if Path(c).exists() and (Path(c) / "config.json").exists():
        MODEL_PATH = c
        break

tokenizer = None
if MODEL_PATH:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"Loaded tokenizer from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")

SYSTEM_PROMPTS = {
    "math": "You are a precise mathematics expert. Solve problems step-by-step, showing all work clearly. Always verify your final answer.",
    "code": "You are an expert programmer. Write clean, correct, well-documented code. Explain your approach before implementation.",
    "science": "You are a rigorous scientist. Answer questions with precise scientific reasoning, citing relevant principles and evidence.",
}


def extract_qa_from_qwen_format(text):
    """Extract user question and assistant answer from Qwen chat format."""
    # Pattern: <|im_start|>role\ncontent<|im_end|>
    user_msg = ""
    assistant_msg = ""
    
    parts = text.split("<|im_start|>")
    for part in parts:
        part = part.strip()
        if part.startswith("user\n"):
            content = part[len("user\n"):]
            content = content.replace("<|im_end|>", "").strip()
            user_msg = content
        elif part.startswith("assistant\n"):
            content = part[len("assistant\n"):]
            content = content.replace("<|im_end|>", "").strip()
            assistant_msg = content
    
    return user_msg, assistant_msg


def format_for_nemotron(system, user, assistant):
    """Format using Nemotron's chat template, with fallback."""
    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    
    # Fallback: generic format that works with most models
    return f"<|begin_of_text|>System: {system}\n\nUser: {user}\n\nAssistant: {assistant}<|end_of_text|>"


total_in = 0
total_out = 0

for domain in ["math", "code", "science"]:
    input_file = INPUT_DIR / f"{domain}_train.jsonl"
    output_file = OUTPUT_DIR / f"{domain}_train.jsonl"
    
    if not input_file.exists():
        print(f"SKIP {domain}: {input_file} not found")
        continue
    
    count = 0
    skipped = 0
    
    with open(input_file) as fin, open(output_file, "w") as fout:
        for line_num, line in enumerate(fin):
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue
            
            old_text = row.get("text", "")
            if not old_text:
                skipped += 1
                continue
            
            user_q, assistant_a = extract_qa_from_qwen_format(old_text)
            
            if not user_q or not assistant_a:
                skipped += 1
                continue
            
            new_text = format_for_nemotron(
                SYSTEM_PROMPTS.get(domain, "You are a helpful assistant."),
                user_q,
                assistant_a,
            )
            
            fout.write(json.dumps({"text": new_text, "domain": domain}) + "\n")
            count += 1
            
            # Cap at 50k for math (to use full MetaMathQA), 20k for others
            domain_cap = 50000 if domain == "math" else 20000
            if count >= domain_cap:
                break
    
    total_in += count + skipped
    total_out += count
    print(f"{domain}: {count} examples reformatted ({skipped} skipped) -> {output_file}")

print(f"\nTotal: {total_out} examples written from {total_in} input rows")
print(f"Output directory: {OUTPUT_DIR}")
