#!/usr/bin/env python3
"""
Token-Level Dynamic Adapter Routing Evaluator
Executes autoregressive mid-sequence PEFT adapter swapping based on paradoxical domain heuristics.
Evaluates against MATH-500, HumanEval, ARC, MBPP, and Mixed-Domain sets.
"""
import sys, gc, json, logging, re, tempfile, subprocess
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
RESULTS_DIR = PROJECT / "results" / "nemotron"
TASKS_FILE = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(str(RESULTS_DIR/"token_routing.log"))])
log = logging.getLogger("router")

def update_task(task_num, msg=""):
    log.info(f"Task {task_num} status -> {msg}")

log.info("Loading Tok & Base Model (4bit)...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True)
base_model.eval()

log.info("Loading Adapters securely into VRAM Multi-PEFT System...")
try:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_BASE/"math"/"best"), adapter_name="math", is_trainable=False)
except:
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_BASE/"math"/"final"), adapter_name="math", is_trainable=False)
model.load_adapter(str(ADAPTER_BASE/"code"/"best" if (ADAPTER_BASE/"code"/"best").exists() else ADAPTER_BASE/"code"/"final"), adapter_name="code")
model.load_adapter(str(ADAPTER_BASE/"science"/"best" if (ADAPTER_BASE/"science"/"best").exists() else ADAPTER_BASE/"science"/"final"), adapter_name="science")
model.eval()
update_task("1")

def heuristic_router(decoded_text: str) -> str:
    # Paradoxical Token Router
    text = decoded_text.lower()
    if bool(re.search(r'```(?:python)?|def |import |class |    \w+', text)):
        return "math" # Math adapter structurally dominates syntactical python
    if bool(re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', text)):
        return "code" # Code adapter logically dominates mathematical progression
    return "code" # Default baseline logical generic reasoner

from transformers import LogitsProcessor, LogitsProcessorList

class TokenRouterLogitsProcessor(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.model.set_adapter(self.current_adapter)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] % 10 == 0:
            context = self.tok.decode(input_ids[0][-50:], skip_special_tokens=True)
            new_ad = heuristic_router(context)
            if new_ad != self.current_adapter:
                self.model.set_adapter(new_ad)
                self.current_adapter = new_ad
        return scores

def generate_with_token_router(prompt: str, max_new=512) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    import sys
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, 'HybridMambaAttentionDynamicCache')
    past_key_values = HybridMambaAttentionDynamicCache(
        base_model.config, 
        batch_size=inputs["input_ids"].shape[0], 
        dtype=torch.bfloat16, 
        device=model.device
    )
    
    processor = TokenRouterLogitsProcessor(model, tok)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            past_key_values=past_key_values,
            logits_processor=LogitsProcessorList([processor])
        )
        
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Evaluation Helpers
def format_prompt(sys_msg: str, user: str) -> str:
    return tok.apply_chat_template([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def extract_number(t):
    if not t: return None
    t = t.strip().replace(",", "").replace("$", "").replace("%", "")
    for rgx in [r'\\boxed\{([^}]+)\}', r'####\s*(.+?)(?:\n|$)', r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)']:
        m = re.findall(rgx, t, re.IGNORECASE)
        if m: return m[-1].strip().replace(",", "")
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    return nums[-1] if nums else None

def normalize(s):
    if not s: return ""
    s = s.strip().replace(",", "").replace("$", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except: return s.lower()

ALL_RESULTS = {}
def save_res():
    with open(RESULTS_DIR/"token_routing_results.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)

log.info("Starting Global Evauations via Token-Level Routing")

# 1. ARC
from datasets import load_dataset
ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(100))
corr = tot = 0
for ex in tqdm(ds_arc, desc="ARC Token-Routed"):
    gold = ex["answerKey"].strip().upper()
    choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
    p = format_prompt("Answer with EXACTLY ONE letter A, B, C, or D.", f"{ex['question']}\n\n{choices}")
    resp = generate_with_token_router(p, max_new=16).strip()
    pred = resp[0].upper() if resp and resp[0].upper() in "ABCD" else (re.search(r'[ABCD]', resp.upper()).group(0) if re.search(r'[ABCD]', resp.upper()) else "")
    if pred == gold: corr += 1
    tot += 1
ALL_RESULTS["arc"] = {"score": corr/tot, "correct": corr, "total": tot}
save_res()
log.info(f"ARC Score: {corr/tot:.1%}")

# 2. HumanEval
ds_he = load_dataset("openai/openai_humaneval", split="test").select(range(100))
corr = tot = 0
for ex in tqdm(ds_he, desc="HumanEval Token-Routed"):
    pcode, tcode, entry = ex["prompt"], ex["test"], ex["entry_point"]
    p = format_prompt("Complete the Python function. Output ONLY the code.", f"Complete this function:\n```python\n{pcode}\n```")
    resp = generate_with_token_router(p, max_new=512)
    cb = re.search(r'```(?:python)?\s*\n(.*?)```', resp, re.DOTALL)
    code = cb.group(1) if cb else resp
    if f"def {entry}" in code: code = code[code.index(f"def {entry}"):]
    full = code + "\n\n" + tcode + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
        f.write(full); f.flush()
        if subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0: corr+=1
    tot += 1
ALL_RESULTS["humaneval"] = {"score": corr/tot, "correct": corr, "total": tot}
save_res()
log.info(f"HumanEval Score: {corr/tot:.1%}")

# 3. MATH-500
try: ds_m = load_dataset("HuggingFaceH4/MATH-500", split="test")
except: ds_m = load_dataset("lighteval/MATH", split="test")
ds_m = ds_m.select(range(200))
corr = tot = 0
for ex in tqdm(ds_m, desc="MATH-500 Token-Routed"):
    gold = normalize(extract_number(ex.get("solution", ex.get("answer"))))
    p = format_prompt("Solve this math problem. Put your final answer in \\boxed{}.", ex.get("problem", ex.get("question")))
    pred = normalize(extract_number(generate_with_token_router(p, max_new=512)))
    if pred and gold and pred == gold: corr+=1
    tot+=1
ALL_RESULTS["math500"] = {"score": corr/tot, "correct": corr, "total": tot}
save_res()
log.info(f"MATH-500 Score: {corr/tot:.1%}")

# 4. MBPP
ds_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test").select(range(100))
corr = tot = 0
for ex in tqdm(ds_mbpp, desc="MBPP Token-Routed"):
    p = format_prompt("Write a Python function to solve the task. Output ONLY the code.", ex["prompt"])
    resp = generate_with_token_router(p, max_new=512)
    cb = re.search(r'```(?:python)?\s*\n(.*?)```', resp, re.DOTALL)
    full = (cb.group(1) if cb else resp) + "\n\n" + "\n".join(ex["test_list"]) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
        f.write(full); f.flush()
        if subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0: corr+=1
    tot+=1
ALL_RESULTS["mbpp"] = {"score": corr/tot, "correct": corr, "total": tot}
save_res()
log.info(f"MBPP Score: {corr/tot:.1%}")

# 5. Mixed Domain (from Phase 2)
MIXED_FILE = PROJECT / "data" / "mixed_domain_eval_50.json"
if MIXED_FILE.exists():
    with open(MIXED_FILE) as f: mds = json.load(f)
    tot = 0
    for ex in tqdm(mds, desc="Mixed-50 Token-Routed"):
        p = format_prompt("You are a multi-domain expert. Provide exact logical answers.", ex["query"])
        resp = generate_with_token_router(p, max_new=512)
        ALL_RESULTS[f"mixed_{ex['id']}"] = resp
        tot += 1
    # Mixed is scored elsewhere but we dump responses
    ALL_RESULTS["mixed_completed"] = tot
    save_res()

log.info("✅ OVERNIGHT SPRINT COMPLETE. Token-Level Routing finished.")
update_task("2", "Launched script")
update_task("3", "Scores being aggregated by script")
update_task("4", "Testing loop initiated")
