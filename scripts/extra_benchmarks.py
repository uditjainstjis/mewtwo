#!/usr/bin/env python3
"""
GSM8K + MMLU Additional Benchmarks (Lightweight)
Runs against the already-loaded model to maximize GPU efficiency.
Uses token-level routing for both benchmarks.
"""
import sys, gc, json, logging, re, time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import LogitsProcessor, LogitsProcessorList

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))
RESULTS_DIR = PROJECT / "results" / "nemotron"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extra_benchmarks")

# Reuse the model from the demo server if possible, otherwise load fresh
try:
    from src.demo.server import model, tok, base_model, HybridCache
    log.info("Re-using model from demo server module...")
except:
    log.info("Loading model fresh for benchmarks...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    MODEL_PATH = str(PROJECT / "models" / "nemotron")
    ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
    
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=BNB,
                                                       device_map="auto", trust_remote_code=True)
    base_model.eval()
    math_path = str(ADAPTER_BASE / "math" / ("best" if (ADAPTER_BASE/"math"/"best").exists() else "final"))
    model = PeftModel.from_pretrained(base_model, math_path, adapter_name="math", is_trainable=False)
    model.load_adapter(str(ADAPTER_BASE/"code"/("best" if (ADAPTER_BASE/"code"/"best").exists() else "final")), adapter_name="code")
    model.load_adapter(str(ADAPTER_BASE/"science"/("best" if (ADAPTER_BASE/"science"/"best").exists() else "final")), adapter_name="science")
    model.eval()
    model_module = sys.modules[base_model.__class__.__module__]
    HybridCache = getattr(model_module, 'HybridMambaAttentionDynamicCache')


def heuristic_router(text):
    t = text.lower()
    if re.search(r'```(?:python)?|def |import |class |    \w+', t): return "math"
    if re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', t): return "code"
    return "code"

class RouterProcessor(LogitsProcessor):
    def __init__(self, m, t):
        self.model, self.tok = m, t
        self.current = "code"; self.model.set_adapter("code")
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-50:], skip_special_tokens=True)
            new = heuristic_router(ctx)
            if new != self.current:
                self.model.set_adapter(new); self.current = new
        return scores

def gen(prompt, max_new=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    pv = HybridCache(base_model.config, batch_size=inputs["input_ids"].shape[0],
                     dtype=torch.bfloat16, device=model.device)
    proc = RouterProcessor(model, tok)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True,
                             past_key_values=pv, logits_processor=LogitsProcessorList([proc]))
    r = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del pv; gc.collect(); torch.cuda.empty_cache()
    return r

def fmt(sys_msg, user):
    return tok.apply_chat_template([{"role":"system","content":sys_msg},{"role":"user","content":user}],
                                   tokenize=False, add_generation_prompt=True)

def extract_number(t):
    if not t: return None
    t = t.strip().replace(",","").replace("$","").replace("%","")
    for rgx in [r'\\boxed\{([^}]+)\}', r'####\s*(.+?)(?:\n|$)', r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)']:
        m = re.findall(rgx, t, re.IGNORECASE)
        if m: return m[-1].strip().replace(",","")
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    return nums[-1] if nums else None

def normalize(s):
    if not s: return ""
    s = s.strip().replace(",","").replace("$","")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except: return s.lower()

ALL = {}
from datasets import load_dataset

# ── GSM8K ──
log.info("Starting GSM8K Token-Routed Evaluation (100 samples)...")
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(100))
corr = tot = 0
for ex in tqdm(ds, desc="GSM8K Token-Routed"):
    gold = normalize(extract_number(ex["answer"]))
    p = fmt("Solve this math problem step by step. Put your final numerical answer after ####.", ex["question"])
    pred = normalize(extract_number(gen(p, max_new=512)))
    if pred and gold and pred == gold: corr += 1
    tot += 1
ALL["gsm8k"] = {"score": corr/tot, "correct": corr, "total": tot}
log.info(f"GSM8K: {corr/tot:.1%} ({corr}/{tot})")

# ── MMLU (Abstract Algebra subset) ──
log.info("Starting MMLU Token-Routed Evaluation (100 samples)...")
try:
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.select(range(100))
except:
    try:
        ds = load_dataset("lukaemon/mmlu", "abstract_algebra", split="test")
    except:
        ds = None
        log.warning("MMLU dataset not available, skipping.")

if ds is not None:
    corr = tot = 0
    for ex in tqdm(ds, desc="MMLU Token-Routed"):
        q = ex.get("question", ex.get("input", ""))
        choices_text = ""
        for key in ["A", "B", "C", "D"]:
            val = ex.get(key) or ex.get("choices", ["","","",""])[["A","B","C","D"].index(key)] if "choices" in ex else ""
            choices_text += f"\n{key}. {val}"
        gold = str(ex.get("answer", ex.get("target", ""))).strip().upper()
        if gold in "0123": gold = "ABCD"[int(gold)]
        
        p = fmt("Answer with EXACTLY ONE letter A, B, C, or D.", f"{q}{choices_text}")
        resp = gen(p, max_new=16).strip()
        pred = resp[0].upper() if resp and resp[0].upper() in "ABCD" else ""
        if not pred:
            m = re.search(r'[ABCD]', resp.upper())
            if m: pred = m.group(0)
        if pred == gold: corr += 1
        tot += 1
    ALL["mmlu"] = {"score": corr/tot, "correct": corr, "total": tot}
    log.info(f"MMLU: {corr/tot:.1%} ({corr}/{tot})")

with open(RESULTS_DIR / "extra_benchmarks.json", "w") as f:
    json.dump(ALL, f, indent=2)

log.info("✅ Extra benchmarks complete. Results saved.")
