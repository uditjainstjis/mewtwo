#!/usr/bin/env python3
"""
Format-Aware Token-Level Routing Evaluator
Tests a "routing guard" that LOCKS the current adapter when generating
syntactically sensitive code blocks (inside def/class/if/for blocks).

Hypothesis: The HumanEval regression (60% → 45%) in the original token router
was caused by mid-function adapter swaps breaking Python indentation.
A format-aware guard should recover HumanEval performance while preserving
MATH-500 gains.

This script runs a quick A/B test:
- Group A: Original paradoxical regex router (baseline)
- Group B: Format-aware guarded router (new)
Both evaluated on the same HumanEval + MATH-500 subset.
"""
import sys, gc, json, logging, re, time, tempfile, subprocess
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
RESULTS_DIR = PROJECT / "results" / "nemotron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("format_guard")

# ─── Load Model ───
log.info("Loading Nemotron-30B (4-bit) + 3 Adapters...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=BNB,
                                                   device_map="auto", trust_remote_code=True)
base_model.eval()

math_path = str(ADAPTER_BASE / "math" / ("best" if (ADAPTER_BASE / "math" / "best").exists() else "final"))
model = PeftModel.from_pretrained(base_model, math_path, adapter_name="math", is_trainable=False)
model.load_adapter(str(ADAPTER_BASE / "code" / ("best" if (ADAPTER_BASE / "code" / "best").exists() else "final")), adapter_name="code")
model.load_adapter(str(ADAPTER_BASE / "science" / ("best" if (ADAPTER_BASE / "science" / "best").exists() else "final")), adapter_name="science")
model.eval()

model_module = sys.modules[base_model.__class__.__module__]
HybridCache = getattr(model_module, 'HybridMambaAttentionDynamicCache')
log.info("✅ Model loaded.")


# ─── Router Variants ───

def heuristic_router_v1(text: str) -> str:
    """Original paradoxical regex router (baseline)."""
    t = text.lower()
    if re.search(r'```(?:python)?|def |import |class |    \w+', t):
        return "math"
    if re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', t):
        return "code"
    return "code"


class FormatAwareRouter:
    """
    Format-Aware Router with Syntax Lock Guard.
    
    When inside a code block (detected by ``` markers or active def/class),
    the router LOCKS the current adapter until the block closes.
    This prevents mid-function adapter swaps that break indentation.
    """
    def __init__(self):
        self.in_code_block = False
        self.code_block_depth = 0  # Track nested indentation
        self.lock_adapter = None
        
    def route(self, text: str) -> tuple:
        """Returns (adapter_name, is_locked)."""
        t = text.lower()
        
        # Check for code block markers
        backtick_count = text.count('```')
        if backtick_count % 2 == 1:  # Odd count means we're inside a block
            self.in_code_block = True
        elif backtick_count > 0 and backtick_count % 2 == 0:
            self.in_code_block = False
            self.lock_adapter = None
            
        # Detect active function/class definition (unclosed)
        lines = text.split('\n')
        last_lines = lines[-5:] if len(lines) >= 5 else lines
        in_function = False
        for line in last_lines:
            stripped = line.lstrip()
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except')):
                in_function = True
            # Check if we're in an indented block
            if line and line[0] == ' ' and len(line) - len(line.lstrip()) >= 4:
                in_function = True
                
        # If locked (inside code block or function), keep current adapter
        if self.in_code_block or in_function:
            if self.lock_adapter is None:
                # Lock to the adapter that's best for code: "math" (paradox)
                self.lock_adapter = "math"
            return self.lock_adapter, True
            
        # Outside code blocks: use the paradoxical router
        self.lock_adapter = None
        adapter = heuristic_router_v1(t)
        return adapter, False


class OriginalRouterProcessor(LogitsProcessor):
    """V1: Original paradoxical router, no format awareness."""
    def __init__(self, model_ref, tok_ref):
        self.model = model_ref
        self.tok = tok_ref
        self.current_adapter = "code"
        self.model.set_adapter("code")
        self.swaps = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-50:], skip_special_tokens=True)
            new_ad = heuristic_router_v1(ctx)
            if new_ad != self.current_adapter:
                self.model.set_adapter(new_ad)
                self.current_adapter = new_ad
                self.swaps += 1
        return scores


class FormatAwareRouterProcessor(LogitsProcessor):
    """V2: Format-aware router with syntax lock guard."""
    def __init__(self, model_ref, tok_ref):
        self.model = model_ref
        self.tok = tok_ref
        self.current_adapter = "code"
        self.model.set_adapter("code")
        self.router = FormatAwareRouter()
        self.swaps = 0
        self.locks = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            ctx = self.tok.decode(input_ids[0][-100:], skip_special_tokens=True)
            new_ad, locked = self.router.route(ctx)
            if locked:
                self.locks += 1
            if new_ad != self.current_adapter:
                self.model.set_adapter(new_ad)
                self.current_adapter = new_ad
                self.swaps += 1
        return scores


# ─── Generation ───

def generate(prompt, processor_cls, max_new=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    past_key_values = HybridCache(
        base_model.config, batch_size=inputs["input_ids"].shape[0],
        dtype=torch.bfloat16, device=model.device
    )
    processor = processor_cls(model, tok)
    
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, use_cache=True,
            past_key_values=past_key_values,
            logits_processor=LogitsProcessorList([processor])
        )
    t1 = time.time()
    
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del past_key_values
    gc.collect()
    torch.cuda.empty_cache()
    
    return resp, processor.swaps, t1 - t0


def format_prompt(sys_msg, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )


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


# ─── A/B Test ───

RESULTS = {"original_router": {}, "format_aware_router": {}}
SAMPLE_SIZE = 25  # 25 samples each for speed

from datasets import load_dataset

# ── HumanEval A/B ──
log.info("=" * 60)
log.info("A/B TEST: HumanEval — Format-Aware Guard vs Original Router")
log.info("=" * 60)

ds_he = load_dataset("openai/openai_humaneval", split="test").select(range(SAMPLE_SIZE))

for variant, proc_cls in [("original_router", OriginalRouterProcessor), ("format_aware_router", FormatAwareRouterProcessor)]:
    corr = tot = 0
    total_swaps = 0
    for ex in tqdm(ds_he, desc=f"HumanEval [{variant}]"):
        pcode, tcode, entry = ex["prompt"], ex["test"], ex["entry_point"]
        p = format_prompt("Complete the Python function. Output ONLY the code.", f"Complete this function:\n```python\n{pcode}\n```")
        resp, swaps, elapsed = generate(p, proc_cls, max_new=512)
        total_swaps += swaps
        
        cb = re.search(r'```(?:python)?\s*\n(.*?)```', resp, re.DOTALL)
        code = cb.group(1) if cb else resp
        if f"def {entry}" in code:
            code = code[code.index(f"def {entry}"):]
        full = code + "\n\n" + tcode + f"\n\ncheck({entry})\n"
        
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
                f.write(full); f.flush()
                if subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0:
                    corr += 1
        except:
            pass
        tot += 1
    
    score = corr / tot if tot > 0 else 0
    RESULTS[variant]["humaneval"] = {"score": score, "correct": corr, "total": tot, "total_swaps": total_swaps}
    log.info(f"  [{variant}] HumanEval: {score:.1%} ({corr}/{tot}), Swaps: {total_swaps}")

# ── MATH-500 A/B ──
log.info("=" * 60)
log.info("A/B TEST: MATH-500 — Format-Aware Guard vs Original Router")
log.info("=" * 60)

try: ds_m = load_dataset("HuggingFaceH4/MATH-500", split="test")
except: ds_m = load_dataset("lighteval/MATH", split="test")
ds_m = ds_m.select(range(SAMPLE_SIZE))

for variant, proc_cls in [("original_router", OriginalRouterProcessor), ("format_aware_router", FormatAwareRouterProcessor)]:
    corr = tot = 0
    total_swaps = 0
    for ex in tqdm(ds_m, desc=f"MATH-500 [{variant}]"):
        gold = normalize(extract_number(ex.get("solution", ex.get("answer"))))
        p = format_prompt("Solve this math problem. Put your final answer in \\boxed{}.", ex.get("problem", ex.get("question")))
        resp, swaps, elapsed = generate(p, proc_cls, max_new=512)
        total_swaps += swaps
        pred = normalize(extract_number(resp))
        if pred and gold and pred == gold:
            corr += 1
        tot += 1
    
    score = corr / tot if tot > 0 else 0
    RESULTS[variant]["math500"] = {"score": score, "correct": corr, "total": tot, "total_swaps": total_swaps}
    log.info(f"  [{variant}] MATH-500: {score:.1%} ({corr}/{tot}), Swaps: {total_swaps}")


# ── Save Results ──
with open(RESULTS_DIR / "format_guard_ab_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

log.info("=" * 60)
log.info("FINAL A/B SUMMARY")
log.info("=" * 60)
for variant in ["original_router", "format_aware_router"]:
    he = RESULTS[variant].get("humaneval", {})
    m5 = RESULTS[variant].get("math500", {})
    log.info(f"  {variant}:")
    log.info(f"    HumanEval: {he.get('score', 0):.1%} (swaps: {he.get('total_swaps', 0)})")
    log.info(f"    MATH-500:  {m5.get('score', 0):.1%} (swaps: {m5.get('total_swaps', 0)})")

he_orig = RESULTS["original_router"].get("humaneval", {}).get("score", 0)
he_guard = RESULTS["format_aware_router"].get("humaneval", {}).get("score", 0)
m5_orig = RESULTS["original_router"].get("math500", {}).get("score", 0)
m5_guard = RESULTS["format_aware_router"].get("math500", {}).get("score", 0)

log.info(f"\nDelta HumanEval: {(he_guard - he_orig)*100:+.1f}pp")
log.info(f"Delta MATH-500:  {(m5_guard - m5_orig)*100:+.1f}pp")
log.info(f"\nVerdict: {'FORMAT GUARD WINS' if he_guard > he_orig and m5_guard >= m5_orig * 0.95 else 'INCONCLUSIVE — NEEDS MORE DATA'}")
log.info("✅ Format-Aware A/B Test Complete.")
