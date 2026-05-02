#!/usr/bin/env python3
"""
Cold-Swap Latency Evaluator
Measures SSD-to-VRAM throughput by physically loading and unloading adapters 
during Token-Level Routing to establish baseline latency for low-RAM devices.
"""
import sys, gc, json, logging, re, time, os, tempfile, subprocess
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(str(RESULTS_DIR/"cold_swap.log"))])
log = logging.getLogger("cold_swap")

log.info("Loading Tok & Base Model (4bit)...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True)
base_model.eval()

# We wrap the model in Peft WITHOUT loading adapters yet to test cold swapping
log.info("Wrapping in empty PEFT model for Cold Swaps...")
try:
    dummy_ad = str(ADAPTER_BASE/"math"/"best")
    model = PeftModel.from_pretrained(base_model, dummy_ad, adapter_name="default", is_trainable=False)
except Exception:
    dummy_ad = str(ADAPTER_BASE/"math"/"final")
    model = PeftModel.from_pretrained(base_model, dummy_ad, adapter_name="default", is_trainable=False)
model.eval()

# Heuristic from the original framework
def heuristic_router(decoded_text: str) -> str:
    text = decoded_text.lower()
    if bool(re.search(r'```(?:python)?|def |import |class |    \w+', text)): return "math"
    if bool(re.search(r'\\\[|\\\]|\$\$|\\frac|\\sqrt|\\int|\d+[\+\-\*\/]\d+', text)): return "code"
    return "code"

class ColdSwapRouterLogitsProcessor(LogitsProcessor):
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.current_adapter = "code"
        self.cold_swap_times = []
        
        # Load initial adapter from disk
        path = str(ADAPTER_BASE/"code"/"best" if (ADAPTER_BASE/"code"/"best").exists() else ADAPTER_BASE/"code"/"final")
        self.model.load_adapter(path, adapter_name="code")
        self.model.set_adapter("code")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] % 10 == 0:
            context = self.tok.decode(input_ids[0][-50:], skip_special_tokens=True)
            new_ad = heuristic_router(context)
            if new_ad != self.current_adapter:
                # COLD SWAP PROFILING
                t0 = time.time()
                
                # Delete old adapter from VRAM memory
                try:
                    self.model.delete_adapter(self.current_adapter)
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
                
                # Load new adapter strictly from disk
                path = str(ADAPTER_BASE/new_ad/"best" if (ADAPTER_BASE/new_ad/"best").exists() else ADAPTER_BASE/new_ad/"final")
                self.model.load_adapter(path, adapter_name=new_ad)
                self.model.set_adapter(new_ad)
                
                torch.cuda.synchronize()
                t1 = time.time()
                
                self.cold_swap_times.append(t1 - t0)
                self.current_adapter = new_ad
        return scores

def generate_cold_swap(prompt: str, max_new=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    import sys
    model_module = sys.modules[base_model.__class__.__module__]
    HybridMambaAttentionDynamicCache = getattr(model_module, 'HybridMambaAttentionDynamicCache')
    past_key_values = HybridMambaAttentionDynamicCache(
        base_model.config, batch_size=inputs["input_ids"].shape[0], dtype=torch.bfloat16, device=model.device
    )
    
    processor = ColdSwapRouterLogitsProcessor(model, tok)
    t_start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
            use_cache=True, past_key_values=past_key_values, logits_processor=LogitsProcessorList([processor])
        )
    t_end = time.time()
    
    return {
        "response": tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True),
        "total_generation_time": t_end - t_start,
        "cold_swap_times": processor.cold_swap_times
    }

def format_prompt(sys_msg, user):
    return tok.apply_chat_template([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

LATENCY_RESULTS = {"arc": [], "humaneval": [], "math500": [], "mbpp": []}
SAMPLES_PER_DS = 10

log.info("Starting SSD Cold Swap Profiling (10 Samples per Dataset)...")
from datasets import load_dataset

# 1. ARC
ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(SAMPLES_PER_DS))
for ex in tqdm(ds_arc, desc="ARC Cold-Swap"):
    choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
    p = format_prompt("Answer with EXACTLY ONE letter A, B, C, or D.", f"{ex['question']}\n\n{choices}")
    res = generate_cold_swap(p, max_new=16)
    LATENCY_RESULTS["arc"].extend(res["cold_swap_times"])

# 2. HumanEval
ds_he = load_dataset("openai/openai_humaneval", split="test").select(range(SAMPLES_PER_DS))
for ex in tqdm(ds_he, desc="HumanEval Cold-Swap"):
    p = format_prompt("Complete the Python function. Output ONLY the code.", f"Complete this function:\n```python\n{ex['prompt']}\n```")
    res = generate_cold_swap(p, max_new=512)
    LATENCY_RESULTS["humaneval"].extend(res["cold_swap_times"])

# 3. MATH
try: ds_m = load_dataset("HuggingFaceH4/MATH-500", split="test")
except: ds_m = load_dataset("lighteval/MATH", split="test")
ds_m = ds_m.select(range(SAMPLES_PER_DS))
for ex in tqdm(ds_m, desc="MATH Cold-Swap"):
    p = format_prompt("Solve this math problem. Put your final answer in \\boxed{}.", ex.get("problem", ex.get("question")))
    res = generate_cold_swap(p, max_new=512)
    LATENCY_RESULTS["math500"].extend(res["cold_swap_times"])

# 4. MBPP
ds_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test").select(range(SAMPLES_PER_DS))
for ex in tqdm(ds_mbpp, desc="MBPP Cold-Swap"):
    p = format_prompt("Write a Python function to solve the task. Output ONLY the code.", ex["prompt"])
    res = generate_cold_swap(p, max_new=512)
    LATENCY_RESULTS["mbpp"].extend(res["cold_swap_times"])


all_times = LATENCY_RESULTS["arc"] + LATENCY_RESULTS["humaneval"] + LATENCY_RESULTS["math500"] + LATENCY_RESULTS["mbpp"]
if all_times:
    avg_latency = sum(all_times) / len(all_times)
    max_latency = max(all_times)
    log.info(f"\nFinal Cold-Swap Statistics:")
    log.info(f"Total Swaps Initiated: {len(all_times)}")
    log.info(f"Average NVMe SSD-to-VRAM Latency: {avg_latency*1000:.2f} ms")
    log.info(f"Worst-Case Swap Latency: {max_latency*1000:.2f} ms")
else:
    log.info("No adapter swaps were triggered during these 40 questions.")

with open(RESULTS_DIR/"cold_swap_metrics.json", "w") as f:
    json.dump({
        "total_swaps": len(all_times),
        "average_latency_sec": sum(all_times)/len(all_times) if all_times else 0,
        "max_latency_sec": max(all_times) if all_times else 0,
        "raw_data": LATENCY_RESULTS
    }, f, indent=2)

log.info("Cold-Swap Analysis Complete. Metrics saved.")
