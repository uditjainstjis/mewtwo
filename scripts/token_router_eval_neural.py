#!/usr/bin/env python3
"""
Neural Token-Level Router Evaluation (Nemotron-30B)
Replaces the 'Regex Heuristic' with the trained MLP Router.
"""
import sys, gc, json, logging, re, time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
ROUTER_PATH = PROJECT / "router_adapters" / "neural_mlp_router.pt"
RESULTS_DIR = PROJECT / "results" / "nemotron"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("neural_eval")

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

# ─── Load Neural Router ───
from scripts.train_neural_router_v2 import SimpleNeuralRouter
neural_router = SimpleNeuralRouter().to(model.device)
neural_router.load_state_dict(torch.load(ROUTER_PATH))
neural_router.eval()
log.info("✅ Neural Router loaded.")

# Map class indices to domain names
DOMAIN_MAP = {0: "math", 1: "code", 2: "science"}

class NeuralRouterProcessor(LogitsProcessor):
    def __init__(self, m, t, n_router):
        self.model = m
        self.tok = t
        self.neural_router = n_router
        self.current_adapter = "code"
        self.model.set_adapter("code")
        self.swaps = 0

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            # Extract last token embedding directly using base model's embedding layer
            # (B, S) -> (B, S, D) -> (B, D)
            with torch.no_grad():
                last_token_idx = input_ids[:, -1:]
                embeds = base_model.backbone.embeddings(last_token_idx).squeeze(1).float()
                
                # Predict
                logits = self.neural_router(embeds)
                pred_idx = logits.argmax(dim=-1).item()
                new_ad = DOMAIN_MAP[pred_idx]
                
            if new_ad != self.current_adapter:
                self.model.set_adapter(new_ad)
                self.current_adapter = new_ad
                self.swaps += 1
        return scores


def gen(prompt, max_new=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    past_key_values = HybridCache(
        base_model.config, batch_size=inputs["input_ids"].shape[0],
        dtype=torch.bfloat16, device=model.device
    )
    processor = NeuralRouterProcessor(model, tok, neural_router)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, use_cache=True,
            past_key_values=past_key_values,
            logits_processor=LogitsProcessorList([processor])
        )
    
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del past_key_values
    return resp, processor.swaps

def format_prompt(sys_msg, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True
    )

def extract_number(t):
    if not t: return None
    t = t.strip()
    for rgx in [r'\\boxed\{([^}]+)\}', r'####\s*(.+?)(?:\n|$)', r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)']:
        m = re.findall(rgx, t, re.IGNORECASE)
        if m: return m[-1].strip().replace(",", "")
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    return nums[-1] if nums else None

if __name__ == "__main__":
    from datasets import load_dataset
    log.info("Starting Neural Routing Evaluation (Small Subset for Speed)...")
    
    # ── MATH-500 subset ──
    try: ds_m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except: ds_m = load_dataset("lighteval/MATH", split="test")
    ds_m = ds_m.select(range(10))
    
    corr = tot = swabs_m = 0
    for ex in tqdm(ds_m, desc="MATH-500 (Neural)"):
        gold = extract_number(ex.get("solution", ex.get("answer")))
        p = format_prompt("Solve this math problem. Put your final answer in \\boxed{}.", ex.get("problem", ex.get("question")))
        resp, swaps = gen(p, max_new=256)
        swabs_m += swaps
        pred = extract_number(resp)
        if pred == gold: corr += 1
        tot += 1
        
    log.info(f"MATH-500 Neural Score: {corr/tot:.1%} (Avg Swaps: {swabs_m/tot:.1f})")
    log.info("✅ Neural Router integration test complete.")
