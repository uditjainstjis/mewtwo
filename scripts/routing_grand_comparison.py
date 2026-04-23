#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  ROUTING GRAND COMPARISON PIPELINE — Nemotron-30B
  Compares 8 routing strategies head-to-head on the same evaluation set.
═══════════════════════════════════════════════════════════════════════════

Strategies tested:
  1. No Adapter (base model only)
  2. Single Best Adapter (math-only, code-only — ablation)
  3. Regex Heuristic Router (current baseline)
  4. Format-Aware Guard (syntax lock)
  5. Neural MLP Router (trained on domain labels)
  6. Perplexity-Reactive Router (switch when ppl spikes)
  7. Entropy Gate Router (switch when output entropy exceeds threshold)
  8. Oracle Router (try all adapters per token, pick lowest loss — ceiling)

Each strategy is evaluated on:
  - MATH-500 (25 problems)
  - HumanEval (25 problems)

Metrics per strategy:
  - Accuracy / Pass@1
  - Total adapter swaps
  - Average per-token perplexity
  - Wall-clock time
"""
import sys, gc, json, logging, re, time, tempfile, subprocess, math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
RESULTS_DIR = PROJECT / "results" / "nemotron"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT / "logs" / "grand_comparison.log", mode="w"),
    ],
)
log = logging.getLogger("grand_comparison")

SAMPLE_SIZE = 25
ADAPTERS = ["math", "code", "science"]
DOMAIN_MAP = {0: "math", 1: "code", 2: "science"}

# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

log.info("=" * 70)
log.info("  ROUTING GRAND COMPARISON — Loading Nemotron-30B")
log.info("=" * 70)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=BNB, device_map="auto", trust_remote_code=True
)
base_model.eval()

math_path = str(ADAPTER_BASE / "math" / ("best" if (ADAPTER_BASE/"math"/"best").exists() else "final"))
model = PeftModel.from_pretrained(base_model, math_path, adapter_name="math", is_trainable=False)
model.load_adapter(str(ADAPTER_BASE/"code"/("best" if (ADAPTER_BASE/"code"/"best").exists() else "final")), adapter_name="code")
model.load_adapter(str(ADAPTER_BASE/"science"/("best" if (ADAPTER_BASE/"science"/"best").exists() else "final")), adapter_name="science")
model.eval()

HybridCache = getattr(sys.modules[base_model.__class__.__module__], "HybridMambaAttentionDynamicCache")
log.info("✅ Model loaded with 3 adapters in VRAM.")

# Load Neural Router
from scripts.train_neural_router_v2 import SimpleNeuralRouter
ROUTER_PATH = PROJECT / "router_adapters" / "neural_mlp_router.pt"
neural_router_mlp = SimpleNeuralRouter().to("cuda")
neural_router_mlp.load_state_dict(torch.load(ROUTER_PATH))
neural_router_mlp.eval()
log.info("✅ Neural MLP Router loaded.")


# ══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def make_cache():
    return HybridCache(base_model.config, batch_size=1, dtype=torch.bfloat16, device=model.device)

def fmt(sys_msg, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )

def extract_number(t):
    if not t:
        return None
    t = t.strip().replace(",", "").replace("$", "").replace("%", "")
    for rgx in [r'\\boxed\{([^}]+)\}', r'####\s*(.+?)(?:\n|$)', r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)']:
        m = re.findall(rgx, t, re.IGNORECASE)
        if m:
            return m[-1].strip().replace(",", "")
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    return nums[-1] if nums else None

def normalize(s):
    if not s:
        return ""
    s = s.strip().replace(",", "").replace("$", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6f}".rstrip("0").rstrip(".")
    except:
        return s.lower()


# ══════════════════════════════════════════════════════════════════════════
# ROUTER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════

# --- 1. No-Op Router (base model, no adapter switching) ---
class NoAdapterRouter(LogitsProcessor):
    """Disables all adapters — pure base model."""
    def __init__(self):
        self.swaps = 0
        self.current_adapter = "none"
        self.ppl_log = []
        model.disable_adapter_layers()

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)
        return scores

    def cleanup(self):
        model.enable_adapter_layers()


# --- 2. Single Adapter Router (static, no switching) ---
class SingleAdapterRouter(LogitsProcessor):
    """Locks to one adapter for the entire generation."""
    def __init__(self, adapter_name):
        self.swaps = 0
        self.current_adapter = adapter_name
        self.ppl_log = []
        model.set_adapter(adapter_name)

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)
        return scores

    def cleanup(self):
        pass


# --- 3. Regex Heuristic Router (our existing baseline) ---
class RegexRouter(LogitsProcessor):
    """Original paradoxical regex heuristic."""
    def __init__(self):
        self.swaps = 0
        self.current_adapter = "code"
        self.ppl_log = []
        model.set_adapter("code")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 10 == 0:
            ctx = tok.decode(input_ids[0][-50:], skip_special_tokens=True).lower()
            if re.search(r'```(?:python)?|def |import |class |    \w+', ctx):
                new = "math"
            elif re.search(r'\\\\|\\frac|\\sqrt|\\int|\d+[+\-*/]\d+', ctx):
                new = "code"
            else:
                new = "code"
            if new != self.current_adapter:
                model.set_adapter(new)
                self.current_adapter = new
                self.swaps += 1
        return scores

    def cleanup(self):
        pass


# --- 4. Format-Aware Guard Router ---
class FormatGuardRouter(LogitsProcessor):
    """Syntax Lock: freezes adapter inside code blocks."""
    def __init__(self):
        self.swaps = 0
        self.current_adapter = "code"
        self.ppl_log = []
        self.locked = False
        model.set_adapter("code")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 10 == 0:
            ctx = tok.decode(input_ids[0][-100:], skip_special_tokens=True)
            # Check if we're in a code block
            if ctx.count('```') % 2 == 1:
                self.locked = True
            else:
                lines = ctx.split('\n')[-5:]
                in_func = any(l.lstrip().startswith(('def ', 'class ', 'if ', 'for '))
                              for l in lines)
                in_indent = any(l.startswith('    ') for l in lines if l.strip())
                self.locked = in_func or in_indent

            if not self.locked:
                t = ctx.lower()
                if re.search(r'```(?:python)?|def |import |class ', t):
                    new = "math"
                elif re.search(r'\\\\|\\frac|\\sqrt|\d+[+\-*/]\d+', t):
                    new = "code"
                else:
                    new = "code"
                if new != self.current_adapter:
                    model.set_adapter(new)
                    self.current_adapter = new
                    self.swaps += 1
        return scores

    def cleanup(self):
        pass


# --- 5. Neural MLP Router ---
class NeuralMLPRouter(LogitsProcessor):
    """Trained MLP on hidden-state embeddings."""
    def __init__(self):
        self.swaps = 0
        self.current_adapter = "code"
        self.ppl_log = []
        model.set_adapter("code")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 10 == 0:
            with torch.no_grad():
                embeds = base_model.backbone.embeddings(input_ids[:, -1:]).squeeze(1).float()
                logits = neural_router_mlp(embeds)
                new = DOMAIN_MAP[logits.argmax(dim=-1).item()]
            if new != self.current_adapter:
                model.set_adapter(new)
                self.current_adapter = new
                self.swaps += 1
        return scores

    def cleanup(self):
        pass


# --- 6. Perplexity-Reactive Router ---
class PerplexityReactiveRouter(LogitsProcessor):
    """
    Monitors rolling perplexity. When it spikes above threshold,
    probes all adapters for 1 token and picks the one with lowest loss.
    """
    WINDOW = 10
    SPIKE_THRESHOLD = 1.5  # spike = current_ppl > avg_ppl * threshold

    def __init__(self):
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        self.ppl_window = []
        model.set_adapter("math")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            neg_lp = -top_lp
            self.ppl_log.append(neg_lp)
            self.ppl_window.append(neg_lp)
            if len(self.ppl_window) > self.WINDOW:
                self.ppl_window.pop(0)

        # Only probe if we have enough history and ppl is spiking
        if len(self.ppl_window) >= self.WINDOW and input_ids.shape[1] % 5 == 0:
            avg_ppl = sum(self.ppl_window) / len(self.ppl_window)
            current_ppl = neg_lp

            if current_ppl > avg_ppl * self.SPIKE_THRESHOLD:
                # Perplexity spike detected — probe all adapters
                best_adapter = self.current_adapter
                best_loss = float("inf")

                for ad in ADAPTERS:
                    model.set_adapter(ad)
                    with torch.no_grad():
                        out = model(input_ids)
                        ad_logits = out.logits[:, -1, :]
                        ad_log_probs = F.log_softmax(ad_logits, dim=-1)
                        ad_loss = -ad_log_probs[0, ad_logits[0].argmax()].item()
                    if ad_loss < best_loss:
                        best_loss = ad_loss
                        best_adapter = ad

                if best_adapter != self.current_adapter:
                    model.set_adapter(best_adapter)
                    self.current_adapter = best_adapter
                    self.swaps += 1
                else:
                    model.set_adapter(self.current_adapter)

        return scores

    def cleanup(self):
        pass


# --- 7. Entropy Gate Router ---
class EntropyGateRouter(LogitsProcessor):
    """
    Monitors output distribution entropy.
    When entropy exceeds threshold, probes adapters and picks most confident.
    """
    ENTROPY_THRESHOLD = 3.0  # nats — typical vocab entropy for confident prediction is ~1-2

    def __init__(self):
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        model.set_adapter("math")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).item()
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 5 == 0 and entropy > self.ENTROPY_THRESHOLD:
            best_adapter = self.current_adapter
            lowest_entropy = entropy

            for ad in ADAPTERS:
                model.set_adapter(ad)
                with torch.no_grad():
                    out = model(input_ids)
                    ad_logits = out.logits[:, -1, :]
                    ad_lp = F.log_softmax(ad_logits, dim=-1)
                    ad_probs = torch.exp(ad_lp)
                    ad_entropy = -(ad_probs * ad_lp).sum(dim=-1).item()
                if ad_entropy < lowest_entropy:
                    lowest_entropy = ad_entropy
                    best_adapter = ad

            if best_adapter != self.current_adapter:
                model.set_adapter(best_adapter)
                self.current_adapter = best_adapter
                self.swaps += 1
            else:
                model.set_adapter(self.current_adapter)

        return scores

    def cleanup(self):
        pass


# --- 8. Oracle Router (ceiling — probes all adapters every N tokens) ---
class OracleRouter(LogitsProcessor):
    """
    Every N tokens, runs all adapters and picks the one that produces
    the highest probability for its own top prediction.
    This is the theoretical ceiling for any routing strategy.
    """
    PROBE_INTERVAL = 10

    def __init__(self):
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        model.set_adapter("math")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % self.PROBE_INTERVAL == 0:
            best_adapter = self.current_adapter
            best_confidence = -float("inf")

            for ad in ADAPTERS:
                model.set_adapter(ad)
                with torch.no_grad():
                    out = model(input_ids)
                    ad_logits = out.logits[:, -1, :]
                    ad_log_probs = F.log_softmax(ad_logits, dim=-1)
                    # Confidence = log-prob of the most likely token
                    confidence = ad_log_probs.max(dim=-1).values.item()
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_adapter = ad

            if best_adapter != self.current_adapter:
                model.set_adapter(best_adapter)
                self.current_adapter = best_adapter
                self.swaps += 1
            else:
                model.set_adapter(self.current_adapter)

        return scores

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
# GENERATION + EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════

def generate_with_router(prompt, router_instance, max_new=512):
    """Generate text using a specific router, collecting metrics."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    pv = make_cache()

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            past_key_values=pv,
            logits_processor=LogitsProcessorList([router_instance]),
        )
    elapsed = time.time() - t0

    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    avg_ppl = sum(router_instance.ppl_log) / max(len(router_instance.ppl_log), 1)

    del pv
    gc.collect()
    torch.cuda.empty_cache()

    return resp, {
        "swaps": router_instance.swaps,
        "avg_neg_logprob": round(avg_ppl, 4),
        "tokens_generated": len(router_instance.ppl_log),
        "elapsed_sec": round(elapsed, 2),
        "final_adapter": router_instance.current_adapter,
    }


def check_humaneval(prompt_code, test_code, entry_point, response):
    """Execute HumanEval test case, return pass/fail."""
    cb = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    code = cb.group(1) if cb else response
    if f"def {entry_point}" in code:
        code = code[code.index(f"def {entry_point}"):]
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
            f.write(full)
            f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
            return r.returncode == 0
    except:
        return False


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ══════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "1_no_adapter":       lambda: NoAdapterRouter(),
    "2_single_math":      lambda: SingleAdapterRouter("math"),
    "3_single_code":      lambda: SingleAdapterRouter("code"),
    "4_regex_heuristic":  lambda: RegexRouter(),
    "5_format_guard":     lambda: FormatGuardRouter(),
    "6_neural_mlp":       lambda: NeuralMLPRouter(),
    "7_ppl_reactive":     lambda: PerplexityReactiveRouter(),
    "8_entropy_gate":     lambda: EntropyGateRouter(),
    "9_oracle":           lambda: OracleRouter(),
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datasets import load_dataset

    log.info("=" * 70)
    log.info("  GRAND COMPARISON: 9 Routing Strategies × 3 Benchmarks")
    log.info("=" * 70)

    # Load datasets
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(SAMPLE_SIZE))
    ds_code = load_dataset("openai/openai_humaneval", split="test").select(range(SAMPLE_SIZE))
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(SAMPLE_SIZE))
    log.info(f"Loaded {len(ds_math)} MATH, {len(ds_code)} HumanEval, and {len(ds_arc)} ARC-Challenge problems.")

    ALL_RESULTS = {}

    for strat_name, strat_factory in STRATEGIES.items():
        log.info("─" * 70)
        log.info(f"  STRATEGY: {strat_name}")
        log.info("─" * 70)

        strat_results = {"math500": {}, "humaneval": {}, "arc_challenge": {}}

        # ── MATH-500 ──
        corr = tot = total_swaps = 0
        all_ppl = []
        all_time = []
        for ex in tqdm(ds_math, desc=f"MATH [{strat_name}]"):
            router = strat_factory()
            gold = normalize(extract_number(ex.get("solution", ex.get("answer"))))
            p = fmt("Solve this math problem step by step. Put your final answer in \\boxed{}.",
                     ex.get("problem", ex.get("question")))

            resp, metrics = generate_with_router(p, router, max_new=384)
            if hasattr(router, 'cleanup'):
                router.cleanup()

            pred = normalize(extract_number(resp))
            is_correct = pred and gold and pred == gold
            if is_correct:
                corr += 1
            tot += 1
            total_swaps += metrics["swaps"]
            all_ppl.append(metrics["avg_neg_logprob"])
            all_time.append(metrics["elapsed_sec"])

        math_score = corr / tot if tot else 0
        strat_results["math500"] = {
            "accuracy": round(math_score, 4),
            "correct": corr,
            "total": tot,
            "total_swaps": total_swaps,
            "avg_swaps": round(total_swaps / tot, 1),
            "avg_neg_logprob": round(sum(all_ppl) / len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time) / len(all_time), 2),
        }
        log.info(f"  MATH-500: {math_score:.1%} ({corr}/{tot}) | "
                 f"Swaps: {total_swaps} | AvgPPL: {strat_results['math500']['avg_neg_logprob']:.3f} | "
                 f"Time: {sum(all_time):.0f}s")

        # ── HumanEval ──
        corr = tot = total_swaps = 0
        all_ppl = []
        all_time = []
        for ex in tqdm(ds_code, desc=f"CODE [{strat_name}]"):
            router = strat_factory()
            p = fmt("Complete the Python function. Output ONLY the code.",
                     f"Complete this function:\n```python\n{ex['prompt']}\n```")

            resp, metrics = generate_with_router(p, router, max_new=512)
            if hasattr(router, 'cleanup'):
                router.cleanup()

            passed = check_humaneval(ex["prompt"], ex["test"], ex["entry_point"], resp)
            if passed:
                corr += 1
            tot += 1
            total_swaps += metrics["swaps"]
            all_ppl.append(metrics["avg_neg_logprob"])
            all_time.append(metrics["elapsed_sec"])

        code_score = corr / tot if tot else 0
        strat_results["humaneval"] = {
            "accuracy": round(code_score, 4),
            "correct": corr,
            "total": tot,
            "total_swaps": total_swaps,
            "avg_swaps": round(total_swaps / tot, 1),
            "avg_neg_logprob": round(sum(all_ppl) / len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time) / len(all_time), 2),
        }
        log.info(f"  HumanEval: {code_score:.1%} ({corr}/{tot}) | "
                 f"Swaps: {total_swaps} | AvgPPL: {strat_results['humaneval']['avg_neg_logprob']:.3f} | "
                 f"Time: {sum(all_time):.0f}s")

        # ── ARC-Challenge ──
        corr = tot = total_swaps = 0
        all_ppl = []
        all_time = []
        for ex in tqdm(ds_arc, desc=f"ARC [{strat_name}]"):
            router = strat_factory()
            choices = ex.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            opts = "\n".join([f"{l}: {t}" for l, t in zip(labels, texts)])
            
            p = fmt("Answer the multiple-choice science question. Output ONLY the correct option label (e.g., A, B, C, D, 1, 2, 3, 4).",
                     f"{ex['question']}\nOptions:\n{opts}")

            resp, metrics = generate_with_router(p, router, max_new=16)
            if hasattr(router, 'cleanup'):
                router.cleanup()

            pred = resp.strip().upper()
            target = ex["answerKey"].strip().upper()
            
            # Simple check if target is the start of prediction, or if it is isolated in the response
            if target == pred or pred.startswith(target) or f" {target} " in f" {pred} ":
                corr += 1
            tot += 1
            total_swaps += metrics["swaps"]
            all_ppl.append(metrics["avg_neg_logprob"])
            all_time.append(metrics["elapsed_sec"])

        arc_score = corr / tot if tot else 0
        strat_results["arc_challenge"] = {
            "accuracy": round(arc_score, 4),
            "correct": corr,
            "total": tot,
            "total_swaps": total_swaps,
            "avg_swaps": round(total_swaps / tot, 1),
            "avg_neg_logprob": round(sum(all_ppl) / len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time) / len(all_time), 2),
        }
        log.info(f"  ARC-Challenge: {arc_score:.1%} ({corr}/{tot}) | "
                 f"Swaps: {total_swaps} | AvgPPL: {strat_results['arc_challenge']['avg_neg_logprob']:.3f} | "
                 f"Time: {sum(all_time):.0f}s")


        ALL_RESULTS[strat_name] = strat_results

        # Save incrementally
        with open(RESULTS_DIR / "grand_comparison_results.json", "w") as f:
            json.dump(ALL_RESULTS, f, indent=2)
        log.info(f"  [saved incrementally to grand_comparison_results.json]")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════

    log.info("\n" + "=" * 110)
    log.info("  GRAND COMPARISON — FINAL RESULTS TABLE")
    log.info("=" * 110)
    log.info(f"{'Strategy':<20} {'MATH%':>6} {'HE%':>6} {'ARC%':>6} {'M-Swap':>6} {'H-Swap':>6} {'A-Swap':>6} {'M-PPL':>6} {'H-PPL':>6} {'A-PPL':>6}")
    log.info("-" * 110)

    for name, res in ALL_RESULTS.items():
        m = res.get("math500", {})
        h = res.get("humaneval", {})
        a = res.get("arc_challenge", {})
        log.info(
            f"{name[:20]:<20} "
            f"{m.get('accuracy', 0)*100:>5.1f}% "
            f"{h.get('accuracy', 0)*100:>5.1f}% "
            f"{a.get('accuracy', 0)*100:>5.1f}% "
            f"{m.get('total_swaps', 0):>6d} "
            f"{h.get('total_swaps', 0):>6d} "
            f"{a.get('total_swaps', 0):>6d} "
            f"{m.get('avg_neg_logprob', 0):>6.3f} "
            f"{h.get('avg_neg_logprob', 0):>6.3f} "
            f"{a.get('avg_neg_logprob', 0):>6.3f}"
        )

    log.info("=" * 110)
    log.info("✅ Grand Comparison Complete. Results saved to results/nemotron/grand_comparison_results.json")
