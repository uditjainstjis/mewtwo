#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  ROUTING GRAND COMPARISON v2 — FORENSIC EDITION
  Fixed: repetition_penalty, full generation logging, multi-interval sweep
═══════════════════════════════════════════════════════════════════════════

Changes from v1:
  - repetition_penalty=1.2 on ALL strategies
  - Every generation saved to results/nemotron/generation_traces/<strategy>.jsonl
  - Repetition detection: flags if 30-char substring repeats 3+ times
  - Multiple swap intervals tested for dynamic routers (10, 20, 40)
  - Quality checkpoint every 10 generations with log review
"""
import sys, gc, json, logging, re, time, tempfile, subprocess, math as pymath
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
TRACES_DIR = RESULTS_DIR / "generation_traces"
LOG_DIR = PROJECT / "logs" / "strategy_logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT / "logs" / "grand_comparison_v2.log", mode="w"),
    ],
)
log = logging.getLogger("grand_v2")

SAMPLE_SIZE = 25
ADAPTERS = ["math", "code", "science"]
DOMAIN_MAP = {0: "math", 1: "code", 2: "science"}
REPETITION_PENALTY = 1.2


# ══════════════════════════════════════════════════════════════════════════
# REPETITION DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_repetition(text, min_len=30, min_repeats=3):
    """Check if any substring of min_len chars repeats min_repeats+ times."""
    if len(text) < min_len * min_repeats:
        return False, ""
    # Check sliding windows
    for window_size in [min_len, 50, 80]:
        for i in range(len(text) - window_size):
            chunk = text[i:i + window_size]
            if chunk.strip() == "":
                continue
            count = text.count(chunk)
            if count >= min_repeats:
                return True, chunk[:60]
    return False, ""


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

log.info("=" * 70)
log.info("  ROUTING GRAND COMPARISON v2 — FORENSIC EDITION")
log.info("  repetition_penalty=%.1f | Full trace logging | Multi-interval", REPETITION_PENALTY)
log.info("=" * 70)

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

BNB = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
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
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def make_cache():
    return HybridCache(base_model.config, batch_size=1, dtype=torch.bfloat16, device=model.device)

def fmt(sys_msg, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
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

def check_humaneval(prompt_code, test_code, entry_point, response):
    cb = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    code = cb.group(1) if cb else response
    if f"def {entry_point}" in code:
        code = code[code.index(f"def {entry_point}"):]
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as f:
            f.write(full); f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
            return r.returncode == 0
    except:
        return False


# ══════════════════════════════════════════════════════════════════════════
# ROUTER IMPLEMENTATIONS (all with configurable interval)
# ══════════════════════════════════════════════════════════════════════════

class NoAdapterRouter(LogitsProcessor):
    def __init__(self, **kwargs):
        self.swaps = 0; self.current_adapter = "none"; self.ppl_log = []
        model.disable_adapter_layers()
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        return scores
    def cleanup(self):
        model.enable_adapter_layers()


class SingleAdapterRouter(LogitsProcessor):
    def __init__(self, adapter_name="math", **kwargs):
        self.swaps = 0; self.current_adapter = adapter_name; self.ppl_log = []
        model.set_adapter(adapter_name)
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        return scores
    def cleanup(self): pass


class NeuralMLPRouter(LogitsProcessor):
    def __init__(self, interval=10, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "math"; self.ppl_log = []
        model.set_adapter("math")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        if input_ids.shape[1] % self.interval == 0:
            with torch.no_grad():
                embeds = base_model.backbone.embeddings(input_ids[:, -1:]).squeeze(1).float()
                logits = neural_router_mlp(embeds)
                new = DOMAIN_MAP[logits.argmax(dim=-1).item()]
            if new != self.current_adapter:
                model.set_adapter(new); self.current_adapter = new; self.swaps += 1
        return scores
    def cleanup(self): pass


class RegexRouter(LogitsProcessor):
    def __init__(self, interval=10, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "code"; self.ppl_log = []
        model.set_adapter("code")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        if input_ids.shape[1] % self.interval == 0:
            ctx = tok.decode(input_ids[0][-50:], skip_special_tokens=True).lower()
            if re.search(r'```(?:python)?|def |import |class |    \w+', ctx):
                new = "math"
            elif re.search(r'\\\\|\\frac|\\sqrt|\\int|\d+[+\-*/]\d+', ctx):
                new = "code"
            else:
                new = "code"
            if new != self.current_adapter:
                model.set_adapter(new); self.current_adapter = new; self.swaps += 1
        return scores
    def cleanup(self): pass


class FormatGuardRouter(LogitsProcessor):
    def __init__(self, interval=10, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "code"; self.ppl_log = []; self.locked = False
        model.set_adapter("code")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        if input_ids.shape[1] % self.interval == 0:
            ctx = tok.decode(input_ids[0][-100:], skip_special_tokens=True)
            if ctx.count('```') % 2 == 1:
                self.locked = True
            else:
                lines = ctx.split('\n')[-5:]
                self.locked = any(l.lstrip().startswith(('def ', 'class ', 'if ', 'for ')) for l in lines) or \
                              any(l.startswith('    ') for l in lines if l.strip())
            if not self.locked:
                t = ctx.lower()
                if re.search(r'```(?:python)?|def |import |class ', t): new = "math"
                elif re.search(r'\\\\|\\frac|\\sqrt|\d+[+\-*/]\d+', t): new = "code"
                else: new = "code"
                if new != self.current_adapter:
                    model.set_adapter(new); self.current_adapter = new; self.swaps += 1
        return scores
    def cleanup(self): pass


class PerplexityReactiveRouter(LogitsProcessor):
    WINDOW = 10; SPIKE_THRESHOLD = 1.5
    def __init__(self, interval=5, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "math"; self.ppl_log = []; self.ppl_window = []
        model.set_adapter("math")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            neg_lp = -lp[0, scores[0].argmax()].item()
            self.ppl_log.append(neg_lp); self.ppl_window.append(neg_lp)
            if len(self.ppl_window) > self.WINDOW: self.ppl_window.pop(0)
        if len(self.ppl_window) >= self.WINDOW and input_ids.shape[1] % self.interval == 0:
            avg = sum(self.ppl_window) / len(self.ppl_window)
            if neg_lp > avg * self.SPIKE_THRESHOLD:
                best, best_loss = self.current_adapter, float("inf")
                for ad in ADAPTERS:
                    model.set_adapter(ad)
                    with torch.no_grad():
                        out = model(input_ids)
                        al = F.log_softmax(out.logits[:, -1, :], dim=-1)
                        loss = -al[0, out.logits[:, -1, :][0].argmax()].item()
                    if loss < best_loss: best_loss = loss; best = ad
                if best != self.current_adapter:
                    model.set_adapter(best); self.current_adapter = best; self.swaps += 1
                else:
                    model.set_adapter(self.current_adapter)
        return scores
    def cleanup(self): pass


class EntropyGateRouter(LogitsProcessor):
    ENTROPY_THRESHOLD = 3.0
    def __init__(self, interval=5, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "math"; self.ppl_log = []
        model.set_adapter("math")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            probs = torch.exp(lp)
            entropy = -(probs * lp).sum(dim=-1).item()
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        if input_ids.shape[1] % self.interval == 0 and entropy > self.ENTROPY_THRESHOLD:
            best, low_ent = self.current_adapter, entropy
            for ad in ADAPTERS:
                model.set_adapter(ad)
                with torch.no_grad():
                    out = model(input_ids)
                    al = F.log_softmax(out.logits[:, -1, :], dim=-1)
                    ae = -(torch.exp(al) * al).sum(dim=-1).item()
                if ae < low_ent: low_ent = ae; best = ad
            if best != self.current_adapter:
                model.set_adapter(best); self.current_adapter = best; self.swaps += 1
            else:
                model.set_adapter(self.current_adapter)
        return scores
    def cleanup(self): pass


class OracleRouter(LogitsProcessor):
    def __init__(self, interval=10, **kwargs):
        self.interval = interval
        self.swaps = 0; self.current_adapter = "math"; self.ppl_log = []
        model.set_adapter("math")
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            lp = F.log_softmax(scores, dim=-1)
            self.ppl_log.append(-lp[0, scores[0].argmax()].item())
        if input_ids.shape[1] % self.interval == 0:
            best, best_conf = self.current_adapter, -float("inf")
            for ad in ADAPTERS:
                model.set_adapter(ad)
                with torch.no_grad():
                    out = model(input_ids)
                    al = F.log_softmax(out.logits[:, -1, :], dim=-1)
                    conf = al.max(dim=-1).values.item()
                if conf > best_conf: best_conf = conf; best = ad
            if best != self.current_adapter:
                model.set_adapter(best); self.current_adapter = best; self.swaps += 1
            else:
                model.set_adapter(self.current_adapter)
        return scores
    def cleanup(self): pass


# ══════════════════════════════════════════════════════════════════════════
# GENERATION ENGINE (with full forensics)
# ══════════════════════════════════════════════════════════════════════════

def generate_with_router(prompt, router_instance, max_new=512):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    pv = make_cache()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, use_cache=True,
            past_key_values=pv,
            repetition_penalty=REPETITION_PENALTY,
            logits_processor=LogitsProcessorList([router_instance]),
        )
    elapsed = time.time() - t0
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    avg_ppl = sum(router_instance.ppl_log) / max(len(router_instance.ppl_log), 1)
    del pv; gc.collect(); torch.cuda.empty_cache()

    # Repetition check
    has_rep, rep_chunk = detect_repetition(resp)

    return resp, {
        "swaps": router_instance.swaps,
        "avg_neg_logprob": round(avg_ppl, 4),
        "tokens_generated": len(router_instance.ppl_log),
        "elapsed_sec": round(elapsed, 2),
        "final_adapter": router_instance.current_adapter,
        "has_repetition": has_rep,
        "repetition_sample": rep_chunk,
    }


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY (with interval variants for dynamic routers)
# ══════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    # ⚡ EXECUTING NEURAL MLP FIRST (Target Focus)
    "06_neural_mlp":         {"factory": lambda **kw: NeuralMLPRouter(**kw), "intervals": [10, 20, 40]},

    # Static (no swap interval matters)
    "01_no_adapter":         {"factory": lambda **kw: NoAdapterRouter(**kw), "intervals": [None]},
    "02_single_math":        {"factory": lambda **kw: SingleAdapterRouter("math", **kw), "intervals": [None]},
    "03_single_code":        {"factory": lambda **kw: SingleAdapterRouter("code", **kw), "intervals": [None]},

    # Heuristic (test intervals)
    "04_regex":              {"factory": lambda **kw: RegexRouter(**kw), "intervals": [10, 20, 40]},
    "05_format_guard":       {"factory": lambda **kw: FormatGuardRouter(**kw), "intervals": [10, 20, 40]},

    # Learned/Dynamic (test intervals)
    "07_ppl_reactive":       {"factory": lambda **kw: PerplexityReactiveRouter(**kw), "intervals": [10, 20, 40]},
    "08_entropy_gate":       {"factory": lambda **kw: EntropyGateRouter(**kw), "intervals": [10, 20, 40]},

    # Oracle (ceiling)
    "09_oracle":             {"factory": lambda **kw: OracleRouter(**kw), "intervals": [10, 25, 50]},
}


# ══════════════════════════════════════════════════════════════════════════
# QUALITY CHECKPOINT (runs every 10 generations)
# ══════════════════════════════════════════════════════════════════════════

def quality_checkpoint(strat_name, traces, gen_count):
    """Review last 10 generations for quality issues."""
    recent = traces[-10:]
    rep_count = sum(1 for t in recent if t.get("has_repetition"))
    empty_count = sum(1 for t in recent if len(t.get("response", "").strip()) < 10)
    avg_len = sum(len(t.get("response", "")) for t in recent) / max(len(recent), 1)

    report = (
        f"  📋 QUALITY CHECK [{strat_name}] after {gen_count} generations:\n"
        f"     Repetitions detected: {rep_count}/10\n"
        f"     Empty/trivial outputs: {empty_count}/10\n"
        f"     Avg response length: {avg_len:.0f} chars\n"
    )

    if rep_count > 0:
        report += f"     ⚠️ REPETITION WARNING: {rep_count} outputs had repetitive text!\n"
        for t in recent:
            if t.get("has_repetition"):
                report += f"        Problem #{t.get('problem_idx', '?')}: \"{t.get('repetition_sample', '')}\"\n"

    if empty_count > 3:
        report += f"     ⚠️ EMPTY OUTPUT WARNING: {empty_count}/10 outputs were trivially short!\n"

    log.info(report)
    return rep_count, empty_count


# ══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datasets import load_dataset

    log.info("=" * 70)
    log.info("  GRAND COMPARISON v2: %d Strategies × 3 Benchmarks × Multi-Interval", len(STRATEGIES))
    log.info("  repetition_penalty=%.1f | Full trace logging enabled", REPETITION_PENALTY)
    log.info("=" * 70)

    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(SAMPLE_SIZE))
    ds_code = load_dataset("openai/openai_humaneval", split="test").select(range(SAMPLE_SIZE))
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(SAMPLE_SIZE))
    log.info(f"Loaded {len(ds_math)} MATH, {len(ds_code)} HumanEval, {len(ds_arc)} ARC-Challenge.")

    ALL_RESULTS = {}
    total_repetitions_found = 0

    for strat_name, strat_cfg in STRATEGIES.items():
        for interval in strat_cfg["intervals"]:
            # Build unique run name
            if interval is not None:
                run_name = f"{strat_name}_i{interval}"
            else:
                run_name = strat_name

            log.info("\n" + "═" * 70)
            log.info(f"  STRATEGY: {run_name}")
            if interval:
                log.info(f"  Swap interval: every {interval} tokens")
            log.info("═" * 70)

            strat_results = {"math500": {}, "humaneval": {}, "arc_challenge": {}}
            trace_file = TRACES_DIR / f"{run_name}.jsonl"
            all_traces = []

            # ── MATH-500 ──
            log.info(f"  [MATH-500] Starting...")
            corr = tot = total_swaps = 0
            all_ppl = []; all_time = []
            for idx, ex in enumerate(tqdm(ds_math, desc=f"MATH [{run_name}]")):
                factory_kwargs = {"interval": interval} if interval else {}
                router = strat_cfg["factory"](**factory_kwargs)
                gold_raw = ex.get("solution", ex.get("answer"))
                gold = normalize(extract_number(gold_raw))
                question = ex.get("problem", ex.get("question"))
                p = fmt("Solve this math problem step by step. Put your final answer in \\boxed{}.", question)

                resp, metrics = generate_with_router(p, router, max_new=384)
                if hasattr(router, 'cleanup'): router.cleanup()

                pred = normalize(extract_number(resp))
                is_correct = bool(pred and gold and pred == gold)
                if is_correct: corr += 1
                tot += 1
                total_swaps += metrics["swaps"]
                all_ppl.append(metrics["avg_neg_logprob"])
                all_time.append(metrics["elapsed_sec"])

                if metrics["has_repetition"]:
                    total_repetitions_found += 1
                    log.warning(f"  🔴 REPETITION in MATH #{idx}: \"{metrics['repetition_sample']}\"")

                trace = {
                    "strategy": run_name, "benchmark": "math500", "problem_idx": idx,
                    "question": question[:200], "gold_answer": gold,
                    "predicted": pred, "correct": is_correct,
                    "response": resp, "response_len": len(resp),
                    **metrics,
                }
                all_traces.append(trace)
                with open(trace_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")

                # Quality checkpoint every 10
                if (idx + 1) % 10 == 0:
                    quality_checkpoint(run_name, all_traces, idx + 1)

            strat_results["math500"] = {
                "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
                "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
                "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
                "avg_time_sec": round(sum(all_time)/len(all_time), 2),
            }
            log.info(f"  ✅ MATH-500: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps} | "
                     f"PPL: {strat_results['math500']['avg_neg_logprob']:.3f}")

            # ── HumanEval ──
            log.info(f"  [HumanEval] Starting...")
            corr = tot = total_swaps = 0
            all_ppl = []; all_time = []
            for idx, ex in enumerate(tqdm(ds_code, desc=f"CODE [{run_name}]")):
                factory_kwargs = {"interval": interval} if interval else {}
                router = strat_cfg["factory"](**factory_kwargs)
                p = fmt("Complete the Python function. Output ONLY the code.",
                         f"Complete this function:\n```python\n{ex['prompt']}\n```")

                resp, metrics = generate_with_router(p, router, max_new=512)
                if hasattr(router, 'cleanup'): router.cleanup()

                passed = check_humaneval(ex["prompt"], ex["test"], ex["entry_point"], resp)
                if passed: corr += 1
                tot += 1
                total_swaps += metrics["swaps"]
                all_ppl.append(metrics["avg_neg_logprob"])
                all_time.append(metrics["elapsed_sec"])

                if metrics["has_repetition"]:
                    total_repetitions_found += 1
                    log.warning(f"  🔴 REPETITION in CODE #{idx}: \"{metrics['repetition_sample']}\"")

                trace = {
                    "strategy": run_name, "benchmark": "humaneval", "problem_idx": idx,
                    "question": ex["prompt"][:200], "gold_answer": ex["entry_point"],
                    "predicted": "pass" if passed else "fail", "correct": passed,
                    "response": resp, "response_len": len(resp),
                    **metrics,
                }
                all_traces.append(trace)
                with open(trace_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")

                if (idx + 1) % 10 == 0:
                    quality_checkpoint(run_name, all_traces, len(ds_math) + idx + 1)

            strat_results["humaneval"] = {
                "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
                "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
                "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
                "avg_time_sec": round(sum(all_time)/len(all_time), 2),
            }
            log.info(f"  ✅ HumanEval: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps} | "
                     f"PPL: {strat_results['humaneval']['avg_neg_logprob']:.3f}")

            # ── ARC-Challenge ──
            log.info(f"  [ARC-Challenge] Starting...")
            corr = tot = total_swaps = 0
            all_ppl = []; all_time = []
            for idx, ex in enumerate(tqdm(ds_arc, desc=f"ARC [{run_name}]")):
                factory_kwargs = {"interval": interval} if interval else {}
                router = strat_cfg["factory"](**factory_kwargs)
                choices = ex.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                opts = "\n".join([f"{l}: {t}" for l, t in zip(labels, texts)])
                question_text = ex["question"]

                p = fmt("Answer the science question. Reply with ONLY the letter of the correct answer.",
                         f"{question_text}\n\n{opts}")

                resp, metrics = generate_with_router(p, router, max_new=32)
                if hasattr(router, 'cleanup'): router.cleanup()

                # Improved answer extraction for ARC
                pred_raw = resp.strip()
                target = ex["answerKey"].strip().upper()
                # Extract first letter/number that matches an option
                pred = ""
                for char in pred_raw:
                    if char.upper() in [l.upper() for l in labels]:
                        pred = char.upper()
                        break
                if not pred:
                    # fallback
                    pred = pred_raw[:1].upper() if pred_raw else ""

                is_correct = pred == target
                if is_correct: corr += 1
                tot += 1
                total_swaps += metrics["swaps"]
                all_ppl.append(metrics["avg_neg_logprob"])
                all_time.append(metrics["elapsed_sec"])

                if metrics["has_repetition"]:
                    total_repetitions_found += 1
                    log.warning(f"  🔴 REPETITION in ARC #{idx}: \"{metrics['repetition_sample']}\"")

                trace = {
                    "strategy": run_name, "benchmark": "arc_challenge", "problem_idx": idx,
                    "question": question_text[:200], "gold_answer": target,
                    "predicted": pred, "predicted_raw": pred_raw[:100], "correct": is_correct,
                    "response": resp, "response_len": len(resp),
                    **metrics,
                }
                all_traces.append(trace)
                with open(trace_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")

            strat_results["arc_challenge"] = {
                "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
                "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
                "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
                "avg_time_sec": round(sum(all_time)/len(all_time), 2),
            }
            log.info(f"  ✅ ARC-Challenge: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps} | "
                     f"PPL: {strat_results['arc_challenge']['avg_neg_logprob']:.3f}")

            # ── Save strategy results ──
            ALL_RESULTS[run_name] = strat_results
            with open(RESULTS_DIR / "grand_comparison_v2_results.json", "w") as f:
                json.dump(ALL_RESULTS, f, indent=2)
            log.info(f"  💾 Saved. Trace file: {trace_file.name} ({len(all_traces)} entries)")
            log.info(f"  📊 Total repetitions found so far: {total_repetitions_found}")


    # ══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════

    log.info("\n\n" + "═" * 120)
    log.info("  GRAND COMPARISON v2 — FINAL RESULTS")
    log.info("═" * 120)
    log.info(f"{'Strategy':<28} {'MATH%':>6} {'HE%':>6} {'ARC%':>6} {'M-Sw':>5} {'H-Sw':>5} {'A-Sw':>5} {'M-PPL':>6} {'H-PPL':>6} {'A-PPL':>6}")
    log.info("─" * 120)

    for name, res in ALL_RESULTS.items():
        m = res.get("math500", {})
        h = res.get("humaneval", {})
        a = res.get("arc_challenge", {})
        log.info(
            f"{name:<28} "
            f"{m.get('accuracy', 0)*100:>5.1f}% "
            f"{h.get('accuracy', 0)*100:>5.1f}% "
            f"{a.get('accuracy', 0)*100:>5.1f}% "
            f"{m.get('total_swaps', 0):>5d} "
            f"{h.get('total_swaps', 0):>5d} "
            f"{a.get('total_swaps', 0):>5d} "
            f"{m.get('avg_neg_logprob', 0):>6.3f} "
            f"{h.get('avg_neg_logprob', 0):>6.3f} "
            f"{a.get('avg_neg_logprob', 0):>6.3f}"
        )

    log.info("═" * 120)
    log.info(f"  Total repetitions detected across all strategies: {total_repetitions_found}")
    log.info(f"  Generation traces saved to: {TRACES_DIR}/")
    log.info("✅ Grand Comparison v2 Complete.")
