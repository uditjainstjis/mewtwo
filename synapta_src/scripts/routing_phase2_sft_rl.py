#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  PHASE 2: SFT + RL Routing Strategies
  Runs AFTER the grand comparison pipeline completes.
  
  Uses oracle routing traces to train a proper SFT router,
  and implements REINFORCE-based policy gradient routing.
═══════════════════════════════════════════════════════════════════════════

Strategies added:
  10. SFT Router (trained on oracle adapter decisions from Phase 1)
  11. REINFORCE Router (policy-gradient with per-token PPL reward)
  12. UCB Bandit Router (online, zero-shot contextual bandit)
"""
import sys, gc, json, logging, re, time, tempfile, subprocess, math as pymath
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import numpy as np

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
RESULTS_DIR = PROJECT / "results" / "nemotron"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT / "logs" / "phase2_sft_rl.log", mode="w"),
    ],
)
log = logging.getLogger("phase2")

SAMPLE_SIZE = 25
ADAPTERS = ["math", "code", "science"]
DOMAIN_MAP = {0: "math", 1: "code", 2: "science"}
REVERSE_MAP = {"math": 0, "code": 1, "science": 2}

# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

log.info("=" * 70)
log.info("  PHASE 2: SFT + RL Routing — Loading Nemotron-30B")
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
log.info("✅ Model loaded.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: COLLECT ORACLE ROUTING TRACES
# For each problem, generate with each adapter separately and record
# per-token log-probs. The oracle decision at each position is the
# adapter that assigned highest probability to the actual next token.
# ══════════════════════════════════════════════════════════════════════════

def collect_oracle_traces(problems, sys_prompt, max_new=384):
    """
    For each problem:
      1. Generate a reference answer using the math adapter.
      2. Score every token in that answer under all 3 adapters.
      3. Record which adapter was "best" at each token position.
    Returns list of traces, each containing:
      - input_ids, token positions, per-position oracle decisions,
        per-position embeddings (for SFT training).
    """
    traces = []

    for i, prob_text in enumerate(tqdm(problems, desc="Oracle Traces")):
        prompt = fmt(sys_prompt, prob_text)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Generate reference with math adapter
        model.set_adapter("math")
        pv = HybridCache(base_model.config, batch_size=1, dtype=torch.bfloat16, device=model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id, use_cache=True, past_key_values=pv,
            )
        del pv
        gc.collect(); torch.cuda.empty_cache()

        full_ids = out[0]  # (total_len,)
        gen_ids = full_ids[input_len:]  # generated part only
        if len(gen_ids) < 5:
            continue  # skip trivially short generations

        # Now score the full sequence under each adapter
        # We need log_prob(token_t | tokens_<t) for each adapter
        adapter_logprobs = {}

        for ad in ADAPTERS:
            model.set_adapter(ad)
            pv = HybridCache(base_model.config, batch_size=1, dtype=torch.bfloat16, device=model.device)
            with torch.no_grad():
                outputs = model(full_ids.unsqueeze(0), past_key_values=pv)
                logits = outputs.logits[0]  # (seq_len, vocab)
                log_probs = F.log_softmax(logits, dim=-1)

                # For each generated token position t, get log_prob of the actual token
                # logits at position (input_len + t - 1) predicts token at position (input_len + t)
                per_token_lp = []
                for t in range(len(gen_ids)):
                    pos = input_len + t - 1  # position that predicts gen_ids[t]
                    if pos >= 0 and pos < log_probs.shape[0]:
                        lp = log_probs[pos, gen_ids[t]].item()
                        per_token_lp.append(lp)

                adapter_logprobs[ad] = per_token_lp
            del pv
            gc.collect(); torch.cuda.empty_cache()

        # Determine oracle decision at each position
        min_len = min(len(adapter_logprobs[ad]) for ad in ADAPTERS)
        oracle_decisions = []
        for t in range(min_len):
            best_ad = max(ADAPTERS, key=lambda ad: adapter_logprobs[ad][t])
            oracle_decisions.append(best_ad)

        # Extract embeddings at each decision point for SFT training
        # Use the token embeddings (fast, no full forward pass needed)
        with torch.no_grad():
            embeds = base_model.backbone.embeddings(gen_ids[:min_len].unsqueeze(0)).squeeze(0).cpu().float()

        traces.append({
            "oracle_decisions": oracle_decisions,
            "embeddings": embeds,  # (min_len, 2688)
            "adapter_logprobs": {ad: adapter_logprobs[ad][:min_len] for ad in ADAPTERS},
        })

        if (i + 1) % 5 == 0:
            log.info(f"  Trace {i+1}: {min_len} tokens, "
                     f"oracle picks: math={oracle_decisions.count('math')}, "
                     f"code={oracle_decisions.count('code')}, "
                     f"science={oracle_decisions.count('science')}")

    return traces


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN SFT ROUTER ON ORACLE LABELS
# ══════════════════════════════════════════════════════════════════════════

class OracleSFTRouter(nn.Module):
    """MLP trained on (embedding → oracle_adapter_label) pairs."""
    def __init__(self, hidden_dim=2688, bottleneck=256, num_classes=3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(bottleneck, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(self.norm(x))


def train_sft_router(traces):
    """Train a router MLP on oracle-labeled embeddings."""
    # Flatten all traces into a dataset
    all_embeds = []
    all_labels = []
    for tr in traces:
        all_embeds.append(tr["embeddings"])
        all_labels.extend([REVERSE_MAP[d] for d in tr["oracle_decisions"]])

    X = torch.cat(all_embeds, dim=0)  # (N, 2688)
    Y = torch.tensor(all_labels, dtype=torch.long)  # (N,)

    log.info(f"  SFT Training Data: {X.shape[0]} tokens")
    log.info(f"  Label distribution: math={all_labels.count(0)}, code={all_labels.count(1)}, science={all_labels.count(2)}")

    # Train/val split
    n = X.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.8)
    X_train, Y_train = X[perm[:split]].cuda(), Y[perm[:split]].cuda()
    X_val, Y_val = X[perm[split:]].cuda(), Y[perm[split:]].cuda()

    router = OracleSFTRouter().cuda()
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for ep in range(200):
        router.train()
        idx = torch.randperm(X_train.size(0))
        total_loss = 0
        for i in range(0, X_train.size(0), 128):
            batch_x = X_train[idx[i:i+128]]
            batch_y = Y_train[idx[i:i+128]]
            optimizer.zero_grad()
            loss = criterion(router(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        router.eval()
        with torch.no_grad():
            val_logits = router(X_val)
            val_acc = (val_logits.argmax(dim=-1) == Y_val).float().mean().item()
            train_logits = router(X_train)
            train_acc = (train_logits.argmax(dim=-1) == Y_train).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(router.state_dict(), PROJECT / "adapters" / "routers" / "sft_oracle_router.pt")

        if ep % 50 == 0 or ep == 199:
            log.info(f"  Ep {ep:3d} | Loss: {total_loss:.4f} | TrainAcc: {train_acc:.1%} | ValAcc: {val_acc:.1%}")

    log.info(f"  ✅ SFT Router trained. Best ValAcc: {best_val_acc:.1%}")

    # Reload best
    router.load_state_dict(torch.load(PROJECT / "adapters" / "routers" / "sft_oracle_router.pt"))
    router.eval()
    return router


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: REINFORCE ROUTER (POLICY GRADIENT)
# ══════════════════════════════════════════════════════════════════════════

class ReinforcePolicy(nn.Module):
    """Small policy network for REINFORCE routing."""
    def __init__(self, hidden_dim=2688, num_actions=3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return F.softmax(self.net(self.norm(x)), dim=-1)


def train_reinforce_router(traces):
    """
    Train a routing policy using REINFORCE.
    Reward = negative perplexity under the chosen adapter.
    This is per-token, so we get dense reward signal.
    """
    policy = ReinforcePolicy().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    log.info("  Training REINFORCE policy on oracle trace data...")

    for epoch in range(100):
        total_reward = 0
        total_loss = 0

        for tr in traces:
            embeds = tr["embeddings"].cuda()  # (T, 2688)
            adapter_lps = tr["adapter_logprobs"]  # {adapter: [lp_per_token]}

            # For each token, sample an action from the policy
            probs = policy(embeds)  # (T, 3)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()  # (T,)
            log_probs = dist.log_prob(actions)  # (T,)

            # Compute per-token reward: log_prob of actual token under chosen adapter
            rewards = []
            for t in range(len(actions)):
                chosen_ad = DOMAIN_MAP[actions[t].item()]
                # Reward = how well the chosen adapter predicted this token
                r = adapter_lps[chosen_ad][t]
                rewards.append(r)

            rewards = torch.tensor(rewards, device="cuda", dtype=torch.float32)

            # Normalize rewards (variance reduction)
            if rewards.std() > 0:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # REINFORCE loss: -log_prob(action) * reward
            loss = -(log_probs * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_reward += rewards.sum().item()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 99:
            log.info(f"  REINFORCE Ep {epoch:3d} | Loss: {total_loss:.4f} | TotalReward: {total_reward:.1f}")

    torch.save(policy.state_dict(), PROJECT / "adapters" / "routers" / "reinforce_router.pt")
    log.info("  ✅ REINFORCE policy trained and saved.")

    policy.eval()
    return policy


# ══════════════════════════════════════════════════════════════════════════
# ROUTER PROCESSORS (for evaluation)
# ══════════════════════════════════════════════════════════════════════════

class SFTRouterProcessor(LogitsProcessor):
    """Uses the oracle-SFT-trained MLP for routing decisions."""
    def __init__(self, sft_model):
        self.sft = sft_model
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        model.set_adapter("math")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 10 == 0:
            with torch.no_grad():
                embeds = base_model.backbone.embeddings(input_ids[:, -1:]).squeeze(1).float()
                logits = self.sft(embeds)
                new = DOMAIN_MAP[logits.argmax(dim=-1).item()]
            if new != self.current_adapter:
                model.set_adapter(new)
                self.current_adapter = new
                self.swaps += 1
        return scores

    def cleanup(self):
        pass


class ReinforceRouterProcessor(LogitsProcessor):
    """Uses the REINFORCE-trained policy for routing decisions."""
    def __init__(self, policy):
        self.policy = policy
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        model.set_adapter("math")

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            self.ppl_log.append(-top_lp)

        if input_ids.shape[1] % 10 == 0:
            with torch.no_grad():
                embeds = base_model.backbone.embeddings(input_ids[:, -1:]).squeeze(1).float()
                probs = self.policy(embeds)
                # At eval time, take argmax (greedy) instead of sampling
                new = DOMAIN_MAP[probs.argmax(dim=-1).item()]
            if new != self.current_adapter:
                model.set_adapter(new)
                self.current_adapter = new
                self.swaps += 1
        return scores

    def cleanup(self):
        pass


class UCBBanditRouter(LogitsProcessor):
    """
    Upper Confidence Bound bandit — fully online, zero-shot.
    Treats each adapter as an arm. Balances exploration/exploitation.
    """
    def __init__(self):
        self.swaps = 0
        self.current_adapter = "math"
        self.ppl_log = []
        model.set_adapter("math")

        # Bandit state
        self.counts = {ad: 1 for ad in ADAPTERS}  # times each arm pulled
        self.rewards = {ad: 0.0 for ad in ADAPTERS}  # cumulative reward
        self.total_pulls = 3  # start after 1 pull each

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            log_probs = F.log_softmax(scores, dim=-1)
            top_lp = log_probs[0, scores[0].argmax()].item()
            neg_lp = -top_lp
            self.ppl_log.append(neg_lp)

            # Update reward for current adapter (lower ppl = higher reward)
            self.rewards[self.current_adapter] += (-neg_lp)  # positive reward for low ppl

        if input_ids.shape[1] % 10 == 0:
            # UCB selection
            best_ad = self.current_adapter
            best_ucb = -float("inf")
            for ad in ADAPTERS:
                mean_reward = self.rewards[ad] / self.counts[ad]
                exploration = pymath.sqrt(2 * pymath.log(self.total_pulls + 1) / self.counts[ad])
                ucb = mean_reward + exploration
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_ad = ad

            self.counts[best_ad] += 1
            self.total_pulls += 1

            if best_ad != self.current_adapter:
                model.set_adapter(best_ad)
                self.current_adapter = best_ad
                self.swaps += 1

        return scores

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
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


def generate_with_router(prompt, router_instance, max_new=384):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    pv = make_cache()

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id, use_cache=True,
            past_key_values=pv,
            logits_processor=LogitsProcessorList([router_instance]),
        )
    elapsed = time.time() - t0
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    avg_ppl = sum(router_instance.ppl_log) / max(len(router_instance.ppl_log), 1)

    del pv; gc.collect(); torch.cuda.empty_cache()

    return resp, {
        "swaps": router_instance.swaps,
        "avg_neg_logprob": round(avg_ppl, 4),
        "tokens_generated": len(router_instance.ppl_log),
        "elapsed_sec": round(elapsed, 2),
    }


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
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datasets import load_dataset

    log.info("=" * 70)
    log.info("  PHASE 2: Oracle Traces → SFT + REINFORCE + UCB Bandit")
    log.info("=" * 70)

    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(SAMPLE_SIZE))
    ds_code = load_dataset("openai/openai_humaneval", split="test").select(range(SAMPLE_SIZE))
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(SAMPLE_SIZE))

    # ── STEP 1: Collect Oracle Traces ──
    log.info("─" * 70)
    log.info("  COLLECTING ORACLE ROUTING TRACES (this takes ~20 min)...")
    log.info("─" * 70)

    math_problems = [ex.get("problem", ex.get("question")) for ex in ds_math]
    traces = collect_oracle_traces(
        math_problems,
        "Solve this math problem step by step. Put your final answer in \\boxed{}.",
        max_new=256,
    )
    log.info(f"  Collected {len(traces)} traces with "
             f"{sum(len(t['oracle_decisions']) for t in traces)} total token decisions.")

    # Log oracle stats
    all_decisions = []
    for tr in traces:
        all_decisions.extend(tr["oracle_decisions"])
    log.info(f"  Oracle adapter usage: math={all_decisions.count('math')}, "
             f"code={all_decisions.count('code')}, science={all_decisions.count('science')}")

    # ── STEP 2: Train SFT Router ──
    log.info("─" * 70)
    log.info("  TRAINING SFT ROUTER ON ORACLE LABELS...")
    log.info("─" * 70)
    sft_router = train_sft_router(traces)

    # ── STEP 3: Train REINFORCE Router ──
    log.info("─" * 70)
    log.info("  TRAINING REINFORCE ROUTER...")
    log.info("─" * 70)
    rl_policy = train_reinforce_router(traces)

    # ── STEP 4: Evaluate all 3 new strategies ──
    log.info("─" * 70)
    log.info("  EVALUATING: SFT / REINFORCE / UCB Bandit")
    log.info("─" * 70)

    PHASE2_STRATEGIES = {
        "10_sft_oracle":     lambda: SFTRouterProcessor(sft_router),
        "11_reinforce_rl":   lambda: ReinforceRouterProcessor(rl_policy),
        "12_ucb_bandit":     lambda: UCBBanditRouter(),
    }

    # Load Phase 1 results if available
    p1_path = RESULTS_DIR / "grand_comparison_results.json"
    if p1_path.exists():
        with open(p1_path) as f:
            ALL_RESULTS = json.load(f)
        log.info(f"  Loaded {len(ALL_RESULTS)} Phase 1 results.")
    else:
        ALL_RESULTS = {}

    for strat_name, strat_factory in PHASE2_STRATEGIES.items():
        log.info(f"\n  === {strat_name} ===")
        strat_results = {"math500": {}, "humaneval": {}, "arc_challenge": {}}

        # ── MATH-500 ──
        corr = tot = total_swaps = 0
        all_ppl = []; all_time = []
        for ex in tqdm(ds_math, desc=f"MATH [{strat_name}]"):
            router = strat_factory()
            gold = normalize(extract_number(ex.get("solution", ex.get("answer"))))
            p = fmt("Solve this math problem step by step. Put your final answer in \\boxed{}.",
                     ex.get("problem", ex.get("question")))
            resp, metrics = generate_with_router(p, router, max_new=384)
            if hasattr(router, 'cleanup'): router.cleanup()
            pred = normalize(extract_number(resp))
            if pred and gold and pred == gold: corr += 1
            tot += 1
            total_swaps += metrics["swaps"]
            all_ppl.append(metrics["avg_neg_logprob"])
            all_time.append(metrics["elapsed_sec"])

        strat_results["math500"] = {
            "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
            "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
            "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time)/len(all_time), 2),
        }
        log.info(f"  MATH-500: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps}")

        # ── HumanEval ──
        corr = tot = total_swaps = 0
        all_ppl = []; all_time = []
        for ex in tqdm(ds_code, desc=f"CODE [{strat_name}]"):
            router = strat_factory()
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

        strat_results["humaneval"] = {
            "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
            "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
            "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time)/len(all_time), 2),
        }
        log.info(f"  HumanEval: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps}")

        # ── ARC-Challenge ──
        corr = tot = total_swaps = 0
        all_ppl = []; all_time = []
        for ex in tqdm(ds_arc, desc=f"ARC [{strat_name}]"):
            router = strat_factory()
            choices = ex.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            opts = "\n".join([f"{l}: {t}" for l, t in zip(labels, texts)])
            
            p = fmt("Answer the multiple-choice science question. Output ONLY the correct option label (e.g., A, B, C, D, 1, 2, 3, 4).",
                     f"{ex['question']}\nOptions:\n{opts}")
            resp, metrics = generate_with_router(p, router, max_new=16)
            if hasattr(router, 'cleanup'): router.cleanup()
            
            pred = resp.strip().upper()
            target = ex["answerKey"].strip().upper()
            if target == pred or pred.startswith(target) or f" {target} " in f" {pred} ":
                corr += 1
            tot += 1
            total_swaps += metrics["swaps"]
            all_ppl.append(metrics["avg_neg_logprob"])
            all_time.append(metrics["elapsed_sec"])

        strat_results["arc_challenge"] = {
            "accuracy": round(corr/tot, 4), "correct": corr, "total": tot,
            "total_swaps": total_swaps, "avg_swaps": round(total_swaps/tot, 1),
            "avg_neg_logprob": round(sum(all_ppl)/len(all_ppl), 4),
            "avg_time_sec": round(sum(all_time)/len(all_time), 2),
        }
        log.info(f"  ARC-Challenge: {corr/tot:.1%} ({corr}/{tot}) | Swaps: {total_swaps}")

        ALL_RESULTS[strat_name] = strat_results

        with open(RESULTS_DIR / "grand_comparison_results.json", "w") as f:
            json.dump(ALL_RESULTS, f, indent=2)

    # ── Final Table ──
    log.info("\n" + "=" * 110)
    log.info("  COMPLETE COMPARISON TABLE (Phase 1 + Phase 2)")
    log.info("=" * 110)
    log.info(f"{'Strategy':<20} {'MATH%':>6} {'HE%':>6} {'ARC%':>6} {'M-Swap':>6} {'H-Swap':>6} {'A-Swap':>6} {'M-PPL':>6} {'H-PPL':>6} {'A-PPL':>6}")
    log.info("-" * 110)
    for name, res in sorted(ALL_RESULTS.items()):
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
    log.info("✅ Phase 2 Complete. All 12 strategies evaluated.")
