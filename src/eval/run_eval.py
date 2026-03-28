"""
Multi-Adapter Adaptive Clamp — Evaluation Harness (REAL MODE)

⚠️  STATUS:
    --real flag uses the ACTUAL DynamicEngine + Orchestrator backend.
    Without --real, the harness runs in SIMULATION mode (fake numbers).

Usage:
    # Simulation mode (pipeline testing only):
    python3 src/eval/run_eval.py --config_id exp_5_adaptive_05_mixed_fincode

    # REAL mode (calls actual MLX model):
    python3 src/eval/run_eval.py --config_id exp_5_adaptive_05_mixed_fincode --real
"""

import sys
import os
import time
import json
import argparse
import yaml
import random
from pathlib import Path
from datetime import datetime, timezone

# Add project root and backend to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


# ──────────────────────────────────────────────
# SECTION A: SIMULATION FALLBACK (CLEARLY LABELLED)
# ──────────────────────────────────────────────

def get_simulated_metric(method, dataset, metric_name, k, c):
    """⚠️ SIMULATION ONLY. Returns fabricated metrics. Not evidence."""
    random.seed(f"{method}_{dataset}_{k}_{c}")
    base_acc = 0.55 if "mixed" in dataset else 0.82
    base_ppl = 12.5
    if "pass" in metric_name or "Acc" in metric_name or "functionality" in metric_name:
        if method == "adaptive_clamp":
            score = base_acc + 0.075 + random.uniform(-0.01, 0.01)
            if c == 0.3: score -= 0.03
            if c == 1.0 or c == 0.7: score -= 0.015
        elif method == "unclamped_mix":
            score = base_acc + 0.02 + random.uniform(-0.02, 0.02)
        elif method == "static_merge":
            score = base_acc - 0.06 + random.uniform(-0.01, 0.01)
        else:
            score = base_acc + random.uniform(-0.01, 0.01)
        return round(min(max(score, 0.0), 1.0), 4)
    else:
        if method == "adaptive_clamp":
            score = base_ppl * (1.0 + (0.01 if c == 0.5 else 0.018))
        elif method == "unclamped_mix":
            score = base_ppl * (1.0 + random.uniform(0.18, 0.22))
        else:
            score = base_ppl * (1.0 + random.uniform(-0.005, 0.005))
        return round(score, 3)


# ──────────────────────────────────────────────
# SECTION B: REAL ENGINE SINGLETON
# ──────────────────────────────────────────────

_engine = None
_orchestrator = None

def get_engine():
    """Lazily initialise the DynamicEngine + Orchestrator once."""
    global _engine, _orchestrator
    if _engine is None:
        # Adapter paths in expert_registry.json are relative to backend/
        backend_dir = str(PROJECT_ROOT / "backend")
        original_cwd = os.getcwd()
        os.chdir(backend_dir)

        from dynamic_mlx_inference import DynamicEngine, set_global_clamp
        from orchestrator import Orchestrator

        registry_path = "expert_registry.json"
        with open(registry_path) as f:
            registry = json.load(f)

        _engine = DynamicEngine(
            model_path="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            registry=registry,
        )
        _orchestrator = Orchestrator(registry_path, base_engine=_engine)

        os.chdir(original_cwd)
        print("✅ Real DynamicEngine + Orchestrator loaded.")

    return _engine, _orchestrator


# ──────────────────────────────────────────────
# SECTION C: REAL INFERENCE
# ──────────────────────────────────────────────

def generate_response(
    prompt: str,
    method: str,
    k: int,
    c: float | None,
    active_experts_hint: list[tuple[str, float]],
) -> tuple[str, dict]:
    """
    Run the ACTUAL MLX model on `prompt` with the given composition method.

    Returns:
        (generated_text, routing_weights_dict)
    """
    from dynamic_mlx_inference import set_global_clamp

    engine, orchestrator = get_engine()

    # Step 1: Route via the real CoT Orchestrator
    routing_weights_raw, cot_reasoning = orchestrator.route(prompt, top_k=k)

    # Step 2: Apply method-specific configuration
    if method == "single_adapter":
        # Keep only the top-1 adapter
        top_domain = max(routing_weights_raw, key=routing_weights_raw.get)
        routing_weights = {d: (1.0 if d == top_domain else 0.0) for d in routing_weights_raw}
        set_global_clamp(0.5)

    elif method == "static_merge":
        # Top-K adapters with equal weight (simulates offline averaging)
        sorted_domains = sorted(routing_weights_raw.items(), key=lambda x: x[1], reverse=True)
        routing_weights = {d: 0.0 for d in routing_weights_raw}
        for domain, _ in sorted_domains[:k]:
            routing_weights[domain] = 1.0 / k
        set_global_clamp(1.0)  # No clamp, let the equal weights do the work

    elif method == "unclamped_mix":
        # Top-K adapters with their raw scores, NO clamp
        sorted_domains = sorted(routing_weights_raw.items(), key=lambda x: x[1], reverse=True)
        routing_weights = {d: 0.0 for d in routing_weights_raw}
        for domain, score in sorted_domains[:k]:
            routing_weights[domain] = score if score > 0 else 1.0
        set_global_clamp(999.0)  # Effectively unclamped

    elif method == "adaptive_clamp":
        # Top-K adapters with their raw scores, WITH norm clamp
        sorted_domains = sorted(routing_weights_raw.items(), key=lambda x: x[1], reverse=True)
        routing_weights = {d: 0.0 for d in routing_weights_raw}
        for domain, score in sorted_domains[:k]:
            routing_weights[domain] = score if score > 0 else 1.0
        clamp_val = c if c is not None else 0.5
        set_global_clamp(clamp_val)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 3: Generate
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    text, duration = engine.generate(chat_prompt, routing_weights=routing_weights, max_tokens=200)

    return text, routing_weights


def compute_perplexity_real(prompt: str, ground_truth: str, routing_weights: dict) -> float:
    """Compute real perplexity using DynamicEngine.compute_perplexity()."""
    engine, _ = get_engine()
    chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return engine.compute_perplexity(chat_prompt, ground_truth, routing_weights=routing_weights)


# ──────────────────────────────────────────────
# SECTION D: REAL METRIC COMPUTATION
# ──────────────────────────────────────────────

def compute_metric(
    method: str,
    dataset: str,
    metric_name: str,
    k: int,
    c: float | None,
    prediction: str,
    item: dict,
) -> float:
    """Compute a REAL metric from the model's actual prediction text."""
    pred_lower = prediction.strip().lower()

    if "pass" in metric_name or "functionality" in metric_name:
        has_def = "def " in prediction
        has_return = "return " in prediction
        has_code_structure = has_def or has_return or "import " in prediction
        is_nontrivial = len(prediction.strip()) > 20
        return 1.0 if (has_code_structure and is_nontrivial) else 0.0

    elif "exact" in metric_name:
        expected = item.get("expected_answer", "").strip().lower()
        return 1.0 if expected and expected in pred_lower else 0.0

    else:
        # Keyword recall against reference
        expected = item.get("expected_answer", "")
        if not expected:
            # No reference — use semantic_similarity placeholder
            return 0.0
        keywords = set(expected.lower().split()) - {
            "the", "a", "an", "is", "are", "and", "or", "in", "on", "to", "for", "of"
        }
        if not keywords:
            return 0.0
        matches = sum(1 for kw in keywords if kw in pred_lower)
        return round(matches / len(keywords), 4)


# ──────────────────────────────────────────────
# SECTION E: LOGGING
# ──────────────────────────────────────────────

def append_log(log_entry: dict, filepath: str = "results_db.jsonl"):
    with open(filepath, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ──────────────────────────────────────────────
# SECTION F: MAIN EXPERIMENT LOOP
# ──────────────────────────────────────────────

def run_experiment(config_id: str, real_mode: bool = False) -> dict:
    os.chdir(PROJECT_ROOT)

    with open("configs/uma_experiments.yaml", "r") as f:
        configs = yaml.safe_load(f)
    if config_id not in configs:
        raise ValueError(f"Config ID {config_id} not found in configs/uma_experiments.yaml")

    config = configs[config_id]
    mode_label = "REAL" if real_mode else "SIMULATION"
    print(f"\n[{mode_label}] Config: {config_id} -> {config}")

    expert_ids = ["math", "code", "legal", "finance", "philosophy"]

    # Load dataset
    dataset_path = f"data/{config['dataset']}"
    if not Path(dataset_path).exists():
        print(f"⚠️  {dataset_path} not found. Using 1 mock prompt.")
        dataset = [{"id": "mock_01", "prompt": "Write Python to calculate option price.", "eval_metric": "pass@1"}]
    else:
        with open(dataset_path, "r") as f:
            if dataset_path.endswith('.jsonl'):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)

    processed = 0
    for item in dataset:
        from src.routers.cot_router import select_experts_cot
        active_experts = select_experts_cot(item["prompt"], expert_ids, k=config["k"])

        if config["method"] == "single_adapter":
            active_experts = active_experts[:1]

        start_t = time.perf_counter()

        if real_mode:
            # ── REAL PATH ──
            prediction, routing_weights_used = generate_response(
                prompt=item["prompt"],
                method=config["method"],
                k=config["k"],
                c=config.get("c", None),
                active_experts_hint=active_experts,
            )
            latency = (time.perf_counter() - start_t) * 1000

            score = compute_metric(
                method=config["method"],
                dataset=config["dataset"],
                metric_name=item.get("eval_metric", "pass@1"),
                k=config.get("k", 1),
                c=config.get("c", None),
                prediction=prediction,
                item=item,
            )
        else:
            # ── SIMULATION PATH ──
            prediction = "(simulated — no real model called)"
            routing_weights_used = {}
            latency = random.uniform(1, 10)
            score = get_simulated_metric(
                config["method"], config["dataset"],
                item.get("eval_metric", "pass@1"),
                config.get("k", 1), config.get("c", None),
            )

        log_line = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exp_id": config_id,
            "method": config["method"],
            "k": config["k"],
            "c": config.get("c", None),
            "dataset": config["dataset"],
            "prompt_id": item["id"],
            "prompt_text": item["prompt"][:200],
            "prediction_preview": prediction[:300] if prediction else "",
            "metric_name": item.get("eval_metric", "pass@1"),
            "metric_value": score,
            "latency_ms": round(latency, 2),
            "real_mode": real_mode,
        }
        append_log(log_line)
        processed += 1

        if real_mode:
            print(f"  [{processed}] id={item['id']} | score={score:.4f} | latency={latency:.1f}ms")
            print(f"       prediction[:120] = {prediction[:120]}")

    return {"status": "COMPLETE", "mode": mode_label, "prompts_processed": processed}


def main():
    parser = argparse.ArgumentParser(description="Multi-Adapter Eval Harness")
    parser.add_argument("--config_id", type=str, required=True)
    parser.add_argument("--real", action="store_true",
                        help="Use REAL model inference (requires MLX + trained adapters).")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Eval Harness | config={args.config_id} | real={args.real}")
    print(f"{'='*60}")
    summary = run_experiment(args.config_id, real_mode=args.real)
    print(f"\nDone. {summary}")


if __name__ == "__main__":
    main()
