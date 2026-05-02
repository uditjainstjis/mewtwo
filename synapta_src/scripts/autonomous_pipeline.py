#!/usr/bin/env python3
"""
LoRI-MoE Autonomous Multi-Model Training Pipeline + Watchdog

This script:
1. Trains all 5 domain adapters for EACH base model in the queue
2. Auto-retries on OOM by halving batch size
3. Auto-continues to next model when current one finishes
4. Logs everything to pipeline.log
5. Survives via nohup — GPU never sits idle

Usage:
    nohup python scripts/autonomous_pipeline.py >> logs/lori_moe/pipeline.log 2>&1 &

Monitor:
    tail -f logs/lori_moe/pipeline.log
"""

import os
import sys
import json
import time
import subprocess
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta

# ─── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = "/home/learner/Desktop/mewtwo"
VENV_PYTHON = f"{PROJECT_ROOT}/.venv/bin/python"
CHECKPOINT_BASE = f"{PROJECT_ROOT}/adapters/lori_moe"
LOG_DIR = f"{PROJECT_ROOT}/logs/lori_moe"
DOMAINS = ["math", "code", "science", "legal", "medical"]

# Pipeline processes these sequentially. Add new models and they'll be picked up.
DEFAULT_MODEL_QUEUE = [
    {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "short_name": "qwen2.5_14b_instruct",
        "initial_batch_size": 2,
        "min_batch_size": 1,
        "grad_accum": 16,
        "gradient_checkpointing": True,
        "use_4bit": True,
        "is_gdn": False,
        "enabled_domains": ["math", "code", "science", "legal", "medical"]
    },
    {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "short_name": "qwen3.5_0.8b",
        "initial_batch_size": 32,
        "min_batch_size": 4,
        "grad_accum": 2,
        "gradient_checkpointing": False,
        "use_4bit": False,
        "is_gdn": False,
        "enabled_domains": ["math", "code", "science", "legal", "medical"]
    },
    {
        "model_id": "Qwen/Qwen3.5-9B",
        "short_name": "qwen3.5_9b",
        "initial_batch_size": 4,
        "min_batch_size": 1,
        "grad_accum": 16,
        "gradient_checkpointing": True,
        "use_4bit": True,
        "is_gdn": False,
        "enabled_domains": ["math", "code", "science", "legal", "medical"]
    },
    {
        "model_id": "Qwen/Qwen3.5-27B",
        "short_name": "qwen3.5_27b",
        "initial_batch_size": 2,
        "min_batch_size": 1,
        "grad_accum": 32,
        "gradient_checkpointing": True,
        "use_4bit": True,
        "is_gdn": False,
        "enabled_domains": ["math", "code", "science", "legal", "medical"]
    },
]

def load_queue_config():
    config_file = f"{CHECKPOINT_BASE}/queue_config.json"
    active_config = DEFAULT_MODEL_QUEUE
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                saved = json.load(f)
                # Merge saved selections with defaults
                merged = []
                for default in DEFAULT_MODEL_QUEUE:
                    match = next((s for s in saved if s.get("short_name") == default["short_name"] or s.get("short") == default["short_name"]), None)
                    if match:
                        new_item = default.copy()
                        if "enabled_domains" in match:
                            new_item["enabled_domains"] = match["enabled_domains"]
                        merged.append(new_item)
                    else:
                        merged.append(default)
                active_config = merged
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            
    return active_config

TRAINING_CONFIG = {
    "rank": 32,
    "sparsity": 0.8,
    "epochs": 3,
    "max_train_samples": 10000,
    "lr": 5e-5,
    "max_seq_length": 512,
    "log_every": 10,
    "save_every": 500,
    "optimizer_backend": "auto",
    "compile_mode": None,  # DISABLED: Triton autotuning eats ~10GB VRAM and causes OOM
}

SPEED_MODE = os.environ.get("LORI_SPEED_MODE", "quality").strip().lower()
if SPEED_MODE == "turbo":
    TRAINING_CONFIG.update(
        {
            "epochs": 1,
            "max_train_samples": 2048,
            "max_seq_length": 512,
            "log_every": 20,
            "save_every": 1000,
        }
    )

# ─── Logging Setup ──────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

# Use a dedicated logger to avoid duplicate lines from basicConfig
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    _fh = logging.FileHandler(f"{LOG_DIR}/autonomous_pipeline.log")
    _fh.setFormatter(_fmt)
    logger.addHandler(_sh)
    logger.addHandler(_fh)


def banner(msg, char="═"):
    line = char * 55
    logger.info(f"\n{line}\n  {msg}\n{line}")


# ─── State Tracking ────────────────────────────────────────────────────────────

STATE_FILE = f"{CHECKPOINT_BASE}/pipeline_state.json"


def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "started_at": datetime.now().isoformat()}


def save_state(state):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Training Function ─────────────────────────────────────────────────────────


def train_single_adapter(model_config, domain, batch_size, grad_accum):
    """Train a single domain adapter with real-time logging. Returns (success, error_type)."""
    output_dir = f"{CHECKPOINT_BASE}/{model_config['short_name']}"
    log_file_path = f"{LOG_DIR}/train_{model_config['short_name']}_{domain}.log"
    
    cmd = [
        VENV_PYTHON, "-u", "-m", "src.lori_moe.training.train_lori_adapter",
        "--domain", domain,
        "--base_model", model_config["model_id"],
        "--output_dir", output_dir,
        "--rank", str(TRAINING_CONFIG["rank"]),
        "--sparsity", str(TRAINING_CONFIG["sparsity"]),
        "--epochs", str(TRAINING_CONFIG["epochs"]),
        "--max_train_samples", str(TRAINING_CONFIG["max_train_samples"]),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--lr", str(TRAINING_CONFIG["lr"]),
        "--max_seq_length", str(TRAINING_CONFIG["max_seq_length"]),
        "--log_every", str(TRAINING_CONFIG["log_every"]),
        "--save_every", str(TRAINING_CONFIG["save_every"]),
        "--optimizer_backend", str(TRAINING_CONFIG["optimizer_backend"]),
    ]

    if TRAINING_CONFIG["compile_mode"]:
        cmd.extend(["--compile_mode", str(TRAINING_CONFIG["compile_mode"])])
    
    if model_config.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")
        
    if model_config.get("use_4bit", False):
        cmd.append("--use_4bit")
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info(f"  Command: {' '.join(cmd[-8:])}")  # Log last 8 args
    
    logger.info(f"  Streaming logs to: {os.path.basename(log_file_path)}")
    
    try:
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            
            # Wait for completion while monitoring for stop signal
            while process.poll() is None:
                if os.path.exists(f"{CHECKPOINT_BASE}/STOP_PIPELINE"):
                    process.terminate()
                    logger.info("🛑 Subprocess terminated via stop signal.")
                    return False, "TERMINATED"
                time.sleep(1)
            
            if process.returncode == 0:
                return True, None
            
            # Read tail of log file to check for OOM
            with open(log_file_path, "r") as f:
                stderr_tail = f.read()[-1000:]
                
            if "OutOfMemoryError" in stderr_tail or "CUDA out of memory" in stderr_tail:
                return False, "OOM"
            else:
                return False, f"FAILED_EXIT_{process.returncode}"
            
    except Exception as e:
        logger.error(f"  Process Execution Exception: {e}")
        return False, "EXCEPTION"


def train_domain_with_retry(model_config, domain, max_retries=5):
    """Train a domain adapter with automatic OOM retry (halve batch size)."""
    batch_size = model_config["initial_batch_size"]
    grad_accum = model_config["grad_accum"]
    
    for attempt in range(max_retries):
        logger.info(
            f"  [{domain}] Attempt {attempt+1}/{max_retries} "
            f"(bs={batch_size}, accum={grad_accum}, effective={batch_size*grad_accum})"
        )
        
        success, error_type = train_single_adapter(
            model_config, domain, batch_size, grad_accum
        )
        
        if success:
            logger.info(f"  [{domain}] ✅ SUCCESS")
            return True
        
        if error_type == "OOM" and batch_size > model_config["min_batch_size"]:
            old_bs = batch_size
            batch_size = max(batch_size // 2, model_config["min_batch_size"])
            grad_accum = grad_accum * 2  # Keep effective BS same
            logger.warning(
                f"  [{domain}] OOM! Reducing batch: {old_bs}→{batch_size}, "
                f"accum→{grad_accum}"
            )
            time.sleep(5)  # Let GPU memory settle
        elif error_type == "OOM":
            logger.error(f"  [{domain}] ❌ OOM at minimum batch size {batch_size}. Skipping.")
            return False
        else:
            logger.error(f"  [{domain}] ❌ Non-OOM error: {error_type}. Skipping.")
            return False
    
    logger.error(f"  [{domain}] ❌ Failed after {max_retries} attempts")
    return False


# ─── Main Pipeline ──────────────────────────────────────────────────────────────


def run_pipeline():
    state = load_state()
    model_queue = load_queue_config()
    total_models = len(model_queue)
    
    banner("LoRI-MoE AUTONOMOUS MULTI-MODEL PIPELINE")
    logger.info(f"  Models queued: {total_models}")
    logger.info(f"  Domains per model: {len(DOMAINS)}")
    logger.info(f"  Total adapter jobs: {total_models * len(DOMAINS)}")
    logger.info(f"  Speed mode:   {SPEED_MODE}")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Write PID for dashboard control
    with open("/tmp/lori_pipeline.pid", "w") as f:
        f.write(str(os.getpid()))
    
    for model_idx, model_config in enumerate(model_queue):
        # Refresh config every model loop to see dashboard adjustments
        model_queue = load_queue_config()
        model_config = model_queue[model_idx]
        
        # Check for stop signal
        if os.path.exists(f"{CHECKPOINT_BASE}/STOP_PIPELINE"):
            logger.info("🛑 Stop signal detected. Exiting pipeline.")
            break
            
        model_name = model_config["short_name"]
        model_id = model_config["model_id"]
        
        # Check if model is disabled in config
        enabled_domains = model_config.get("enabled_domains", DOMAINS)
        
        # Skip if all domains already completed for this model
        completed_domains = state["completed"].get(model_name, [])
        remaining = [d for d in DOMAINS if d in enabled_domains and d not in completed_domains]
        
        if not remaining:
            logger.info(f"\n[{model_idx+1}/{total_models}] {model_name}: No active domains, skipping.")
            continue
        
        banner(f"MODEL {model_idx+1}/{total_models}: {model_id}")
        logger.info(f"  Short name:  {model_name}")
        logger.info(f"  Batch size:  {model_config['initial_batch_size']}")
        logger.info(f"  GDN arch:    {model_config['is_gdn']}")
        logger.info(f"  Remaining:   {remaining}")
        logger.info(f"  Completed:   {completed_domains}")
        
        model_start = time.time()
        
        for domain_idx, domain in enumerate(remaining):
            domain_start = time.time()
            logger.info(
                f"\n  ┌─ [{model_name}] Domain {domain_idx+1}/{len(remaining)}: {domain}"
            )
            
            success = train_domain_with_retry(model_config, domain)
            
            elapsed = (time.time() - domain_start) / 60
            
            if success:
                if model_name not in state["completed"]:
                    state["completed"][model_name] = []
                state["completed"][model_name].append(domain)
                save_state(state)
                logger.info(f"  └─ [{domain}] Done in {elapsed:.1f} min ✅")
            else:
                if model_name not in state["failed"]:
                    state["failed"][model_name] = []
                state["failed"][model_name].append(domain)
                save_state(state)
                logger.info(f"  └─ [{domain}] FAILED after {elapsed:.1f} min ❌")
        
        model_elapsed = (time.time() - model_start) / 60
        banner(f"{model_name} ADAPTERS COMPLETE — {model_elapsed:.1f} min total")

        # ─── Automatic Router Training ───
        all_completed = state["completed"].get(model_name, [])
        if set(enabled_domains).issubset(set(all_completed)):
            logger.info(f"\n[{model_name}] All adapters complete! Starting Router Training...")
            
            router_cmd = [
                VENV_PYTHON, "-m", "src.lori_moe.training.train_router",
                "--base_model", model_id,
                "--adapter_dir", f"{CHECKPOINT_BASE}/{model_name}",
                "--output_dir", f"{CHECKPOINT_BASE}/{model_name}/router",
                "--epochs", "2",
                "--batch_size", "16",
                "--max_samples_per_domain", "2000"
            ]
            
            router_log_file = f"{LOG_DIR}/train_{model_name}_router.log"
            logger.info(f"  Streaming router logs to: {Path(router_log_file).name}")
            
            try:
                with open(router_log_file, "a") as f:
                    subprocess.run(router_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                logger.info(f"  └─ [Router] Done ✅")
                
                # Mark router as complete to avoid retrying
                if f"{model_name}_router" not in state["completed"]:
                    state["completed"].setdefault(f"{model_name}_router", []).append("router")
                    save_state(state)
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"  └─ [Router] FAILED ❌ Check {Path(router_log_file).name}")
    # ─── Summary ──────────────────────────────────────────────────────────────
    banner("PIPELINE PASS COMPLETE")
    logger.info(f"  Completed: {json.dumps(state['completed'], indent=4)}")
    if state["failed"]:
        logger.info(f"  Failed:    {json.dumps(state['failed'], indent=4)}")
    logger.info(f"  Finished:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return state


def perpetual_loop():
    """Run pipeline forever. After all models done, sleep and re-check for new work."""
    cycle = 0
    while True:
        cycle += 1
        banner(f"PERPETUAL CYCLE #{cycle}")

        # Check stop signal
        if os.path.exists(f"{CHECKPOINT_BASE}/STOP_PIPELINE"):
            logger.info("🛑 STOP_PIPELINE file detected. Exiting perpetual loop.")
            break

        try:
            state = run_pipeline()
        except Exception as e:
            logger.error(f"Pipeline crash in cycle {cycle}: {e}", exc_info=True)
            logger.info("Sleeping 60s before retry...")
            time.sleep(60)
            continue

        # Check if there's still work to do
        model_queue = load_queue_config()
        all_done = True
        for mc in model_queue:
            completed = state["completed"].get(mc["short_name"], [])
            enabled = mc.get("enabled_domains", DOMAINS)
            remaining = [d for d in DOMAINS if d in enabled and d not in completed]
            if remaining:
                all_done = False
                break

        if all_done:
            logger.info("All models and domains complete. Sleeping 5 minutes before re-checking...")
            logger.info("To add new work: edit adapters/lori_moe/queue_config.json")
            logger.info("To stop: touch adapters/lori_moe/STOP_PIPELINE")
            time.sleep(300)  # 5 min sleep, then re-check
        else:
            logger.info("More work found, continuing immediately...")


if __name__ == "__main__":
    try:
        perpetual_loop()
    except KeyboardInterrupt:
        logger.info("\n  Pipeline interrupted by user. Progress saved.")
    except Exception as e:
        logger.error(f"\n  Pipeline crash: {e}", exc_info=True)
        raise
