#!/usr/bin/env python3
"""
LoRI-MoE SHADOW PIPELINE (VRAM Scavenger)

This script runs concurrently to the main pipeline. 
It specifically targets pure-Transformer models (like Qwen 2.5 0.5B) 
which require very little VRAM (~3-4GB) to utilize the remaining 
graphics compute without competing for the GDN layers' memory.

Usage:
    nohup python scripts/shadow_pipeline.py >> logs/lori_moe/shadow.log 2>&1 &
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = "/home/learner/Desktop/mewtwo"
VENV_PYTHON = f"{PROJECT_ROOT}/.venv/bin/python"
CHECKPOINT_BASE = f"{PROJECT_ROOT}/checkpoints/lori_moe"
LOG_DIR = f"{PROJECT_ROOT}/logs/lori_moe"
DOMAINS = ["math", "code", "science", "legal", "medical"]

SHADOW_QUEUE = [
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "short_name": "qwen2.5_0.5b",
        "batch_size": 2,     # Extremely safe batch size for background
        "grad_accum": 8,     # Effective 16
        "gradient_checkpointing": True,
    }
]

TRAINING_CONFIG = {
    "rank": 32,
    "sparsity": 0.8,
    "epochs": 3,
    "max_train_samples": 10000,
    "lr": 2e-4,
    "max_seq_length": 1024,
}

# ─── Logging Setup ──────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SHADOW] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{LOG_DIR}/shadow.log"),
    ],
)
logger = logging.getLogger("shadow")

def banner(msg, char="░"):
    line = char * 55
    logger.info(f"\n{line}\n  {msg}\n{line}")

# ─── State Tracking ────────────────────────────────────────────────────────────

STATE_FILE = f"{CHECKPOINT_BASE}/shadow_state.json"

def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}, "failed": {}}

def save_state(state):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─── Training Function ─────────────────────────────────────────────────────────

def train_single_adapter(model_config, domain):
    output_dir = f"{CHECKPOINT_BASE}/{model_config['short_name']}"
    
    cmd = [
        VENV_PYTHON, "-m", "src.lori_moe.training.train_lori_adapter",
        "--domain", domain,
        "--base_model", model_config["model_id"],
        "--output_dir", output_dir,
        "--rank", str(TRAINING_CONFIG["rank"]),
        "--sparsity", str(TRAINING_CONFIG["sparsity"]),
        "--epochs", str(TRAINING_CONFIG["epochs"]),
        "--max_train_samples", str(TRAINING_CONFIG["max_train_samples"]),
        "--batch_size", str(model_config["batch_size"]),
        "--grad_accum", str(model_config["grad_accum"]),
        "--lr", str(TRAINING_CONFIG["lr"]),
        "--vram_fraction", "0.20", # HARD CAP to 20% (6.5GB) VRAM
    ]
    
    if model_config["gradient_checkpointing"]:
        cmd.append("--gradient_checkpointing")
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        logger.info(f"  Launching {domain} adapter...")
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=7200
        )
        if result.returncode == 0:
            return True, None
        
        stderr = result.stderr
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            return False, "OOM"
        else:
            return False, "ERROR"
            
    except Exception as e:
        return False, "EXCEPTION"

# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def run_shadow_pipeline():
    state = load_state()
    banner("LoRI-MoE SHADOW VRAM SCAVENGER")
    
    for model_config in SHADOW_QUEUE:
        model_name = model_config["short_name"]
        model_id = model_config["model_id"]
        
        completed_domains = state["completed"].get(model_name, [])
        remaining = [d for d in DOMAINS if d not in completed_domains]
        
        if not remaining:
            logger.info(f"  {model_name} already complete in shadow queue.")
            continue
            
        banner(f"SHADOW MODEL: {model_id}")
        
        for domain in remaining:
            logger.info(f"  ┌─ Starting background domain: {domain}")
            
            # Simple retry for OOM without changing batch size (batch=4 is already minimum safe)
            success, err = train_single_adapter(model_config, domain)
            
            if success:
                if model_name not in state["completed"]:
                    state["completed"][model_name] = []
                state["completed"][model_name].append(domain)
                save_state(state)
                logger.info(f"  └─ [{domain}] Done ✅")
            else:
                logger.error(f"  └─ [{domain}] FAILED ({err}) ❌ - Pausing shadow pipeline to protect main training.")
                time.sleep(30)
                sys.exit(1) # Kill shadow worker on failure to preserve main process
                
    banner("SHADOW PIPELINE COMPLETE")

if __name__ == "__main__":
    run_shadow_pipeline()
