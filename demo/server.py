#!/usr/bin/env python3
"""
Synapta Demo Server — REAL inference with Nemotron-30B + LoRA adapters + Router.

Loads the base model, 3 domain adapters (math/code/science), and the trained Neural MLP
router. Serves a WebSocket endpoint that streams real tokens with real routing decisions.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# ── Project paths ──
PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH = PROJECT_ROOT / "models" / "nemotron"
ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "nemotron_lori" / "adapters"
NEURAL_ROUTER_PATH = PROJECT_ROOT / "router_adapters" / "neural_mlp_router.pt"

DOMAINS = ["math", "code", "science"]
CHUNK_SIZE = 10  # Optimal chunk size from research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("synapta-server")

# ── Router (Neural MLP on Layer 32) ──
class NeuralMLPRouter(nn.Module):
    """
    Trained Neural MLP Router operating on Layer 32 hidden states.
    Architecture: LayerNorm(2688) -> Linear(2688, 256) -> SiLU -> Dropout(0.1) -> Linear(256, 3)
    """
    def __init__(self, hidden_dim: int = 2688, num_domains: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_domains)
        )
        self.domains = DOMAINS
        self.router_type = "neural_mlp"

    def forward(self, hidden_state):
        # hidden_state: (D,) or (1, D)
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.squeeze(0)
        
        x = self.norm(hidden_state)
        logits = self.mlp(x)
        return logits

    def route(self, hidden_state: torch.Tensor) -> tuple:
        """Classify a hidden state into a domain."""
        with torch.no_grad():
            logits = self.forward(hidden_state.float())
            probs = F.softmax(logits, dim=-1)
            weights = {d: probs[i].item() for i, d in enumerate(self.domains)}
            selected = max(weights, key=weights.get)
            return selected, weights

# ── Global model state ──
class SynaptaEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.router = None
        self.ready = False
        self.current_adapter = None
        self.load_time = 0

    def load(self):
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("🚀 SYNAPTA ENGINE — Loading Nemotron-30B")
        logger.info("=" * 60)

        # Tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        logger.info("Loading Nemotron-30B in 4-bit...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        # Load adapters
        logger.info("Loading LoRA adapters...")
        adapter_loaded = []
        for domain in DOMAINS:
            for subdir in ["dare_sparsified", "best", "final"]:
                adapter_path = ADAPTER_DIR / domain / subdir
                if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
                    try:
                        if len(adapter_loaded) == 0:
                            self.model = PeftModel.from_pretrained(
                                self.model, str(adapter_path),
                                adapter_name=domain, is_trainable=False
                            )
                        else:
                            self.model.load_adapter(str(adapter_path), adapter_name=domain)
                        adapter_loaded.append(domain)
                        logger.info(f"  ✅ {domain} adapter loaded from {subdir}/")
                        break
                    except Exception as e:
                        logger.warning(f"  ❌ Failed to load {domain}/{subdir}: {e}")
                        continue
        self.model.eval()

        # Router
        self.router = NeuralMLPRouter(hidden_dim=2688, num_domains=3).to(self.model.device)
        if NEURAL_ROUTER_PATH.exists():
            ckpt = torch.load(NEURAL_ROUTER_PATH, map_location=self.model.device, weights_only=True)
            self.router.load_state_dict(ckpt if 'state_dict' not in ckpt else ckpt['state_dict'], strict=False)
            logger.info("✅ Loaded trained Neural MLP Router")
        else:
            logger.warning("❌ Trained Neural router not found, using random weights")

        self.router.eval()
        self.load_time = time.time() - t0
        self.ready = True

        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("=" * 60)
        logger.info(f"✅ SYNAPTA ENGINE READY in {self.load_time:.1f}s | VRAM: {vram_gb:.1f} GB")
        logger.info("=" * 60)

    def get_hidden_state_layer32(self, input_ids, attention_mask=None):
        """Get hidden state from layer 32 for the Neural MLP router."""
        with torch.no_grad():
            with self.model.disable_adapter():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        # 33rd item in tuple is layer 32 output (index 0 is embeddings)
        hidden = outputs.hidden_states[32][:, -1, :]  
        return hidden.squeeze(0)

    def set_adapter(self, domain: str):
        if domain == self.current_adapter:
            return
        try:
            self.model.set_adapter(domain)
            self.current_adapter = domain
        except Exception as e:
            logger.warning(f"Failed to set adapter {domain}: {e}")

    def format_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"<extra_id_0>System\n\n<extra_id_1>User\n{user_text}\n<extra_id_1>Assistant\n"

engine = SynaptaEngine()

# ── FastAPI app ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting model load (this takes ~1-2 minutes)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Synapta Demo Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_DIR = Path(__file__).parent

@app.get("/")
async def root():
    return FileResponse(str(DEMO_DIR / "index.html"))

@app.get("/styles.css")
async def serve_css():
    return FileResponse(str(DEMO_DIR / "styles.css"), media_type="text/css")

@app.get("/simulation.js")
async def serve_js():
    return FileResponse(str(DEMO_DIR / "simulation.js"), media_type="application/javascript")

@app.get("/api/status")
async def status():
    return {
        "ready": engine.ready,
        "adapters": DOMAINS,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0,
        "load_time_s": round(engine.load_time, 1),
    }

@app.websocket("/ws/generate")
async def websocket_generate(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            prompt = data.get("prompt", "").strip()
            mode = data.get("mode", "routed")
            adapter = data.get("adapter", "code")
            max_tokens = min(data.get("max_tokens", 512), 1024)

            if not prompt or not engine.ready:
                await ws.send_json({"type": "error", "message": "Not ready or empty prompt"})
                continue

            import queue
            msg_queue = queue.Queue()

            def run_generation():
                _generate_sync(msg_queue, prompt, mode, adapter, max_tokens)

            loop = asyncio.get_event_loop()
            gen_task = loop.run_in_executor(None, run_generation)

            done = False
            while not done:
                try:
                    while True:
                        try:
                            msg = msg_queue.get_nowait()
                            await ws.send_json(msg)
                            if msg.get("type") in ("done", "error"):
                                done = True
                                break
                        except queue.Empty:
                            break
                    if not done:
                        await asyncio.sleep(0.02)
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    done = True

            await gen_task

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


def _generate_sync(msg_queue, prompt: str, mode: str, adapter: str, max_tokens: int):
    t0 = time.time()
    try:
        formatted = engine.format_prompt(prompt)
        input_ids = engine.tokenizer.encode(formatted, return_tensors="pt").to("cuda")

        if mode == "routed":
            hidden = engine.get_hidden_state_layer32(input_ids)
            current_domain, weights = engine.router.route(hidden)
            engine.set_adapter(current_domain)
            msg_queue.put({"type": "route", "domain": current_domain, "weights": {k: round(v, 4) for k, v in weights.items()}})
        elif mode == "single":
            current_domain = adapter
            engine.set_adapter(current_domain)
            msg_queue.put({"type": "route", "domain": current_domain, "weights": {d: 1.0 if d == adapter else 0.0 for d in DOMAINS}})
        else:
            current_domain = "none"
            msg_queue.put({"type": "route", "domain": "none", "weights": {}})

        total_generated = 0
        swap_count = 0

        while total_generated < max_tokens:
            chunk_size = min(CHUNK_SIZE, max_tokens - total_generated)

            with torch.no_grad():
                if mode == "naked":
                    with engine.model.disable_adapter():
                        output = engine.model.generate(
                            input_ids, max_new_tokens=chunk_size, do_sample=True, temperature=0.6,
                            top_p=0.9, pad_token_id=engine.tokenizer.pad_token_id,
                        )
                else:
                    output = engine.model.generate(
                        input_ids, max_new_tokens=chunk_size, do_sample=True, temperature=0.6,
                        top_p=0.9, pad_token_id=engine.tokenizer.pad_token_id,
                    )

            new_token_ids = output[0][input_ids.shape[1]:]
            if len(new_token_ids) == 0:
                break

            hit_eos = False
            for tid in new_token_ids:
                if tid.item() == engine.tokenizer.eos_token_id:
                    hit_eos = True
                    break

                token_text = engine.tokenizer.decode([tid.item()], skip_special_tokens=False)
                total_generated += 1
                elapsed = time.time() - t0
                
                msg_queue.put({
                    "type": "token", "text": token_text, "domain": current_domain,
                    "index": total_generated, "speed": round(total_generated / max(elapsed, 0.01), 1),
                })

            if hit_eos:
                break

            input_ids = output

            if mode == "routed" and total_generated < max_tokens:
                hidden = engine.get_hidden_state_layer32(input_ids)
                new_domain, new_weights = engine.router.route(hidden)

                if new_domain != current_domain:
                    swap_count += 1
                    msg_queue.put({
                        "type": "swap", "from": current_domain, "to": new_domain,
                        "at_token": total_generated, "weights": {k: round(v, 4) for k, v in new_weights.items()}
                    })
                    current_domain = new_domain
                    engine.set_adapter(current_domain)

        elapsed = time.time() - t0
        msg_queue.put({
            "type": "done", "total_tokens": total_generated, "swaps": swap_count,
            "elapsed_s": round(elapsed, 2), "final_domain": current_domain,
            "speed": round(total_generated / max(elapsed, 0.01), 1),
        })

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        msg_queue.put({"type": "error", "message": str(e)})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logger.info(f"Starting Synapta server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
