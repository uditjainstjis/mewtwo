#!/usr/bin/env python3
"""
Synapta Investor Demo — FastAPI Backend
Serves the Nemotron-30B model with live Token-Level Routing visualization.
Supports WebSocket streaming with per-token adapter metadata.
"""
import asyncio
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

from peft import PeftModel
from scripts.train_neural_router_v2 import SimpleNeuralRouter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    TextIteratorStreamer,
)
from threading import Thread

PROJECT = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH = str(PROJECT / "models" / "nemotron")
ADAPTER_BASE = PROJECT / "adapters" / "nemotron_30b"
DEMO_DIR = PROJECT / "src" / "demo"
RESULTS_DIR = PROJECT / "results" / "nemotron"

ROUTER_PATH = "/home/learner/Desktop/mewtwo/adapters/routers/neural_mlp_router.pt"
DOMAIN_MAP = {0: "math", 1: "code", 2: "science"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synapta-demo")

# ─── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(title="Synapta — Dynamic Token-Level Routing Demo")
app.mount("/static", StaticFiles(directory=str(DEMO_DIR / "static")), name="static")

# ─── Global Model State ─────────────────────────────────────────────────
model = None
tok = None
base_model = None
HybridCache = None
MODEL_LOADED = False
neural_router = None

def load_model():
    """Load Nemotron-30B with all 3 domain adapters into VRAM."""
    global model, tok, base_model, HybridCache, MODEL_LOADED, neural_router

    log.info("🚀 Loading Nemotron-30B (4-bit) + 3 Domain Adapters...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    base_model.eval()

    # Load all 3 adapters into VRAM
    math_path = str(ADAPTER_BASE / "math" / ("best" if (ADAPTER_BASE / "math" / "best").exists() else "final"))
    code_path = str(ADAPTER_BASE / "code" / ("best" if (ADAPTER_BASE / "code" / "best").exists() else "final"))
    sci_path = str(ADAPTER_BASE / "science" / ("best" if (ADAPTER_BASE / "science" / "best").exists() else "final"))

    model = PeftModel.from_pretrained(base_model, math_path, adapter_name="math", is_trainable=False)
    model.load_adapter(code_path, adapter_name="code")
    model.load_adapter(sci_path, adapter_name="science")
    model.eval()

    # Load Neural Router
    neural_router = SimpleNeuralRouter().to("cuda")
    neural_router.load_state_dict(torch.load(ROUTER_PATH))
    neural_router.eval()

    # Cache class for the hybrid architecture
    HybridCache = getattr(sys.modules[base_model.__class__.__module__], "HybridMambaAttentionDynamicCache")

    MODEL_LOADED = True
    log.info("✅ Model loaded with Math, Code, Science adapters in VRAM.")


# ─── Routing Logic ───────────────────────────────────────────────────────

ADAPTER_COLORS = {"code": "#00d4aa", "math": "#6366f1", "science": "#f59e0b"}
ADAPTER_LABELS = {"code": "Code ⚡", "math": "Math 🔢", "science": "Science 🔬"}


class StreamingTokenRouter(LogitsProcessor):
    """Routes adapters every 10 tokens and tracks swap events."""

    def __init__(self, websocket, tokenizer, loop=None):
        self.websocket = websocket
        self.tok = tokenizer
        self.loop = loop
        self.current_adapter = "code"
        self.swaps = 0
        self.swap_events = []

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 15 == 0:
            # Neural Routing
            with torch.no_grad():
                last_token_idx = input_ids[:, -1:]
                embeds = base_model.backbone.embeddings(last_token_idx).squeeze(1).float()
                logits = neural_router(embeds)
                pred_idx = logits.argmax(dim=-1).item()
                new_ad = DOMAIN_MAP[pred_idx]
            
            if new_ad != self.current_adapter:
                old_ad = self.current_adapter
                model.set_adapter(new_ad)
                self.current_adapter = new_ad
                self.swaps += 1
                self.swap_events.append({"token_idx": input_ids.shape[1], "from": old_ad, "to": new_ad})
                if self.loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send_json({"type": "swap", "from": old_ad, "to": new_ad, "token_idx": input_ids.shape[1]}),
                        self.loop
                    )
        return scores


# ─── Generation Modes ────────────────────────────────────────────────────

async def generate_streamed(prompt: str, websocket: WebSocket, mode: str = "routed", max_tokens: int = 512):
    """Generate tokens and stream each one with metadata via WebSocket."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    past_key_values = HybridCache(
        base_model.config,
        batch_size=inputs["input_ids"].shape[0],
        dtype=torch.bfloat16,
        device=model.device,
    )

    # Set up streamer for token-by-token output
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    router = None
    if mode == "routed":
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        router = StreamingTokenRouter(websocket, tok, loop)
        processors = LogitsProcessorList([router])
    else:
        # For "merged" or "single" modes, just use one adapter
        model.set_adapter("code")
        processors = LogitsProcessorList()

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "pad_token_id": tok.pad_token_id,
        "repetition_penalty": 1.3,
        "use_cache": True,
        "past_key_values": past_key_values,
        "logits_processor": processors,
        "streamer": streamer,
    }

    # Run generation in background thread
    t_start = time.time()
    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    token_idx = 0
    full_text = ""
    try:
        for new_text in streamer:
            token_idx += 1
            full_text += new_text

            # Determine current adapter for this token
            current_adapter = router.current_adapter if router else "code"

            await websocket.send_json({
                "type": "token",
                "text": new_text,
                "adapter": current_adapter,
                "color": ADAPTER_COLORS.get(current_adapter, "#ffffff"),
                "label": ADAPTER_LABELS.get(current_adapter, current_adapter),
                "token_idx": token_idx,
                "elapsed": round(time.time() - t_start, 2),
            })
            await asyncio.sleep(0)  # Yield to event loop
    except WebSocketDisconnect:
        pass

    thread.join()
    t_total = time.time() - t_start

    # Send completion summary
    try:
        await websocket.send_json({
            "type": "done",
            "total_tokens": token_idx,
            "total_time": round(t_total, 2),
            "tokens_per_sec": round(token_idx / t_total, 1) if t_total > 0 else 0,
            "swap_events": router.swap_events if router else [],
            "full_text": full_text,
        })
    except:
        pass

    # Cleanup
    del past_key_values
    gc.collect()
    torch.cuda.empty_cache()


# ─── Routes ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(str(DEMO_DIR / "static" / "index.html"))


@app.get("/api/status")
async def status():
    return {
        "model_loaded": MODEL_LOADED,
        "model": "Nemotron-3-Nano-30B",
        "adapters": ["math", "code", "science"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1) if torch.cuda.is_available() else 0,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
    }


@app.get("/api/research")
async def research_data():
    """Return all benchmark results for the dashboard."""
    data = {}
    for fname in ["master_results.json", "token_routing_results.json", "cold_swap_metrics.json"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data[fname.replace(".json", "")] = json.load(f)
    return data


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            req = json.loads(raw)
            prompt_text = req.get("prompt", "")
            mode = req.get("mode", "routed")  # "routed" or "single"
            max_tokens = min(req.get("max_tokens", 512), 1024)

            # Build chat prompt
            chat_prompt = tok.apply_chat_template(
                [
                    {"role": "system", "content": "You are Synapta — a multi-domain AI assistant powered by dynamic token-level adapter routing. Provide clear, accurate, expert-level answers."},
                    {"role": "user", "content": prompt_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            await generate_streamed(chat_prompt, websocket, mode=mode, max_tokens=max_tokens)
    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")


# ─── Startup ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False, workers=1)
