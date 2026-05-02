#!/usr/bin/env python3
"""
Synapta Demo Server — REAL inference with Nemotron-30B + LoRA adapters + Router.

Loads the base model, 3 domain adapters (math/code/science), and the trained Neural MLP
router. Serves a WebSocket endpoint that streams real tokens with real routing decisions.
"""

import os
import re
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
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "nemotron_30b"
NEURAL_ROUTER_PATH = PROJECT_ROOT / "adapters" / "routers" / "neural_mlp_router.pt"

DOMAINS = ["math", "code", "science"]
CHUNK_SIZE = 10  # Optimal chunk size from research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("synapta-server")


def _task_style(prompt: str) -> tuple[str, Optional[str]]:
    text = prompt.lower()
    if "answer with only one letter" in text:
        return "choice", None
    if "\\boxed{" in prompt or "solve the following math problem" in text or "compute\\n\\[" in text:
        return "math", "math"
    if "```python" in text or "complete the code" in text or "write a function" in text or "\ndef " in text:
        return "code", "code"
    if "prove" in text or "derive" in text or "equation" in text:
        return "math", "math"
    if "algorithm" in text or "time complexity" in text:
        return "code", "code"
    if any(word in text for word in ["physics", "chemistry", "electrical", "thermal", "serdes", "bandwidth"]):
        return "science", "science"
    return "general", None


def _first_complete_boxed_end(text: str) -> Optional[int]:
    start = text.find("\\boxed{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _first_complete_code_fence_end(text: str) -> Optional[int]:
    start = text.find("```")
    if start == -1:
        return None
    end = text.find("```", start + 3)
    if end == -1:
        return None
    return end + 3


def _trim_at_first_marker(text: str, markers: list[str]) -> tuple[str, bool]:
    cut = None
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    if cut is None:
        return text, False
    return text[:cut].rstrip(), True


def _should_stop_generation(prompt: str, text: str, thinking: bool) -> tuple[str, bool]:
    if not thinking:
        style, _ = _task_style(prompt)
        trimmed, hit_marker = _trim_at_first_marker(
            text,
            ["<think>", "</think>", "\nUser:", "\nSystem:", "User:", "System:"],
        )
        if hit_marker:
            return trimmed, True

        if style == "math" and "\\boxed{" in prompt:
            boxed_end = _first_complete_boxed_end(text)
            if boxed_end is not None:
                return text[:boxed_end].rstrip(), True

        if style == "choice":
            match = re.search(r"\b([ABCD])\b", text)
            if match:
                return match.group(1), True

        if style == "code":
            fence_end = _first_complete_code_fence_end(text)
            if fence_end is not None:
                return text[:fence_end].rstrip(), True

    return text, False


def _apply_hint_to_weights(
    weights: dict[str, float],
    hint_domain: Optional[str],
    hint_strength: float,
) -> tuple[str, dict[str, float]]:
    if not hint_domain or hint_domain not in weights or hint_strength <= 0:
        selected = max(weights, key=weights.get)
        return selected, weights

    adjusted = dict(weights)
    adjusted[hint_domain] += hint_strength
    total = sum(adjusted.values()) or 1.0
    adjusted = {k: v / total for k, v in adjusted.items()}
    selected = max(adjusted, key=adjusted.get)
    return selected, adjusted

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.router = NeuralMLPRouter(hidden_dim=2688, num_domains=3).to(self.device)
        if NEURAL_ROUTER_PATH.exists():
            ckpt = torch.load(NEURAL_ROUTER_PATH, map_location=self.device, weights_only=True)
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
        # index 0 is embeddings, so index 33 is output of layer 32
        hidden = outputs.hidden_states[33][:, -1, :]  
        return hidden.squeeze(0)

    def set_adapter(self, domain: str):
        if domain == self.current_adapter:
            return
        if domain == "none":
            self.model.base_model.disable_adapter_layers()
            if hasattr(self.model, "_adapters_disabled"):
                self.model._adapters_disabled = True
            self.current_adapter = "none"
            return
        try:
            self.model.base_model.enable_adapter_layers()
            if hasattr(self.model, "_adapters_disabled"):
                self.model._adapters_disabled = False
            self.model.set_adapter(domain)
            self.current_adapter = domain
        except Exception as e:
            logger.warning(f"Failed to set adapter {domain}: {e}")

    def format_prompt(self, user_text: str, thinking: bool = False) -> str:
        # Raw completion prompts work better with the non-instruct base model than
        # chat transcript wrappers, which trigger User:/System: continuation.
        style, _ = _task_style(user_text)
        if thinking:
            return f"{user_text.rstrip()}\n\n<think>"
        if style == "math":
            suffix = "\nRespond with only the final answer in \\boxed{}.\n"
        elif style == "choice":
            suffix = "\nRespond with exactly one letter: A, B, C, or D.\n"
        elif style == "code":
            suffix = "\nReturn only code. Do not explain.\n"
        else:
            suffix = "\nRespond directly with no meta-commentary.\n"
        return user_text.rstrip() + suffix

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
        "device": str(engine.device),
        "pid": os.getpid(),
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
            thinking = data.get("thinking", False)
            do_sample = bool(data.get("do_sample", True))
            temperature = float(data.get("temperature", 0.6))
            top_p = float(data.get("top_p", 0.9))
            repetition_penalty = float(data.get("repetition_penalty", 1.1))
            chunk_size = max(1, min(int(data.get("chunk_size", CHUNK_SIZE)), 128))
            routing_interval = max(1, min(int(data.get("routing_interval", chunk_size)), 256))
            min_tokens_before_swap = max(0, min(int(data.get("min_tokens_before_swap", routing_interval)), 512))
            domain_hint_strength = max(0.0, min(float(data.get("domain_hint_strength", 0.18)), 1.0))
            swap_margin = max(0.0, min(float(data.get("swap_margin", 0.2)), 1.0))
            lock_prompt_domain = bool(data.get("lock_prompt_domain", True))
            prompt_anchor = bool(data.get("prompt_anchor", True))

            if not prompt or not engine.ready:
                await ws.send_json({"type": "error", "message": "Not ready or empty prompt"})
                continue

            import queue
            msg_queue = queue.Queue()

            def run_generation():
                _generate_sync(
                    msg_queue=msg_queue,
                    prompt=prompt,
                    mode=mode,
                    adapter=adapter,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    chunk_size=chunk_size,
                    routing_interval=routing_interval,
                    min_tokens_before_swap=min_tokens_before_swap,
                    domain_hint_strength=domain_hint_strength,
                    swap_margin=swap_margin,
                    lock_prompt_domain=lock_prompt_domain,
                    prompt_anchor=prompt_anchor,
                )

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


def _generate_sync(
    msg_queue,
    prompt: str,
    mode: str,
    adapter: str,
    max_tokens: int,
    thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chunk_size: int,
    routing_interval: int,
    min_tokens_before_swap: int,
    domain_hint_strength: float,
    swap_margin: float,
    lock_prompt_domain: bool,
    prompt_anchor: bool,
):
    t0 = time.time()
    try:
        formatted = engine.format_prompt(prompt, thinking)
        encoded = engine.tokenizer(formatted, return_tensors="pt")
        input_ids = encoded["input_ids"].to(engine.device)
        attention_mask = encoded["attention_mask"].to(engine.device)
        task_style, hinted_domain = _task_style(prompt)
        prompt_locked_domain = hinted_domain if lock_prompt_domain else None

        if mode == "routed":
            # INITIAL ROUTING: Use ONLY the raw user prompt without System Instruction pollution
            # to get a clean, unbiased classification of the user's intent.
            clean_prompt_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(engine.device)
            clean_hidden = engine.get_hidden_state_layer32(clean_prompt_ids)
            current_domain, weights = engine.router.route(clean_hidden)
            if prompt_anchor:
                current_domain, weights = _apply_hint_to_weights(weights, hinted_domain, domain_hint_strength)
            
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
        output_text = ""
        tokens_since_route = 0

        code_bans = [
            "####",
            "\\boxed",
            "<think>",
            "</think>",
            "\nTask:",
            "\nQuestion:",
            "\nUser:",
            "\nSystem:",
        ]
        general_bans = [
            "<think>",
            "</think>",
            "\nTask:",
            "\nQuestion:",
            "\nUser:",
            "\nSystem:",
        ]
        bad_words_code = [engine.tokenizer.encode(w, add_special_tokens=False) for w in code_bans]
        bad_words_general = [engine.tokenizer.encode(w, add_special_tokens=False) for w in general_bans]

        while total_generated < max_tokens:
            current_chunk = min(chunk_size, max_tokens - total_generated)

            with torch.no_grad():
                bad_words = bad_words_code if current_domain == "code" else bad_words_general
                gen_kwargs = {
                    "max_new_tokens": current_chunk,
                    "do_sample": do_sample,
                    "pad_token_id": engine.tokenizer.pad_token_id,
                    "bad_words_ids": bad_words,
                    "repetition_penalty": repetition_penalty,
                    "attention_mask": attention_mask,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                if mode == "naked":
                    with engine.model.disable_adapter():
                        output = engine.model.generate(input_ids, **gen_kwargs)
                else:
                    output = engine.model.generate(input_ids, **gen_kwargs)

            new_token_ids = output[0][input_ids.shape[1]:]
            if len(new_token_ids) == 0:
                break

            hit_eos = False
            for tid in new_token_ids:
                if tid.item() == engine.tokenizer.eos_token_id:
                    hit_eos = True
                    break

                token_text = engine.tokenizer.decode([tid.item()], skip_special_tokens=False)
                output_text += token_text

                output_text, should_stop = _should_stop_generation(prompt, output_text, thinking)
                if should_stop:
                    hit_eos = True
                    break

                total_generated += 1
                tokens_since_route += 1
                elapsed = time.time() - t0
                
                msg_queue.put({
                    "type": "token", "text": token_text, "domain": current_domain,
                    "index": total_generated, "speed": round(total_generated / max(elapsed, 0.01), 1),
                })

            if hit_eos:
                break

            input_ids = output
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

            if mode == "routed" and total_generated < max_tokens and tokens_since_route >= routing_interval:
                tokens_since_route = 0
                hidden = engine.get_hidden_state_layer32(input_ids)
                new_domain, new_weights = engine.router.route(hidden)
                if prompt_anchor:
                    new_domain, new_weights = _apply_hint_to_weights(new_weights, hinted_domain, domain_hint_strength)

                max_conf = max(new_weights.values()) if new_weights else 0
                
                # ENTROPY GATE HYSTERESIS:
                # In a 3-way Softmax, max_conf is guaranteed to be >= 0.33.
                # If currently routing, require conf to drop below 0.50 to become none (low confidence).
                # If currently none, require conf to exceed 0.70 to become routed (high confidence).
                if current_domain == "none":
                    if max_conf < 0.70:
                        new_domain = "none"
                else:
                    if max_conf < 0.50:
                        new_domain = current_domain if prompt_locked_domain else "none"

                if prompt_locked_domain and current_domain == prompt_locked_domain and total_generated < max(min_tokens_before_swap, routing_interval):
                    new_domain = current_domain

                if new_domain != current_domain:
                    # Interspecial hysteresis (e.g. math -> code)
                    curr_weight = new_weights.get(current_domain, 0.0)
                    new_weight = new_weights.get(new_domain, 0.0)
                    if prompt_locked_domain and current_domain == prompt_locked_domain and new_domain != prompt_locked_domain:
                        if total_generated < min_tokens_before_swap:
                            new_domain = current_domain
                        elif new_weight <= curr_weight + max(swap_margin, 0.25):
                            new_domain = current_domain

                    if new_domain != current_domain and (new_domain == "none" or new_weight > curr_weight + swap_margin):
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
            "full_text": output_text,
            "mode": mode,
            "adapter": adapter,
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
