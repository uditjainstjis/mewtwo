from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
from orchestrator import Orchestrator
from dynamic_mlx_inference import DynamicEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine = None
orchestrator = None
registry_data = {}

@app.on_event("startup")
def startup_event():
    global engine, orchestrator, registry_data
    with open("expert_registry.json", "r") as f:
        registry_data = json.load(f)
        
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry_data)
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    start = time.time()
    weights, reasoning = orchestrator.route(req.query, top_k=2)
    route_dur = time.time() - start
    
    prompt = f"<|im_start|>user\n{req.query}<|im_end|>\n<|im_start|>assistant\n"
    ans, gen_dur = engine.generate(prompt, routing_weights=weights, max_tokens=100)
    
    return {
        "response": ans,
        "routing_weights": weights,
        "routing_reasoning": reasoning,
        "metrics": {
            "routing_latency_ms": round(route_dur * 1000, 2),
            "generation_latency_ms": round(gen_dur * 1000, 2)
        }
    }

@app.get("/api/adapters")
def get_adapters():
    return registry_data

@app.get("/api/model")
def get_model():
    return {"model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "status": "loaded"}
