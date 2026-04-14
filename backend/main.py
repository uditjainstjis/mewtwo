from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import time
from orchestrator import Orchestrator
from dynamic_mlx_inference import DynamicEngine
<<<<<<< HEAD
from runtime_backend import get_runtime_summary
=======
from agent_cluster import AdversarialAgentCluster
from collaborative_reasoning import CollaborativeReasoner
>>>>>>> origin/main

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
agent_cluster = None
collaborative_reasoner = None
registry_data = {}

@app.on_event("startup")
def startup_event():
    global engine, orchestrator, agent_cluster, collaborative_reasoner, registry_data
    with open("expert_registry.json", "r") as f:
        registry_data = json.load(f)
        
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry_data)
    orchestrator = Orchestrator("expert_registry.json", base_engine=engine)
    agent_cluster = AdversarialAgentCluster(engine=engine, orchestrator=orchestrator)
    collaborative_reasoner = CollaborativeReasoner(
        engine,
        "expert_registry.json",
        parallel_workers=1,
        backend_dir=".",
        router_model_name_or_path=os.environ.get("TCAR_ROUTER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        router_adapter_path=os.environ.get("TCAR_ROUTER_ADAPTER") or None,
    )

class ChatRequest(BaseModel):
    query: str
    mode: str = "standard"
    max_experts: int = 2


@app.on_event("shutdown")
def shutdown_event():
    global collaborative_reasoner
    if collaborative_reasoner is not None:
        collaborative_reasoner.shutdown()

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.mode == "cluster_strict":
        start = time.time()
        cluster_result = agent_cluster.run(req.query)
        total_dur = time.time() - start
        return {
            "response": cluster_result.response,
            "mode": "cluster_strict",
            "passed": cluster_result.passed,
            "veto_reasons": cluster_result.veto_reasons,
            "trace": cluster_result.trace,
            "metrics": {
                "total_latency_ms": round(total_dur * 1000, 2),
                "proxy_enabled": getattr(agent_cluster.proxy, "enabled", False),
                **cluster_result.metrics,
            },
        }

    if req.mode == "collaborative_reasoning":
        start = time.time()
        result = collaborative_reasoner.run(req.query, max_experts=req.max_experts)
        total_dur = time.time() - start
        return {
            "response": result.final_answer,
            "mode": "collaborative_reasoning",
            "router": {
                "thinking": result.router.thinking,
                "experts": result.router.experts,
                "raw_text": result.router.raw_text,
            },
            "branches": [
                {
                    "expert": branch.expert,
                    "answer": branch.answer,
                    "latency_s": round(branch.latency_s, 3),
                    "mode": branch.mode,
                }
                for branch in result.branches
            ],
            "metrics": {
                "router_latency_ms": round(result.router_latency_s * 1000, 2),
                "branch_latency_ms": round(result.branch_latency_s * 1000, 2),
                "refine_latency_ms": round(result.refine_latency_s * 1000, 2),
                "total_latency_ms": round(total_dur * 1000, 2),
                "parallel_workers": result.parallel_workers,
            },
        }

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
    summary = get_runtime_summary("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    summary["status"] = "loaded"
    return summary
