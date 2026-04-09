from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from md_dataset import prepare_md_items  # noqa: E402


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _norm(pred) == _norm(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p = _norm(pred).split()
    r = _norm(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    p_count, r_count = {}, {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in r:
        r_count[t] = r_count.get(t, 0) + 1
    overlap = sum(min(p_count[t], r_count.get(t, 0)) for t in p_count)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


_sem_model = None


def semantic_similarity(pred: str, ref: str) -> float:
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer

        _sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = _sem_model.encode([pred or "", ref or ""], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def build_weights_two_domains(registry: dict, d1: str, d2: str, w1: float, w2: float) -> dict:
    out = {d: 0.0 for d in registry}
    if d1 in out:
        out[d1] = w1
    if d2 in out:
        out[d2] = w2
    return out


def query_ollama(
    prompt: str,
    model_name: str,
    num_predict: int,
    extra_options: dict | None = None,
) -> tuple[str, float, str | None]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    start = time.time()
    try:
        options = {"num_predict": num_predict}
        if extra_options:
            options.update(extra_options)
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=180,
        )
        duration = time.time() - start
        data = r.json()
        if data.get("error"):
            return "", duration, str(data["error"])
        return str(data.get("response") or ""), duration, None
    except Exception as e:
        return "", time.time() - start, str(e)


def run_qwen_method(engine, registry: dict, item: dict, method: str) -> str:
    doms = item.get("required_adapters") or item.get("domains") or []
    d1, d2 = doms[0], doms[1]
    prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
    n_layers = engine._num_layers or 28
    late_start = max(1, n_layers // 2)
    late_last_q = max(1, int(n_layers * 0.75))
    early_max = max(0, n_layers // 3 - 1)
    w_mix = build_weights_two_domains(registry, d1, d2, 0.5, 0.5)
    w_a = build_weights_two_domains(registry, d1, d2, 1.0, 0.0)
    w_b = build_weights_two_domains(registry, d1, d2, 0.0, 1.0)

    from dynamic_mlx_inference import set_global_clamp

    if method == "weighted_merge":
        engine.set_adapter_layer_gate(0, -1)
        return engine.generate(prompt, routing_weights=w_mix, max_tokens=180)[0]
    if method == "late_layer_injection":
        engine.set_adapter_layer_gate(late_start, -1)
        text = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)[0]
        engine.set_adapter_layer_gate(0, -1)
        return text
    if method == "late_last_quarter":
        engine.set_adapter_layer_gate(late_last_q, -1)
        text = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)[0]
        engine.set_adapter_layer_gate(0, -1)
        return text
    if method == "early_third_only":
        engine.set_adapter_layer_gate(0, early_max)
        text = engine.generate(prompt, routing_weights=w_mix, max_tokens=180)[0]
        engine.set_adapter_layer_gate(0, -1)
        return text
    if method == "sequential_token_segments":
        engine.set_adapter_layer_gate(0, -1)
        return engine.generate_sequential_segments(prompt, [(w_a, 48), (w_b, 132)])[0]
    if method == "sequential_reverse":
        engine.set_adapter_layer_gate(0, -1)
        return engine.generate_sequential_segments(prompt, [(w_b, 48), (w_a, 132)])[0]
    if method == "oracle_single_d1":
        engine.set_adapter_layer_gate(0, -1)
        return engine.generate(prompt, routing_weights=w_a, max_tokens=180)[0]
    if method == "oracle_single_d2":
        engine.set_adapter_layer_gate(0, -1)
        return engine.generate(prompt, routing_weights=w_b, max_tokens=180)[0]
    if method == "merge_high_clamp":
        set_global_clamp(1.0)
        try:
            engine.set_adapter_layer_gate(0, -1)
            return engine.generate(prompt, routing_weights=w_mix, max_tokens=180)[0]
        finally:
            set_global_clamp(0.5)
    raise ValueError(f"Unsupported method: {method}")


def run_tcar_method(
    pipeline,
    item: dict,
    *,
    max_experts: int,
    router_max_tokens: int,
    expert_max_tokens: int,
    refine_max_tokens: int,
):
    return pipeline.run(
        item["question"],
        max_experts=max_experts,
        router_max_tokens=router_max_tokens,
        expert_max_tokens=expert_max_tokens,
        refine_max_tokens=refine_max_tokens,
    )


def run_tcar_oracle_method(
    pipeline,
    item: dict,
    *,
    expert_max_tokens: int,
    refine_max_tokens: int,
):
    experts = item.get("required_adapters") or item.get("domains") or []
    return pipeline.run_with_experts(
        item["question"],
        experts,
        expert_max_tokens=expert_max_tokens,
        refine_max_tokens=refine_max_tokens,
        router_thinking="oracle experts supplied from dataset metadata",
    )


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/md_head_to_head.jsonl")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--methods",
        default="weighted_merge,sequential_reverse,late_layer_injection,mistral",
    )
    parser.add_argument("--mistral-model", default="mistral:7b")
    parser.add_argument("--mistral-num-predict", type=int, default=180)
    parser.add_argument("--ollama-num-gpu", type=int, default=None)
    parser.add_argument("--ollama-num-thread", type=int, default=None)
    parser.add_argument("--ollama-num-batch", type=int, default=None)
    parser.add_argument("--ollama-num-ctx", type=int, default=None)
    parser.add_argument("--live-log", action="store_true")
    parser.add_argument("--debug-log", default="")
    parser.add_argument("--print-answer-chars", type=int, default=0)
    parser.add_argument("--tcar-max-experts", type=int, default=2)
    parser.add_argument("--tcar-router-max-tokens", type=int, default=120)
    parser.add_argument("--tcar-expert-max-tokens", type=int, default=72)
    parser.add_argument("--tcar-refine-max-tokens", type=int, default=220)
    parser.add_argument("--tcar-parallel-workers", type=int, default=1)
    parser.add_argument("--tcar-router-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tcar-router-adapter", default="")
    parser.add_argument("--tcar-router-samples", type=int, default=4)
    parser.add_argument("--tcar-router-temperature", type=float, default=0.7)
    parser.add_argument("--tcar-router-top-p", type=float, default=0.9)
    args = parser.parse_args()

    os.chdir(BACKEND_DIR)
    path, items = prepare_md_items(PROJECT_ROOT, args.data, limit=args.limit, two_domain_only=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    total_jobs = len(items) * len(methods)

    debug_fp: TextIO | None = None
    if args.debug_log:
        debug_path = Path(args.debug_log)
        if not debug_path.is_absolute():
            debug_path = PROJECT_ROOT / debug_path
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_fp = open(debug_path, "a")

    def log_line(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if args.live_log:
            print(line, flush=True)
        if debug_fp is not None:
            debug_fp.write(line + "\n")
            debug_fp.flush()

    ollama_options: dict = {}
    if args.ollama_num_gpu is not None:
        ollama_options["num_gpu"] = args.ollama_num_gpu
    if args.ollama_num_thread is not None:
        ollama_options["num_thread"] = args.ollama_num_thread
    if args.ollama_num_batch is not None:
        ollama_options["num_batch"] = args.ollama_num_batch
    if args.ollama_num_ctx is not None:
        ollama_options["num_ctx"] = args.ollama_num_ctx

    log_line(f"Dataset={path.relative_to(PROJECT_ROOT)} items={len(items)} methods={methods} jobs={total_jobs}")
    log_line("MLX backend note: Qwen path uses MLX GPU backend by default on Apple Silicon.")
    if ollama_options:
        log_line(f"Ollama options={ollama_options}")

    registry = json.load(open("expert_registry.json"))
    from dynamic_mlx_inference import DynamicEngine

    qwen_needed = any(m != "mistral" for m in methods)
    load_start = time.time()
    engine = DynamicEngine("mlx-community/Qwen2.5-1.5B-Instruct-4bit", registry) if qwen_needed else None
    tcar_pipeline = None
    if qwen_needed:
        log_line(f"Qwen engine loaded in {time.time() - load_start:.2f}s")
    if any(m in methods for m in ("tcar_collaborative", "tcar_oracle_collaborative")):
        from collaborative_reasoning import CollaborativeReasoner

        tcar_pipeline = CollaborativeReasoner(
            engine,
            "expert_registry.json",
            parallel_workers=args.tcar_parallel_workers,
            backend_dir=BACKEND_DIR,
            router_model_name_or_path=args.tcar_router_model,
            router_adapter_path=args.tcar_router_adapter or None,
            router_num_samples=args.tcar_router_samples,
            router_temperature=args.tcar_router_temperature,
            router_top_p=args.tcar_router_top_p,
        )
        log_line(
            "TCAR collaborative pipeline ready "
            f"(parallel_workers={args.tcar_parallel_workers}, max_experts={args.tcar_max_experts}, "
            f"router_adapter={'none' if not args.tcar_router_adapter else args.tcar_router_adapter}, "
            f"router_samples={args.tcar_router_samples}, temp={args.tcar_router_temperature}, top_p={args.tcar_router_top_p})"
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    run_start = time.time()
    job_idx = 0
    with open(out_path, "w") as f:
        for item_i, item in enumerate(items, start=1):
            prompt = item["question"]
            ref = item["reference_answer"]
            for method in methods:
                job_idx += 1
                log_line(
                    f"START job={job_idx}/{total_jobs} item={item_i}/{len(items)} "
                    f"id={item['id']} method={method} domains={item.get('required_adapters') or item.get('domains') or []}"
                )
                start = time.time()
                ollama_error = None
                method_meta = None
                if method == "mistral":
                    answer, latency, ollama_error = query_ollama(
                        prompt,
                        args.mistral_model,
                        args.mistral_num_predict,
                        ollama_options or None,
                    )
                elif method == "tcar_collaborative":
                    result = run_tcar_method(
                        tcar_pipeline,
                        item,
                        max_experts=args.tcar_max_experts,
                        router_max_tokens=args.tcar_router_max_tokens,
                        expert_max_tokens=args.tcar_expert_max_tokens,
                        refine_max_tokens=args.tcar_refine_max_tokens,
                    )
                    answer = result.final_answer
                    latency = result.total_latency_s
                    method_meta = {
                        "router_experts": result.router.experts,
                        "router_thinking": result.router.thinking,
                        "router_latency_s": round(result.router_latency_s, 3),
                        "branch_latency_s": round(result.branch_latency_s, 3),
                        "verifier_latency_s": round(result.verifier_latency_s, 3),
                        "parallel_workers": result.parallel_workers,
                        "selected_expert": result.selected_expert,
                        "route_candidate_count": len(result.route_candidates or []),
                        "route_candidates": result.route_candidates or [],
                        "expert_branches": [
                            {
                                "expert": branch.expert,
                                "latency_s": round(branch.latency_s, 3),
                                "mode": branch.mode,
                                "verifier_score": None if branch.verifier_score is None else round(branch.verifier_score, 4),
                                "answer_preview": (branch.answer or "")[:240],
                            }
                            for branch in result.branches
                        ],
                    }
                elif method == "tcar_oracle_collaborative":
                    result = run_tcar_oracle_method(
                        tcar_pipeline,
                        item,
                        expert_max_tokens=args.tcar_expert_max_tokens,
                        refine_max_tokens=args.tcar_refine_max_tokens,
                    )
                    answer = result.final_answer
                    latency = result.total_latency_s
                    method_meta = {
                        "router_experts": result.router.experts,
                        "router_thinking": result.router.thinking,
                        "router_latency_s": round(result.router_latency_s, 3),
                        "branch_latency_s": round(result.branch_latency_s, 3),
                        "verifier_latency_s": round(result.verifier_latency_s, 3),
                        "parallel_workers": result.parallel_workers,
                        "selected_expert": result.selected_expert,
                        "route_candidate_count": len(result.route_candidates or []),
                        "route_candidates": result.route_candidates or [],
                        "expert_branches": [
                            {
                                "expert": branch.expert,
                                "latency_s": round(branch.latency_s, 3),
                                "mode": branch.mode,
                                "verifier_score": None if branch.verifier_score is None else round(branch.verifier_score, 4),
                                "answer_preview": (branch.answer or "")[:240],
                            }
                            for branch in result.branches
                        ],
                    }
                else:
                    answer = run_qwen_method(engine, registry, item, method)
                    latency = time.time() - start
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "dataset_path": str(path.relative_to(PROJECT_ROOT)),
                    "item_id": item["id"],
                    "domains": item.get("required_adapters") or item.get("domains") or [],
                    "method": method,
                    "question": prompt,
                    "reference_answer": ref,
                    "answer": answer,
                    "semantic_sim": round(semantic_similarity(answer, ref), 4),
                    "exact_match": round(exact_match(answer, ref), 4),
                    "token_f1": round(token_f1(answer, ref), 4),
                    "latency_s": round(latency, 3),
                }
                if method_meta:
                    row["method_meta"] = method_meta
                    row["router_latency_s"] = method_meta["router_latency_s"]
                    row["shared_prefill_branch_latency_s"] = method_meta["branch_latency_s"]
                    row["verifier_latency_s"] = method_meta["verifier_latency_s"]
                if ollama_error:
                    row["ollama_error"] = ollama_error
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                elapsed = time.time() - run_start
                avg_job = elapsed / max(1, job_idx)
                eta = avg_job * (total_jobs - job_idx)
                print(
                    f"{item['id']:16s} | {method:22s} | "
                    f"EM={row['exact_match']:.2f} F1={row['token_f1']:.2f} "
                    f"Sim={row['semantic_sim']:.3f} Lat={row['latency_s']:.2f}s"
                )
                latency_breakdown = ""
                if method_meta:
                    latency_breakdown = (
                        f" router={row['router_latency_s']:.2f}s"
                        f" branches={row['shared_prefill_branch_latency_s']:.2f}s"
                        f" verifier={row['verifier_latency_s']:.2f}s"
                    )
                log_line(
                    f"DONE  job={job_idx}/{total_jobs} id={item['id']} method={method} "
                    f"lat={row['latency_s']:.2f}s em={row['exact_match']:.2f} "
                    f"f1={row['token_f1']:.2f} sim={row['semantic_sim']:.3f}"
                    f"{latency_breakdown} avg/job={avg_job:.2f}s eta={_fmt_seconds(eta)}"
                    + (f" ollama_error={ollama_error}" if ollama_error else "")
                )
                if args.print_answer_chars > 0:
                    snippet = (answer or "").replace("\n", " ").strip()[: args.print_answer_chars]
                    log_line(f"ANS   id={item['id']} method={method} preview={snippet}")
    print(f"saved {len(rows)} rows to {out_path}")
    if tcar_pipeline is not None:
        tcar_pipeline.shutdown()
    if debug_fp is not None:
        debug_fp.close()


if __name__ == "__main__":
    main()
