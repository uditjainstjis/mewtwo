import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from post_dpo_benchmarks import (
    BASE_RANK,
    MODEL_SPECS,
    RESULTS_DIR,
    RANKS,
    STAGE_ORDER,
    cleanup,
    eval_targets,
    format_chat_prompt,
    generate_responses,
    hf_token,
    load_model_and_tokenizer,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
PLANBENCH_ROOT = ROOT / "benchmarks" / "LLMs-Planning" / "plan-bench"
AGENTIC_RESULTS_DIR = RESULTS_DIR / "agentic_eval"
PLANBENCH_RESULTS_JSON = AGENTIC_RESULTS_DIR / "planbench_results.json"
PLANBENCH_SUMMARY_MD = AGENTIC_RESULTS_DIR / "planbench_summary.md"
PLANBENCH_LOCK = AGENTIC_RESULTS_DIR / "planbench_results.lock"
PLANBENCH_GENERATIONS_DIR = AGENTIC_RESULTS_DIR / "benchmark_generations"

MODEL_ORDER = ["qwen_0.8b", "nemotron_4b"]
ALL_EVAL_RANKS = [BASE_RANK] + RANKS

TASK_METADATA = {
    "t1": {
        "task_file": "task_1_plan_generation",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t2": {
        "task_file": "task_2_plan_optimality",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t4": {
        "task_file": "task_4_plan_reuse",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t5": {
        "task_file": "task_5_plan_generalization",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t6": {
        "task_file": "task_6_replanning",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t7": {
        "task_file": "task_7_plan_execution",
        "kind": "state",
        "max_new_tokens": 256,
        "system_prompt": (
            "Answer using only the final state facts requested by the benchmark. Do not add explanations."
        ),
    },
    "t8_1": {
        "task_file": "task_8_1_goal_shuffling",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t8_2": {
        "task_file": "task_8_2_full_to_partial",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
    "t8_3": {
        "task_file": "task_8_3_partial_to_full",
        "kind": "plan",
        "max_new_tokens": 256,
        "system_prompt": (
            "You are a precise classical planning assistant. Continue from [PLAN] with only action lines, "
            "one action per line, and end with [PLAN END]. Do not explain your reasoning."
        ),
    },
}


def planbench_key(model_key: str, rank: int, stage: str, domain: str, task: str) -> str:
    return f"model={model_key}|rank={rank}|stage={stage}|domain={domain}|task={task}"


def load_results():
    if not PLANBENCH_RESULTS_JSON.exists():
        return {"benchmarks": {}}
    try:
        return json.loads(PLANBENCH_RESULTS_JSON.read_text())
    except Exception:
        return {"benchmarks": {}}


def save_results(results: dict):
    AGENTIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLANBENCH_RESULTS_JSON.write_text(json.dumps(results, indent=2))


def update_results_entry(key: str, outcome: dict):
    AGENTIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PLANBENCH_LOCK.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        results = load_results()
        results.setdefault("benchmarks", {})[key] = outcome
        save_results(results)
        write_summary(results)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return results


def write_summary(results: dict):
    lines = [
        "# PlanBench Summary",
        "",
        "| Model | Rank | Stage | Domain | Task | Score | 95% CI | Correct / Total |",
        "|-------|------|-------|--------|------|-------|---------|-----------------|",
    ]
    grouped = results.get("benchmarks", {})
    for model_key in MODEL_ORDER:
        for rank in ALL_EVAL_RANKS:
            for stage in STAGE_ORDER:
                for key, value in sorted(grouped.items()):
                    parts = dict(part.split("=", 1) for part in key.split("|"))
                    if (
                        parts["model"] != model_key
                        or int(parts["rank"]) != rank
                        or parts["stage"] != stage
                    ):
                        continue
                    score = value.get("score")
                    ci_low = value.get("ci95_low")
                    ci_high = value.get("ci95_high")
                    ci_str = "-" if ci_low is None or ci_high is None else f"[{ci_low:.3f}, {ci_high:.3f}]"
                    score_str = "-" if score is None else f"{score:.3f}"
                    lines.append(
                        f"| {model_key} | {rank} | {stage} | {parts['domain']} | {parts['task']} | "
                        f"{score_str} | {ci_str} | {value.get('correct', 0)} / {value.get('total', 0)} |"
                    )
    PLANBENCH_SUMMARY_MD.write_text("\n".join(lines) + "\n")


def available_targets(models: list[str], ranks: list[int]):
    targets = []
    for model_key, rank, stages in eval_targets():
        if model_key not in models or rank not in ranks:
            continue
        filtered = {stage_name: adapter_path for stage_name, adapter_path in stages.items() if stage_name in STAGE_ORDER}
        if filtered:
            targets.append((model_key, rank, filtered))
    return targets


def sanitize_engine_name(model_key: str, rank: int, stage_name: str) -> str:
    return f"{model_key.replace('.', '_')}_rank{rank}_{stage_name}"


def prompt_path(domain: str, task_file: str) -> Path:
    return PLANBENCH_ROOT / "prompts" / domain / f"{task_file}.json"


def response_path(domain_name: str, engine_name: str, task_file: str) -> Path:
    return PLANBENCH_ROOT / "responses" / domain_name / engine_name / f"{task_file}.json"


def scored_path(domain_name: str, engine_name: str, task_file: str) -> Path:
    return PLANBENCH_ROOT / "results" / domain_name / engine_name / f"{task_file}.json"


def generation_path(model_key: str, rank: int, stage: str, domain: str, task: str) -> Path:
    filename = f"planbench_{domain}_{task}_{model_key}_rank{rank}_{stage}.jsonl"
    return PLANBENCH_GENERATIONS_DIR / filename


def should_run(existing_result: dict | None, desired_total: int, generation_target: Path, scored_target: Path) -> bool:
    if not existing_result:
        return True
    if existing_result.get("metric") == "error":
        return True
    if int(existing_result.get("total", 0) or 0) < desired_total:
        return True
    if not generation_target.exists() or not scored_target.exists():
        return True
    return False


def select_instances(prompt_data: dict, limit: int | None, instance_ids: list[int] | None):
    instances = prompt_data["instances"]
    if instance_ids:
        wanted = set(instance_ids)
        instances = [row for row in instances if int(row["instance_id"]) in wanted]
    if limit is not None:
        instances = instances[:limit]
    return instances


def normalize_plan_response(text: str) -> str:
    text = (text or "").strip()
    if "[PLAN END]" not in text:
        text = text.rstrip() + "\n[PLAN END]"
    return text + ("\n" if not text.endswith("\n") else "")


def write_generations(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_planbench_responses(
    model,
    tok,
    model_key: str,
    rank: int,
    stage_name: str,
    domain: str,
    domain_name: str,
    task: str,
    instances: list[dict],
):
    task_meta = TASK_METADATA[task]
    if model_key == "qwen_0.8b":
        batch_size = max(1, min(int(MODEL_SPECS[model_key]["eval_batch_size"]), 32))
    else:
        batch_size = max(1, min(int(MODEL_SPECS[model_key]["eval_batch_size"]), 8))
    prompts = [
        format_chat_prompt(tok, task_meta["system_prompt"], row["query"])
        for row in instances
    ]
    responses = []
    for start in range(0, len(prompts), batch_size):
        responses.extend(
            generate_responses(
                model,
                tok,
                prompts[start : start + batch_size],
                max_new_tokens=task_meta["max_new_tokens"],
                prompt_max_length=MODEL_SPECS[model_key]["prompt_max_length"],
            )
        )

    engine_name = sanitize_engine_name(model_key, rank, stage_name)
    response_rows = []
    generation_rows = []
    for row, response in zip(instances, responses):
        instance_row = dict(row)
        normalized = normalize_plan_response(response)
        instance_row["llm_raw_response"] = normalized
        response_rows.append(instance_row)
        generation_rows.append(
            {
                "instance_id": row["instance_id"],
                "question": row["query"],
                "gold_answer": row.get("ground_truth_plan"),
                "generated_text": normalized,
            }
        )

    payload = {
        "task": task,
        "engine": engine_name,
        "prompt_type": instances[0].get("prompt_type", "oneshot") if instances else "oneshot",
        "domain": domain,
        "instances": response_rows,
    }
    out_path = response_path(domain_name, engine_name, task_meta["task_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return engine_name, out_path, generation_rows


def run_official_evaluation(task: str, config_name: str, engine_name: str, instance_ids: list[int]):
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    env["VAL"] = str((PLANBENCH_ROOT.parent / "planner_tools" / "VAL").resolve())
    if hf_token():
        env.setdefault("HF_TOKEN", hf_token())

    venv_python = ROOT.parent / ".venv" / "bin" / "python"
    cmd = [
        str(venv_python),
        "response_evaluation.py",
        "--task",
        task,
        "--engine",
        engine_name,
        "--config",
        config_name,
        "--ignore_existing",
    ]
    if instance_ids:
        cmd.extend(["--specific_instances", *[str(x) for x in instance_ids]])
    subprocess.run(cmd, cwd=PLANBENCH_ROOT, env=env, check=True)


def summarize_scored_result(task: str, scored_file: Path):
    payload = json.loads(scored_file.read_text())
    correct_field = "llm_correct"
    task_kind = TASK_METADATA[task]["kind"]
    if task_kind == "verification":
        correct_field = "llm_correct_binary"

    counted_rows = [row for row in payload["instances"] if correct_field in row]
    correct = sum(1 for row in counted_rows if bool(row.get(correct_field)))
    total = len(counted_rows)
    ci_low, ci_high = wilson_interval(correct, total)
    generations = []
    for row in counted_rows:
        generations.append(
            {
                "instance_id": row.get("instance_id"),
                "question": row.get("query"),
                "gold_answer": row.get("ground_truth_plan"),
                "generated_text": row.get("llm_raw_response"),
                "extracted_answer": row.get("extracted_llm_plan"),
                "correct": bool(row.get(correct_field)),
            }
        )
    return {
        "metric": "accuracy",
        "score": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "generations": generations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen/Nemotron adapters on PlanBench with official evaluation.")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=["qwen_0.8b"])
    parser.add_argument("--ranks", nargs="+", type=int, default=ALL_EVAL_RANKS)
    parser.add_argument("--task", choices=sorted(TASK_METADATA.keys()), default="t1")
    parser.add_argument("--domain", default="blocksworld_3")
    parser.add_argument("--config", default="blocksworld_3")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--instance-ids", nargs="+", type=int, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    AGENTIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLANBENCH_GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)

    task_meta = TASK_METADATA[args.task]
    prompt_file = prompt_path(args.domain, task_meta["task_file"])
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt_data = json.loads(prompt_file.read_text())
    selected_instances = select_instances(prompt_data, args.limit, args.instance_ids)
    selected_ids = [int(row["instance_id"]) for row in selected_instances]

    results = load_results()
    targets = available_targets(args.models, args.ranks)
    for model_key, rank, stages in targets:
        for stage_name in STAGE_ORDER:
            if stage_name not in stages:
                continue
            adapter_path = stages[stage_name]
            key = planbench_key(model_key, rank, stage_name, args.domain, args.task)
            engine_name = sanitize_engine_name(model_key, rank, stage_name)
            gen_target = generation_path(model_key, rank, stage_name, args.domain, args.task)
            score_target = scored_path(args.config, engine_name, task_meta["task_file"])
            if not should_run(results.get("benchmarks", {}).get(key), len(selected_ids), gen_target, score_target):
                continue

            model = None
            started = time.time()
            try:
                model, tok = load_model_and_tokenizer(
                    model_key=model_key,
                    adapter_path=adapter_path,
                    offload_name=f"planbench_{model_key}_rank{rank}_{stage_name}",
                )
                print(f"Running {key} on {len(selected_ids)} PlanBench instances", flush=True)
                _, _, raw_generations = generate_planbench_responses(
                    model=model,
                    tok=tok,
                    model_key=model_key,
                    rank=rank,
                    stage_name=stage_name,
                    domain=args.domain,
                    domain_name=args.config,
                    task=args.task,
                    instances=selected_instances,
                )
                run_official_evaluation(args.task, args.config, engine_name, selected_ids)
                outcome = summarize_scored_result(args.task, score_target)
                write_generations(gen_target, outcome.pop("generations"))
                outcome.update(
                    {
                        "model": model_key,
                        "rank": rank,
                        "stage": stage_name,
                        "domain": args.domain,
                        "config": args.config,
                        "task": args.task,
                        "task_file": task_meta["task_file"],
                        "samples": len(selected_ids),
                        "adapter_path": None if adapter_path is None else str(adapter_path),
                        "response_file": str(response_path(args.config, engine_name, task_meta["task_file"])),
                        "scored_file": str(score_target),
                        "generation_file": str(gen_target),
                        "duration_seconds": time.time() - started,
                        "engine_name": engine_name,
                        "preview_generations": raw_generations[:3],
                    }
                )
            except Exception as exc:
                outcome = {
                    "metric": "error",
                    "score": None,
                    "correct": 0,
                    "total": 0,
                    "model": model_key,
                    "rank": rank,
                    "stage": stage_name,
                    "domain": args.domain,
                    "config": args.config,
                    "task": args.task,
                    "task_file": task_meta["task_file"],
                    "samples": len(selected_ids),
                    "adapter_path": None if adapter_path is None else str(adapter_path),
                    "response_file": str(response_path(args.config, engine_name, task_meta["task_file"])),
                    "scored_file": str(score_target),
                    "generation_file": str(gen_target),
                    "duration_seconds": time.time() - started,
                    "engine_name": engine_name,
                    "error": str(exc),
                    "ci95_low": None,
                    "ci95_high": None,
                }
            finally:
                cleanup(model=model)
            results = update_results_entry(key, outcome)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
