import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TRAINER = ROOT / "advanced_training_pipeline.py"
BENCHMARK_RUNNER = ROOT / "post_dpo_benchmarks.py"
OUTPUT_DIR = ROOT / "outputs"
OFFLOAD_DIR = ROOT / "offload_cache"
DATA_CACHE_DIR = ROOT / "data" / "dpo_cache"
STATE_FILE = ROOT / "autonomous_dpo_state.json"
LOG_FILE = ROOT / "autonomous_dpo.log"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"

MODEL_ORDER = ["qwen_0.8b", "nemotron_4b"]
RANK_ORDER = [1, 2, 8, 128, 1024, 3072]
TARGET_QUEUE = [
    ("qwen_0.8b", 1),
    ("qwen_0.8b", 2),
    ("qwen_0.8b", 8),
    ("qwen_0.8b", 128),
    ("qwen_0.8b", 1024),
    ("qwen_0.8b", 3072),
    ("nemotron_4b", 128),
    ("nemotron_4b", 1024),
    ("nemotron_4b", 3072),
]

SECONDS_AT_10K = {
    "qwen_0.8b": 1500,
    "nemotron_4b": 3200,
}

RANK_TIME_MULTIPLIER = {
    1: 1.00,
    2: 1.00,
    8: 1.00,
    128: 1.08,
    1024: 1.18,
    3072: 1.32,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_seconds(seconds: float) -> str:
    if seconds == float("inf"):
        return "unbounded"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log(message: str):
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_state(state: dict):
    with STATE_FILE.open("w") as handle:
        json.dump(state, handle, indent=2)


def job_name(model_key: str, rank: int) -> str:
    return f"{model_key}_math_DPO_rank{rank}"


def merged_adapter_exists(model_key: str, rank: int) -> bool:
    merged_root = OUTPUT_DIR / f"{model_key}_merged_DARE_rank{rank}"
    return (merged_root / "adapter_config.json").exists() or (merged_root / "merged_sft" / "adapter_config.json").exists()


def job_complete(model_key: str, rank: int) -> bool:
    return (OUTPUT_DIR / job_name(model_key, rank) / "adapter_config.json").exists()


def desired_queue():
    queue = []
    for model_key, rank in TARGET_QUEUE:
        if not merged_adapter_exists(model_key, rank):
            continue
        if job_complete(model_key, rank):
            continue
        queue.append((model_key, rank))
    return queue


def status_snapshot():
    pending = []
    completed = []
    missing_merged = []
    all_targets = []
    for model_key, rank in TARGET_QUEUE:
        name = job_name(model_key, rank)
        all_targets.append(name)
        if not merged_adapter_exists(model_key, rank):
            missing_merged.append(name)
            continue
        if job_complete(model_key, rank):
            completed.append(name)
        else:
            pending.append(name)
    return {
        "targets": all_targets,
        "completed": completed,
        "pending": pending,
        "missing_merged": missing_merged,
        "completed_count": len(completed),
        "pending_count": len(pending),
    }


def estimate_job_seconds(model_key: str, rank: int, dataset_limit: int) -> int:
    base = SECONDS_AT_10K[model_key]
    multiplier = RANK_TIME_MULTIPLIER.get(rank, 1.0)
    return int(base * multiplier * (dataset_limit / 10000))


def planned_dataset_limit(queue, requested_limit: int, seconds_left: float) -> int:
    if not queue:
        return requested_limit

    expected_full = sum(estimate_job_seconds(model_key, rank, requested_limit) for model_key, rank in queue)
    soft_budget = max(1, int(seconds_left * 0.92))
    if expected_full <= soft_budget:
        return requested_limit

    scale = soft_budget / expected_full
    scaled = int(math.floor((requested_limit * scale) / 500.0) * 500)
    return max(2000, min(requested_limit, scaled))


def job_dataset_limit(model_key: str, rank: int, dataset_limit: int) -> int:
    if model_key != "nemotron_4b":
        return dataset_limit
    if rank >= 3072:
        return min(dataset_limit, 2000)
    if rank >= 1024:
        return min(dataset_limit, 2500)
    return min(dataset_limit, 3000)


def dedupe_profiles(profiles):
    seen = set()
    unique = []
    for profile in profiles:
        key = tuple(sorted(profile.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(profile)
    return unique


def job_profiles(model_key: str, rank: int, dataset_limit: int):
    dataset_limit = job_dataset_limit(model_key, rank, dataset_limit)
    if model_key == "qwen_0.8b":
        if rank >= 3072:
            return dedupe_profiles([
                {"dataset_limit": dataset_limit, "effective_batch_size": 16, "batch_size": 2, "max_length": 384, "max_prompt_length": 192},
                {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 1, "max_length": 384, "max_prompt_length": 192},
                {"dataset_limit": dataset_limit, "effective_batch_size": 6, "batch_size": 1, "max_length": 320, "max_prompt_length": 160},
                {"dataset_limit": max(2000, int(dataset_limit * 0.85)), "effective_batch_size": 4, "batch_size": 1, "max_length": 256, "max_prompt_length": 128},
            ])
        if rank >= 1024:
            return dedupe_profiles([
                {"dataset_limit": dataset_limit, "effective_batch_size": 16, "batch_size": 2, "max_length": 512, "max_prompt_length": 256},
                {"dataset_limit": dataset_limit, "effective_batch_size": 12, "batch_size": 1, "max_length": 448, "max_prompt_length": 224},
                {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 1, "max_length": 384, "max_prompt_length": 192},
                {"dataset_limit": max(2000, int(dataset_limit * 0.85)), "effective_batch_size": 4, "batch_size": 1, "max_length": 320, "max_prompt_length": 160},
            ])
        if rank >= 128:
            return dedupe_profiles([
                {"dataset_limit": dataset_limit, "effective_batch_size": 24, "batch_size": 3, "max_length": 640, "max_prompt_length": 384},
                {"dataset_limit": dataset_limit, "effective_batch_size": 16, "batch_size": 2, "max_length": 640, "max_prompt_length": 384},
                {"dataset_limit": dataset_limit, "effective_batch_size": 12, "batch_size": 1, "max_length": 512, "max_prompt_length": 256},
                {"dataset_limit": max(2500, int(dataset_limit * 0.9)), "effective_batch_size": 8, "batch_size": 1, "max_length": 384, "max_prompt_length": 192},
            ])
        return dedupe_profiles([
            {"dataset_limit": dataset_limit, "effective_batch_size": 32, "batch_size": 4, "max_length": 768, "max_prompt_length": 448},
            {"dataset_limit": dataset_limit, "effective_batch_size": 24, "batch_size": 3, "max_length": 640, "max_prompt_length": 384},
            {"dataset_limit": dataset_limit, "effective_batch_size": 16, "batch_size": 2, "max_length": 640, "max_prompt_length": 384},
            {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 1, "max_length": 512, "max_prompt_length": 256},
        ])

    if rank >= 3072:
        return dedupe_profiles([
            {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 2, "max_length": 256, "max_prompt_length": 128},
            {"dataset_limit": dataset_limit, "effective_batch_size": 6, "batch_size": 1, "max_length": 256, "max_prompt_length": 128},
            {"dataset_limit": dataset_limit, "effective_batch_size": 4, "batch_size": 1, "max_length": 224, "max_prompt_length": 112},
            {"dataset_limit": dataset_limit, "effective_batch_size": 6, "batch_size": 1, "max_length": 256, "max_prompt_length": 128},
        ])
    if rank >= 1024:
        return dedupe_profiles([
            {"dataset_limit": dataset_limit, "effective_batch_size": 12, "batch_size": 2, "max_length": 320, "max_prompt_length": 160},
            {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 2, "max_length": 256, "max_prompt_length": 128},
            {"dataset_limit": dataset_limit, "effective_batch_size": 6, "batch_size": 1, "max_length": 256, "max_prompt_length": 128},
            {"dataset_limit": dataset_limit, "effective_batch_size": 4, "batch_size": 1, "max_length": 224, "max_prompt_length": 112},
        ])
    return dedupe_profiles([
        {"dataset_limit": dataset_limit, "effective_batch_size": 12, "batch_size": 3, "max_length": 384, "max_prompt_length": 192},
        {"dataset_limit": dataset_limit, "effective_batch_size": 16, "batch_size": 2, "max_length": 384, "max_prompt_length": 192},
        {"dataset_limit": dataset_limit, "effective_batch_size": 8, "batch_size": 2, "max_length": 320, "max_prompt_length": 160},
        {"dataset_limit": dataset_limit, "effective_batch_size": 4, "batch_size": 1, "max_length": 256, "max_prompt_length": 128},
    ])


def ensure_log_header(args, queue):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        with LOG_FILE.open("a") as handle:
            handle.write("\n")
    log("Autonomous DPO pipeline starting")
    log(f"Python: {resolve_python()}")
    log(f"Hours budget: {args.hours}")
    log(f"Run post-train evals: {args.run_evals} (mode={args.eval_mode})")
    log(f"Initial queue: {[job_name(model, rank) for model, rank in queue]}")


def resolve_python() -> str:
    if PYTHON_BIN.exists():
        return str(PYTHON_BIN)
    return sys.executable


def create_state(args, queue):
    deadline = time.time() + int(args.hours * 3600)
    state = {
        "started_at": now_utc(),
        "deadline_at": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
        "config": {
            "hours": args.hours,
            "requested_dataset_limit": args.dataset_limit,
            "output_dir": str(OUTPUT_DIR),
            "offload_dir": str(OFFLOAD_DIR),
            "data_cache_dir": str(DATA_CACHE_DIR),
            "python": resolve_python(),
        },
        "queue": [job_name(model, rank) for model, rank in queue],
        "jobs": {},
    }
    write_state(state)
    return state, deadline


def ensure_job_state(state: dict, model_key: str, rank: int):
    key = job_name(model_key, rank)
    jobs = state.setdefault("jobs", {})
    if key not in jobs:
        jobs[key] = {
            "model": model_key,
            "rank": rank,
            "status": "pending",
            "attempts": [],
        }
    return jobs[key]


def spawn_job(model_key: str, rank: int, profile: dict):
    cmd = [
        resolve_python(),
        "-u",
        str(TRAINER),
        "--models",
        model_key,
        "--ranks",
        str(rank),
        "--dataset_limit",
        str(profile["dataset_limit"]),
        "--effective_batch_size",
        str(profile["effective_batch_size"]),
        "--batch_size",
        str(profile["batch_size"]),
        "--max_length",
        str(profile["max_length"]),
        "--max_prompt_length",
        str(profile["max_prompt_length"]),
        "--output_dir",
        str(OUTPUT_DIR),
        "--offload_dir",
        str(OFFLOAD_DIR),
        "--data_cache_dir",
        str(DATA_CACHE_DIR),
    ]
    log(f"Launching {job_name(model_key, rank)} with profile {profile}")
    log(f"Command: {' '.join(cmd)}")
    handle = LOG_FILE.open("a")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle, LOG_FILE.stat().st_size if LOG_FILE.exists() else 0


def read_log_slice(start_offset: int) -> str:
    if not LOG_FILE.exists():
        return ""
    with LOG_FILE.open("rb") as handle:
        handle.seek(start_offset)
        return handle.read().decode("utf-8", errors="replace")


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    oom_markers = [
        "out of memory",
        "cuda out of memory",
        "⚠️ oom",
        "oom:",
    ]
    if any(marker in lowered for marker in oom_markers):
        return "oom"
    if "killed" in lowered or "sigkill" in lowered:
        return "killed"
    if "traceback" in lowered or "❌ error" in lowered:
        return "crash"
    return "unknown"


def monitor_process(process: subprocess.Popen, deadline: float, heartbeat_label: str, log_offset: int, stall_seconds: int):
    last_heartbeat = 0.0
    last_log_size = log_offset
    last_log_progress = time.time()
    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code, read_log_slice(log_offset)
        now = time.time()
        if LOG_FILE.exists():
            current_size = LOG_FILE.stat().st_size
            if current_size > last_log_size:
                last_log_size = current_size
                last_log_progress = now
        if now - last_log_progress >= stall_seconds:
            log(f"Watchdog killing stalled job={heartbeat_label} after {stall_seconds}s without log growth")
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return -9, read_log_slice(log_offset)
        if now - last_heartbeat >= 300:
            seconds_left = deadline - now
            log(f"Heartbeat | job={heartbeat_label} | time_left={fmt_seconds(seconds_left)}")
            last_heartbeat = now
        time.sleep(30)


def run_queue(args):
    queue = desired_queue()
    ensure_log_header(args, queue)
    state, deadline = create_state(args, queue)

    if not queue:
        log("No pending DPO jobs. Everything requested is already complete.")
        return 0

    for index, (model_key, rank) in enumerate(queue, start=1):
        if job_complete(model_key, rank):
            continue

        remaining_queue = [(m, r) for m, r in queue[index - 1:] if not job_complete(m, r)]
        seconds_left = deadline - time.time()
        if seconds_left <= 0:
            log("Time budget exhausted before starting the next job.")
            break

        dataset_limit = planned_dataset_limit(remaining_queue, args.dataset_limit, seconds_left)
        job_state = ensure_job_state(state, model_key, rank)
        job_state["status"] = "running"
        job_state["started_at"] = now_utc()
        job_state["planned_dataset_limit"] = dataset_limit
        write_state(state)

        profiles = job_profiles(model_key, rank, dataset_limit)
        success = False
        for attempt_index, profile in enumerate(profiles, start=1):
            attempt_state = {
                "attempt": attempt_index,
                "started_at": now_utc(),
                "profile": profile,
                "status": "running",
            }
            job_state["attempts"].append(attempt_state)
            write_state(state)

            process, handle, log_offset = spawn_job(model_key, rank, profile)
            return_code = None
            attempt_log = ""
            try:
                return_code, attempt_log = monitor_process(
                    process,
                    deadline,
                    job_name(model_key, rank),
                    log_offset,
                    args.stall_minutes * 60,
                )
            finally:
                handle.close()

            attempt_state["finished_at"] = now_utc()
            attempt_state["return_code"] = return_code
            attempt_state["failure_type"] = None if return_code == 0 and job_complete(model_key, rank) else classify_failure(attempt_log)
            attempt_state["status"] = "success" if return_code == 0 and job_complete(model_key, rank) else "failed"
            write_state(state)

            if return_code == 0 and job_complete(model_key, rank):
                duration = read_json(OUTPUT_DIR / job_name(model_key, rank) / "timing.json", {}).get("duration_seconds")
                job_state["status"] = "completed"
                job_state["completed_at"] = now_utc()
                if duration is not None:
                    job_state["duration_seconds"] = duration
                write_state(state)
                log(f"Completed {job_name(model_key, rank)}")
                success = True
                break

            if job_complete(model_key, rank):
                duration = read_json(OUTPUT_DIR / job_name(model_key, rank) / "timing.json", {}).get("duration_seconds")
                job_state["status"] = "completed"
                job_state["completed_at"] = now_utc()
                if duration is not None:
                    job_state["duration_seconds"] = duration
                write_state(state)
                log(f"Completed {job_name(model_key, rank)} despite non-zero exit from child process")
                success = True
                break

            log(
                f"Attempt {attempt_index} failed for {job_name(model_key, rank)} "
                f"(failure_type={attempt_state['failure_type']}, return_code={return_code})"
            )

        if not success:
            job_state["status"] = "failed"
            job_state["failed_at"] = now_utc()
            write_state(state)
            log(f"Marked failed after retries: {job_name(model_key, rank)}")

    pending = [name for name in state["queue"] if not (OUTPUT_DIR / name / "adapter_config.json").exists()]
    state["finished_at"] = now_utc()
    state["pending_after_run"] = pending
    write_state(state)
    log(f"Run finished. Pending after run: {pending}")

    if args.run_evals:
        state["post_eval"] = {"status": "running", "started_at": now_utc(), "mode": args.eval_mode}
        write_state(state)
        eval_cmd = [
            resolve_python(),
            "-u",
            str(BENCHMARK_RUNNER),
            "--mode",
            args.eval_mode,
        ]
        log(f"Launching post-DPO benchmark suite: {' '.join(eval_cmd)}")
        handle = LOG_FILE.open("a")
        process = subprocess.Popen(
            eval_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            eval_log_offset = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
            return_code, _ = monitor_process(
                process,
                float("inf"),
                f"post_dpo_eval[{args.eval_mode}]",
                eval_log_offset,
                max(30 * 60, args.stall_minutes * 60),
            )
        finally:
            handle.close()
        state["post_eval"]["finished_at"] = now_utc()
        state["post_eval"]["return_code"] = return_code
        state["post_eval"]["status"] = "completed" if return_code == 0 else "failed"
        write_state(state)
        log(f"Post-DPO benchmark suite finished with return_code={return_code}")
        if return_code != 0:
            return 1

    return 0 if not pending else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous 8-hour DPO scheduler.")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--dataset_limit", type=int, default=10000)
    parser.add_argument("--skip_evals", action="store_true")
    parser.add_argument("--eval_mode", choices=["autonomous", "quick", "paper"], default="autonomous")
    parser.add_argument("--stall_minutes", type=int, default=15)
    parser.add_argument("--status_json", action="store_true")
    args = parser.parse_args()
    args.run_evals = not args.skip_evals
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.status_json:
        print(json.dumps(status_snapshot(), indent=2))
        raise SystemExit(0)
    raise SystemExit(run_queue(args))
